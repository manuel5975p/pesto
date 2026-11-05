#include "dns/fvturb.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <format>

namespace dns {

namespace {

// The box at every centroid whose window is nonzero; w per cell.
std::array<std::vector<double>, 3> centroidField(const FaceMesh& fm, const FvTurb& t, std::vector<double>& w,
                                                 int threads)
{
    const std::size_t nC = std::size_t(fm.nC);
    std::vector<std::array<double, 3>> pts(nC);
    std::vector<std::uint8_t> skip(nC, 0);
    w.resize(nC);
    for (std::size_t c = 0; c < nC; ++c) {
        pts[c] = {fm.C[0][c], fm.C[1][c], fm.C[2][c]};
        w[c] = turbWindow(fm.C[0][c], t.xw, t.ew);
        skip[c] = w[c] == 0.0 ? 1 : 0;
    }
    return turbAt(t.modes, t.spec, pts, skip, threads);
}

} // namespace

void fvAddTurbulence(const FaceMesh& fm, const FvTurb& t, FvState& s, int threads)
{
    assert(s.nC == fm.nC && s.nV >= kFvMeanVars);
    std::vector<double> w;
    const auto box = centroidField(fm, t, w, threads);
    double *rho = s.var(0), *mx = s.var(1), *my = s.var(2), *mz = s.var(3), *E = s.var(4);
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const double r = rho[c];
        const double u0 = mx[c] / r, v0 = my[c] / r, w0 = mz[c] / r;
        const double u = u0 + w[c] * box[0][c], v = v0 + w[c] * box[1][c], ww = w0 + w[c] * box[2][c];
        E[c] += 0.5 * r * ((u * u + v * v + ww * ww) - (u0 * u0 + v0 * v0 + w0 * w0));
        mx[c] = r * u;
        my[c] = r * v;
        mz[c] = r * ww;
    }
}

void fvLowMachAddTurbulence(const FaceMesh& fm, const FvTurb& t, FvLowMachIC& ic, int threads)
{
    const std::size_t nC = std::size_t(fm.nC);
    assert(ic.rho.size() == nC && ic.ru.size() == 3 * nC);
    std::vector<double> w;
    const auto box = centroidField(fm, t, w, threads);
    for (std::size_t k = 0; k < 3; ++k)
        for (std::size_t c = 0; c < nC; ++c) ic.ru[k * nC + c] += ic.rho[c] * (w[c] * box[k][c]);
}

std::expected<FvPatchFaces, std::string> fvPatchFaces(const FaceMesh& fm, std::string_view name)
{
    const auto it = std::ranges::find(fm.patches, name, &MeshPatch::name);
    if (it == fm.patches.end()) return std::unexpected(std::format("--turb: no patch '{}' on the mesh", name));
    FvPatchFaces o;
    for (int i = 0; i < it->count; ++i) {
        const int b = it->first + i;
        const std::size_t f = std::size_t(fm.nI + b);
        o.b.push_back(b);
        o.cf.push_back({fm.Cf[0][f], fm.Cf[1][f], fm.Cf[2][f]});
    }
    return o;
}

FvBc fvFarTurbRow(const FvPatchSpec& p, const FvFreestream& f, const std::array<double, 3>& du)
{
    assert(p.kind == FvPatchKind::Far);
    // fvBcOf's Far branch at frac 1, the velocity u dir[k] + du[k].
    FvBc b;
    const double u = 1.0 * f.U;
    std::ranges::copy(p.y, b.y);
    b.kind = FvBcKind::Far;
    b.qFar[0] = f.rho;
    double ke = 0.0;
    for (std::size_t k = 0; k < 3; ++k) {
        const double uk = u * f.dir[k] + du[k];
        const double m = f.rho * uk;
        b.qFar[1 + k] = m;
        ke += m * uk;
    }
    b.qFar[4] = f.p / (f.gamma - 1.0) + 0.5 * ke;
    return b;
}

void fvLowMachTurbRows(const std::vector<int>& b, const std::array<std::vector<double>, 3>& du,
                       const std::vector<double>& bU0, const std::vector<double>& rUb0, const std::vector<double>& bT,
                       double prGas, std::vector<double>& bU, std::vector<double>& rUb)
{
    const std::size_t nB = bT.size();
    assert(bU0.size() == 3 * nB && rUb0.size() == 3 * nB);
    bU = bU0;
    rUb = rUb0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        const std::size_t q = std::size_t(b[i]);
        const double rho = prGas / bT[q];
        for (std::size_t c = 0; c < 3; ++c) {
            bU[c * nB + q] = bU0[c * nB + q] + du[c][i];
            rUb[c * nB + q] = rUb0[c * nB + q] + rho * du[c][i];
        }
    }
}

std::expected<FvFaceRows, std::string> fvFaceRows(const FaceMesh& fm, std::vector<FvBc>& table,
                                                  std::string_view name)
{
    auto in = fvPatchFaces(fm, name);
    if (!in) return std::unexpected(in.error());
    FvFaceRows r{.bcRow = std::vector<int>(std::size_t(fm.nB)), .first = int(table.size()), .faces = std::move(*in)};
    for (std::size_t p = 0; p < fm.patches.size(); ++p)
        for (int i = 0; i < fm.patches[p].count; ++i) r.bcRow[std::size_t(fm.patches[p].first + i)] = int(p);
    const std::size_t row = std::size_t(std::ranges::find(fm.patches, name, &MeshPatch::name) - fm.patches.begin());
    for (const int b : r.faces.b) {
        r.bcRow[std::size_t(b)] = int(table.size());
        table.push_back(table[row]);
    }
    return r;
}

} // namespace dns
