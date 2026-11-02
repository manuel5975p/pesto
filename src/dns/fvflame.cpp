#include "dns/fvflame.hpp"

#include "dns/flame1d.hpp"
#include "dns/fvcomp_case.hpp"

#include <algorithm>
#include <cmath>
#include <format>

namespace dns {
namespace {

double along(const std::array<std::vector<double>, 3>& v, std::size_t i, const std::array<double, 3>& dir)
{
    return v[0][i] * dir[0] + v[1][i] * dir[1] + v[2][i] * dir[2];
}

// (P_i - origin) . a.
double offset(const std::array<std::vector<double>, 3>& P, std::size_t i, const std::array<double, 3>& origin,
              const std::array<double, 3>& a)
{
    return (P[0][i] - origin[0]) * a[0] + (P[1][i] - origin[1]) * a[1] + (P[2][i] - origin[2]) * a[2];
}

// Mean |d.a| over the internal faces with |d.a| >= |d|/2, 0 if none.
double meanSpacing(const FaceMesh& fm, const std::array<double, 3>& a)
{
    double sum = 0;
    long n = 0;
    for (std::size_t f = 0; f < std::size_t(fm.nI); ++f) {
        const double v = std::abs(along(fm.d, f, a));
        if (2.0 * v < std::hypot(fm.d[0][f], fm.d[1][f], fm.d[2][f])) continue;
        sum += v;
        ++n;
    }
    return n ? sum / double(n) : 0.0;
}

} // namespace

std::expected<FvFlameProbe, std::string> FvFlameProbe::make(const FaceMesh& fm, const FvFlameSpec& spec)
{
    FvFlameProbe p;
    p.fm_ = &fm;
    p.spec_ = spec;
    const std::size_t nI = std::size_t(fm.nI), nF = nI + std::size_t(fm.nB);
    double sum = 0;
    long n = 0;
    for (std::size_t f = 0; f < nI; ++f) {
        const double a = std::abs(along(fm.d, f, spec.dir));
        if (2.0 * a < std::hypot(fm.d[0][f], fm.d[1][f], fm.d[2][f])) continue;
        sum += a;
        ++n;
    }
    if (n == 0) return std::unexpected("flame probe: no face of the mesh crosses the flame direction");
    // The low and high ends: active faces only (a cyclic Cf sits in either frame,
    // but a cyclic normal to the flame has no extent along it).
    p.lo_ = HUGE_VAL;
    p.hi_ = -HUGE_VAL;
    for (std::size_t f = 0; f < nF; ++f) {
        const bool normal = std::abs(along(fm.S, f, spec.dir)) * 1e-9 > 0.0 || f < nI;
        if (!normal) continue;
        const double s = along(fm.Cf, f, spec.dir);
        p.lo_ = std::min(p.lo_, s);
        p.hi_ = std::max(p.hi_, s);
    }
    const double L = p.hi_ - p.lo_;
    p.nBin_ = int(std::lround(L * double(n) / sum));
    if (p.nBin_ < 2) return std::unexpected(std::format("flame probe: {} slabs along the flame direction", p.nBin_));
    p.h_ = L / p.nBin_;
    double vol = 0;
    p.bin_.resize(std::size_t(fm.nC));
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        vol += fm.V[c];
        p.bin_[c] = std::clamp(int(std::floor((along(fm.C, c, spec.dir) - p.lo_) / p.h_)), 0, p.nBin_ - 1);
    }
    p.area_ = vol / L;
    return p;
}

FvFlameMeasure FvFlameProbe::measure(const FvState& st, const std::vector<double>& omega) const
{
    const FaceMesh& m = *fm_;
    const FvFlameSpec& sp = spec_;
    FvFlameMeasure o;
    o.yMin = HUGE_VAL;
    o.yMax = -HUGE_VAL;
    std::vector<FvFlameSlab> b(static_cast<std::size_t>(nBin_));
    for (std::size_t c = 0; c < std::size_t(m.nC); ++c) {
        const FvPrim<double> p =
            fvPrimOf(st.var(0)[c], st.var(1)[c], st.var(2)[c], st.var(3)[c], st.var(4)[c], sp.gamma, sp.Rgas);
        const double ry = st.var(kFvMeanVars + sp.fuel)[c], Y = ry / p.rho, V = m.V[c];
        FvFlameSlab& k = b[std::size_t(bin_[c])];
        k.vol += V;
        k.s += V * along(m.C, c, sp.dir);
        k.T += V * p.temp;
        k.rho += V * p.rho;
        k.u += V * (p.u[0] * sp.dir[0] + p.u[1] * sp.dir[1] + p.u[2] * sp.dir[2]);
        k.p += V * p.p;
        k.ry += V * ry;
        k.omega += V * omega[c];
        o.mass += V * p.rho;
        o.energy += V * (st.var(4)[c] + sp.q * ry);
        o.fuel += V * ry;
        o.burn += V * omega[c];
        o.hits += (Y < -1e-6 || Y > 1.0 + 1e-6) ? 1.0 : 0.0;
        o.yMin = std::min(o.yMin, Y);
        o.yMax = std::max(o.yMax, Y);
    }
    finish(o, std::move(b));
    return o;
}

FvFlameMeasure FvFlameProbe::measureLowMach(const FvFlameLowMachCells& q) const
{
    const FaceMesh& m = *fm_;
    const FvFlameSpec& sp = spec_;
    const std::size_t nC = std::size_t(m.nC);
    const double cp = sp.gamma / (sp.gamma - 1.0) * sp.Rgas;
    FvFlameMeasure o;
    o.yMin = HUGE_VAL;
    o.yMax = -HUGE_VAL;
    std::vector<FvFlameSlab> b(static_cast<std::size_t>(nBin_));
    for (std::size_t c = 0; c < nC; ++c) {
        const double rho = (*q.rho)[c], ry = (*q.ry)[c], om = (*q.omega)[c], V = m.V[c], T = (*q.T)[c], Y = ry / rho;
        const double mn = (*q.ru)[c] * sp.dir[0] + (*q.ru)[nC + c] * sp.dir[1] + (*q.ru)[2 * nC + c] * sp.dir[2];
        FvFlameSlab& k = b[std::size_t(bin_[c])];
        k.vol += V;
        k.s += V * along(m.C, c, sp.dir);
        k.T += V * T;
        k.rho += V * rho;
        k.u += V * (mn / rho);
        k.p += V * (*q.p)[c];
        k.ry += V * ry;
        k.omega += V * om;
        o.mass += V * rho;
        o.energy += V * (cp * rho * T + sp.q * ry);
        o.fuel += V * ry;
        o.burn += V * om;
        o.hits += (Y < -1e-6 || Y > 1.0 + 1e-6) ? 1.0 : 0.0;
        o.yMin = std::min(o.yMin, Y);
        o.yMax = std::max(o.yMax, Y);
    }
    finish(o, std::move(b));
    return o;
}

void FvFlameProbe::finish(FvFlameMeasure& o, std::vector<FvFlameSlab> b) const
{
    const FvFlameSpec& sp = spec_;
    std::erase_if(b, [](const FvFlameSlab& k) { return k.vol == 0.0; });
    for (FvFlameSlab& k : b) {
        for (double* f : {&k.s, &k.T, &k.rho, &k.u, &k.p, &k.ry, &k.omega}) *f /= k.vol;
        k.s -= lo_;
    }
    const int n = int(b.size());
    const auto at = [&](int i) -> const FvFlameSlab& { return b[std::size_t(i)]; };
    const double Tmid = 0.5 * (1.0 + sp.sigma) * sp.Tu, dT = (sp.sigma - 1.0) * sp.Tu;
    double wmax = 0, gmax = -1;
    int kmax = -1;
    bool found = false;
    for (int i = 0; i < n; ++i) {
        o.Tmax = std::max(o.Tmax, at(i).T);
        wmax = std::max(wmax, at(i).omega);
        if (i + 1 == n) break;
        const double Ta = at(i).T, Tb = at(i + 1).T, ds = at(i + 1).s - at(i).s;
        if (!found && Ta < Tmid && Tb >= Tmid) {
            o.xf = at(i).s + (Tmid - Ta) / (Tb - Ta) * ds;
            found = true;
        }
        const double slope = (Tb - Ta) / ds;
        if (slope > gmax) {
            gmax = slope;
            kmax = i;
        }
    }
    // The parabolic peak of the slab slopes around the discrete maximum.
    double peak = gmax;
    if (kmax > 0 && kmax + 2 < n) {
        const auto sl = [&](int i) { return (at(i + 1).T - at(i).T) / (at(i + 1).s - at(i).s); };
        const double a = sl(kmax - 1), mid = sl(kmax), c = sl(kmax + 1);
        const double den = a - 2.0 * mid + c;
        if (den < 0.0) peak = mid - (c - a) * (c - a) / (8.0 * den);
    }
    o.dth = peak > 0.0 ? dT / peak : 0.0;
    o.omegaEdge = wmax > 0.0 ? std::max(at(0).omega, at(n - 1).omega) / wmax : 0.0;
    const FvFlameSlab& ref = at(std::min(n - 1, int(std::lround(2.0 * sp.dth / h_))));
    o.SL = ref.ry > 0.0 ? o.burn / (area_ * ref.ry) : NAN;
    o.uRef = ref.u;
    o.pRef = ref.p;
    o.TRef = ref.T;
    o.slabs = std::move(b);
}

std::vector<double> fvJetLateral(const FaceMesh& fm, const FvJetSpec& js)
{
    const std::size_t nC = std::size_t(fm.nC);
    std::vector<double> y(nC);
    double lo = HUGE_VAL, hi = -HUGE_VAL, vol = 0;
    for (std::size_t c = 0; c < nC; ++c) {
        y[c] = offset(fm.C, c, js.origin, js.lift);
        const double x = offset(fm.C, c, js.origin, js.dir);
        lo = std::min(lo, x);
        hi = std::max(hi, x);
        vol += fm.V[c];
    }
    // A cyclic pair is an internal face whose d is not C_nei - C_own.
    bool cyclic = false;
    for (std::size_t f = 0; f < std::size_t(fm.nI) && !cyclic; ++f) {
        const std::size_t o = std::size_t(fm.own[f]), n = std::size_t(fm.nei[f]);
        const double gap = std::hypot(fm.C[0][n] - fm.C[0][o] - fm.d[0][f], fm.C[1][n] - fm.C[1][o] - fm.d[1][f],
                                      fm.C[2][n] - fm.C[2][o] - fm.d[2][f]);
        cyclic = gap > 1e-9 * std::hypot(fm.d[0][f], fm.d[1][f], fm.d[2][f]);
    }
    if (fm.emptyAxis < 0 || !cyclic) return y;
    const double period = vol / ((hi - lo + meanSpacing(fm, js.dir)) * fvMeshDepth(fm));
    std::ranges::for_each(y, [&](double& v) { v = std::remainder(v, period); });
    return y;
}

std::expected<FvJetProbe, std::string> FvJetProbe::make(const FaceMesh& fm, const FvJetSpec& spec)
{
    FvJetProbe p;
    p.fm_ = &fm;
    p.spec_ = spec;
    p.h_ = meanSpacing(fm, spec.dir);
    p.hLat_ = meanSpacing(fm, spec.lift);
    if (!(p.h_ > 0.0) || !(p.hLat_ > 0.0)) return std::unexpected("jet probe: no face of the mesh along direction or lift");
    const std::size_t nC = std::size_t(fm.nC);
    p.x_.resize(nC);
    p.y_ = fvJetLateral(fm, spec);
    double hi = -HUGE_VAL;
    p.lo_ = HUGE_VAL;
    for (std::size_t c = 0; c < nC; ++c) {
        p.x_[c] = offset(fm.C, c, spec.origin, spec.dir);
        p.lo_ = std::min(p.lo_, p.x_[c]);
        hi = std::max(hi, p.x_[c]);
    }
    p.lo_ -= 0.5 * p.h_;
    p.nBin_ = int(std::lround((hi - p.lo_) / p.h_ + 0.5));
    if (p.nBin_ < 4) return std::unexpected(std::format("jet probe: {} slabs along the jet", p.nBin_));
    p.bin_.resize(nC);
    p.slabs_.assign(std::size_t(p.nBin_), {});
    for (std::size_t c = 0; c < nC; ++c) {
        p.bin_[c] = std::clamp(int(std::floor((p.x_[c] - p.lo_) / p.h_)), 0, p.nBin_ - 1);
        p.slabs_[std::size_t(p.bin_[c])].push_back(int(c));
    }
    for (std::vector<int>& k : p.slabs_)
        std::ranges::stable_sort(k, {}, [&](int c) { return p.y_[std::size_t(c)]; });
    return p;
}

FvJetMeasure FvJetProbe::measure(const FvState& st, const std::vector<double>& omega) const
{
    const FaceMesh& m = *fm_;
    const FvJetSpec& sp = spec_;
    const DiffusionModel& md = sp.model;
    const std::size_t nC = std::size_t(m.nC), nb = std::size_t(nBin_);
    FvJetMeasure o;
    o.zMin = o.yfMin = o.yoMin = HUGE_VAL;
    o.zMax = -HUGE_VAL;
    std::vector<double> Z(nC), axZ(nb, 0.0), axV(nb, 0.0), axX(nb, 0.0), wSlab(nb, 0.0);
    double wmax = 0, vol = 0;
    const double xIn = lo_ + double(sp.skip) * h_, xOut = lo_ + (double(nBin_) - double(sp.skip)) * h_;
    for (std::size_t c = 0; c < nC; ++c) {
        const FvPrim<double> p =
            fvPrimOf(st.var(0)[c], st.var(1)[c], st.var(2)[c], st.var(3)[c], st.var(4)[c], sp.gamma, sp.Rgas);
        const double YF = st.var(kFvMeanVars)[c] / p.rho, YO = st.var(kFvMeanVars + 1)[c] / p.rho, V = m.V[c];
        const double z = mixtureFraction(md, YF, YO);
        Z[c] = z;
        o.Tmax = std::max(o.Tmax, p.temp);
        if (x_[c] >= xIn && x_[c] <= xOut) o.TmaxIn = std::max(o.TmaxIn, p.temp);
        o.burn += V * omega[c];
        o.mass += V * p.rho;
        o.energy += V * (st.var(4)[c] + sp.q * st.var(kFvMeanVars)[c]);
        o.pMean += V * p.p;
        vol += V;
        o.zMin = std::min(o.zMin, z);
        o.zMax = std::max(o.zMax, z);
        o.yfMin = std::min(o.yfMin, YF);
        o.yoMin = std::min(o.yoMin, YO);
        const bool out = z < -1e-6 || z > 1 + 1e-6 || YF < -1e-6 || YF > md.yF1 + 1e-6 || YO < -1e-6 || YO > md.yO2 + 1e-6;
        o.hits += out ? 1.0 : 0.0;
        wmax = std::max(wmax, omega[c]);
        const std::size_t b = std::size_t(bin_[c]);
        wSlab[b] = std::max(wSlab[b], omega[c]);
        if (std::abs(y_[c]) <= hLat_) {
            axZ[b] += V * z;
            axX[b] += V * x_[c];
            axV[b] += V;
        }
    }
    o.pMean /= vol * sp.T0;
    const double zs = md.zst();
    int prev = -1;
    for (int i = 0; i < nBin_ && !std::isfinite(o.xTip); ++i) {
        const std::size_t b = std::size_t(i);
        if (!(axV[b] > 0.0)) continue;
        const double zb = axZ[b] / axV[b], xb = axX[b] / axV[b];
        if (prev >= 0) {
            const std::size_t a = std::size_t(prev);
            const double za = axZ[a] / axV[a], xa = axX[a] / axV[a];
            if (za >= zs && zb < zs) o.xTip = xa + (zs - za) / (zb - za) * (xb - xa) - lo_;
        }
        prev = i;
    }
    for (int i = 0; i < nBin_ && wmax > 0.0; ++i)
        if (wSlab[std::size_t(i)] >= 0.1 * wmax) {
            o.xBase = double(i) * h_;
            break;
        }
    if (!std::isfinite(o.xTip)) return o;
    // The Z_st half-width at x_tip/2: each side walked outward from the axis.
    const std::vector<int>& k = slabs_[std::size_t(std::clamp(int(std::floor(0.5 * o.xTip / h_)), 0, nBin_ - 1))];
    const auto yAt = [&](std::size_t i) { return y_[std::size_t(k[i])]; };
    const auto zAt = [&](std::size_t i) { return Z[std::size_t(k[i])]; };
    double yR = NAN, yL = NAN;
    std::size_t iR = 0;
    while (iR < k.size() && yAt(iR) < 0.0) ++iR;
    for (std::size_t i = iR; i + 1 < k.size() && !std::isfinite(yR); ++i)
        if (zAt(i) >= zs && zAt(i + 1) < zs) yR = yAt(i) + (zs - zAt(i)) / (zAt(i + 1) - zAt(i)) * (yAt(i + 1) - yAt(i));
    for (std::size_t i = iR; i >= 2 && !std::isfinite(yL); --i)
        if (zAt(i - 1) >= zs && zAt(i - 2) < zs)
            yL = yAt(i - 1) + (zs - zAt(i - 1)) / (zAt(i - 2) - zAt(i - 1)) * (yAt(i - 2) - yAt(i - 1));
    if (std::isfinite(yR) && std::isfinite(yL)) o.half = 0.5 * (yR - yL);
    return o;
}

FvJetRows fvJetRows(const FaceMesh& fm, const FvBcSpec& spec, std::vector<FvBc> bc, const FvJetSpec& js)
{
    FvJetRows o{.bc = std::move(bc), .faceRow = std::vector<int>(std::size_t(fm.nB), 0)};
    const std::size_t nI = std::size_t(fm.nI);
    for (std::size_t p = 0; p < fm.patches.size(); ++p) {
        const MeshPatch& m = fm.patches[p];
        std::fill_n(o.faceRow.begin() + m.first, m.count, int(p));
        if (o.bc[p].kind == FvBcKind::Outflow) o.bc[p].ownY = o.bc[p].characteristic = true;
        const auto it = std::ranges::find(spec.patches, m.name, &FvPatchSpec::name);
        if (it == spec.patches.end() || !it->jet) continue;
        for (int i = 0; i < m.count; ++i) {
            const std::size_t f = nI + std::size_t(m.first + i);
            const DiffusionInflowRow r = diffusionInflow(js.model, js.geom, offset(fm.Cf, f, js.origin, js.lift) + js.geom.yc);
            FvBc b = o.bc[p];
            b.tIn = js.T0 * r.t;
            b.diffusive = true;
            for (std::size_t k = 0; k < 3; ++k) b.uIn[k] = r.u * js.dir[k];
            b.y[0] = r.YF;
            b.y[1] = r.YO;
            o.faceRow[std::size_t(m.first + i)] = int(o.bc.size());
            o.bc.push_back(b);
        }
    }
    return o;
}

FvState fvJetState(const FaceMesh& fm, const FvJetSpec& js, double ignition)
{
    const double p0 = js.T0;
    FvState q = FvState::zeros(fm.nC, kFvMeanVars + 2);
    const std::vector<double> lat = fvJetLateral(fm, js);
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const double x = offset(fm.C, c, js.origin, js.dir), y = lat[c] + js.geom.yc;
        DiffusionInflowRow in = diffusionInflow(js.model, js.geom, y);
        if (ignition > 0) {
            const double w = 0.5 * (1.0 - std::tanh((x - ignition) / js.geom.edge));
            const SheetState st = burkeSchumann(js.model, in.Z * w);
            in = {js.geom.uCo + (in.u - js.geom.uCo) * w, in.Z * w, st.YF, st.YO, st.t, in.jet * w};
        }
        const double rho = 1.0 / in.t; // p0 / (R T0 t), R = 1, p0 = T0
        q.var(0)[c] = rho;
        for (int k = 0; k < 3; ++k) q.var(1 + k)[c] = rho * in.u * js.dir[std::size_t(k)];
        q.var(4)[c] = p0 / (js.gamma - 1.0) + 0.5 * rho * in.u * in.u;
        q.var(kFvMeanVars)[c] = rho * in.YF;
        q.var(kFvMeanVars + 1)[c] = rho * in.YO;
    }
    return q;
}

FvState fvFlameState(const FaceMesh& fm, const FvFlameSpec& spec, std::array<double, 3> origin, double m, int nY,
                     const std::function<double(double)>& t)
{
    FvState q = FvState::zeros(fm.nC, kFvMeanVars + nY);
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const double s = (fm.C[0][c] - origin[0]) * spec.dir[0] + (fm.C[1][c] - origin[1]) * spec.dir[1] +
                         (fm.C[2][c] - origin[2]) * spec.dir[2];
        const PremixedPoint p = premixedPoint(spec.gamma, spec.Tu, spec.sigma, m, t(s));
        q.var(0)[c] = p.rho;
        for (int k = 0; k < 3; ++k) q.var(1 + k)[c] = p.rho * p.u * spec.dir[std::size_t(k)];
        q.var(4)[c] = p.E;
        q.var(kFvMeanVars + spec.fuel)[c] = p.rho * p.Y;
    }
    return q;
}

} // namespace dns
