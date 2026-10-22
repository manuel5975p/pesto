#include "dns/gpu/comp_sink.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/index.hpp"

namespace dns::gpu {

namespace {

// Cosine band ramp along one axis: 1 at the extreme row, 0 outside nb cells.
// Same shape family as the kFarSponge strength, so the two bands compare.
__device__ Real bandRamp(int q, int n, int nb)
{
    const Real d = bandDepth(q, n, nb);   // > 0 exactly inside the band
    if (!(d > Real(0))) return Real(0);
    return Real(0.5) * (Real(1) - cosf(Real(M_PI) * d));
}

// Scale-selective damping band: out += cr * rhs with
//   rhs_i = -( sig_{i+1/2} t_{i+1/2} - sig_{i-1/2} t_{i-1/2} )   per axis,
//   t_{m+1/2} = [ (q_{m+2}-q_{m+1}) - 2(q_{m+1}-q_m) + (q_m-q_{m-1}) ] / 16,
// the compounded undivided Laplacian D4 (Fourier symbol sin^4(kh/2)) in
// conservative flux form. Written as nested DIFFERENCES on purpose: a
// constant field gives exactly 0.0f at every face, so a resting or uniform
// state is untouched to the bit. Ghost cells one past each extreme are
// linear-extrapolated (2 q_edge - q_in), which keeps D4 zero on linear ramps
// through the closure as well; the DOMAIN faces themselves carry zero flux,
// so each axis telescopes to an exact zero sum -- the band drains fluctuation
// content and conserves everything else.
//
// Both cells sharing a face evaluate the identical expression (same indices,
// same order), so the telescoping holds in floats, not just on paper.
// Deterministic; no atomics; each thread owns its cell.
__global__ void kCompSink(GpuGrid g, int nb, Real sgx, Real sgy, Real sgz,
                          Real cr, const Real* __restrict__ sr,
                          const Real* __restrict__ smx,
                          const Real* __restrict__ smy,
                          const Real* __restrict__ smz,
                          const Real* __restrict__ sE, Real* orho, Real* omx,
                          Real* omy, Real* omz, Real* oE)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const int ij[3] = {i, j, k}, nn[3] = {g.nx, g.ny, g.nz};
    const Real sg[3] = {sgx, sgy, sgz};
    const Real* f[5] = {sr, smx, smy, smz, sE};
    Real add[5] = {Real(0), Real(0), Real(0), Real(0), Real(0)};
    bool banded = false;
    for (int a = 0; a < 3; ++a) {
        if (sg[a] <= Real(0)) continue;
        const int q = ij[a], n = nn[a];
        // Face strength: cell-average of the ramp, zero on the domain faces.
        auto faceSig = [&](int m) { // face m+1/2
            if (m < 0 || m >= n - 1) return Real(0);
            return Real(0.5) * (bandRamp(m, n, nb) + bandRamp(m + 1, n, nb)) *
                   sg[a];
        };
        const Real shi = faceSig(q), slo = faceSig(q - 1);
        if (shi <= Real(0) && slo <= Real(0)) continue;
        banded = true;
        // Value at offset d along the axis; one ghost past each extreme is
        // linear-extrapolated so D4 stays zero on a linear field there.
        auto at = [&](const Real* v, int d) {
            const int t = q + d;
            int ii = i, jj = j, kk = k;
            int& m = a == 0 ? ii : a == 1 ? jj : kk;
            if (t < 0) {
                m = 0;
                const Real v0 = v[gidx(g, ii, jj, kk)];
                m = 1;
                return Real(2) * v0 - v[gidx(g, ii, jj, kk)];
            }
            if (t > n - 1) {
                m = n - 1;
                const Real v0 = v[gidx(g, ii, jj, kk)];
                m = n - 2;
                return Real(2) * v0 - v[gidx(g, ii, jj, kk)];
            }
            m = t;
            return v[gidx(g, ii, jj, kk)];
        };
        auto flux = [&](const Real* v, int m) { // sig * t at face m+1/2
            const Real qm1 = at(v, m - q - 1), q0 = at(v, m - q);
            const Real qp1 = at(v, m - q + 1), qp2 = at(v, m - q + 2);
            return ((qp2 - qp1) - Real(2) * (qp1 - q0) + (q0 - qm1)) *
                   Real(1.0 / 16.0);
        };
        for (int v = 0; v < 5; ++v)
            add[v] -= shi * flux(f[v], q) - slo * flux(f[v], q - 1);
    }
    if (!banded) return;
    orho[c] += cr * add[0];
    omx[c] += cr * add[1];
    omy[c] += cr * add[2];
    omz[c] += cr * add[3];
    oE[c] += cr * add[4];
}

} // namespace

std::expected<DeviceCompSink, hipError_t> DeviceCompSink::make(const Grid& g,
                                                               Config c)
{
    if (g.nz == 1) c.openZ = false;
    const bool nbOk = c.nb >= 1 && 2 * c.nb < g.nx &&
                      (!c.openY || 2 * c.nb < g.ny) &&
                      (!c.openZ || 2 * c.nb < g.nz);
    // alpha <= 2: the SSP-RK3 real-axis bound. sigma dt <= alpha cfl (1 + d)
    // at the worst corner, and 2 * 0.4 * 3 < 2.51 with margin.
    if (!nbOk || !(c.alpha > 0) || c.alpha > 2.0 || !(c.cInf > 0))
        return std::unexpected(hipErrorInvalidValue);
    return DeviceCompSink(g, c);
}

DeviceCompSink::DeviceCompSink(const Grid& g, const Config& c)
    : g_(GpuGrid::from(g)), c_(c)
{
}

void DeviceCompSink::apply(const DeviceCompState& s, DeviceCompState& out,
                           const StageCombine& cmb) const
{
    const Real a = Real(c_.alpha * c_.cInf);
    kCompSink<<<launchBlocks(g_.size()), blockSize>>>(
        g_, c_.nb, a / g_.hx, c_.openY ? a / g_.hy : Real(0),
        c_.openZ ? a / g_.hz : Real(0), cmb.cr, s.rho.data(), s.mx.data(),
        s.my.data(), s.mz.data(), s.E.data(), out.rho.data(), out.mx.data(),
        out.my.data(), out.mz.data(), out.E.data());
    checkLastKernel("kCompSink");
}

} // namespace dns::gpu
