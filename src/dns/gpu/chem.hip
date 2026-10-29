#include "dns/gpu/chem.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/plane_sums.hpp"
#include "dns/gpu/internal/reduce.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace dns::gpu {
namespace {

// J_c of chemStableDt, FP64 from the conserved state.
struct ChemRate
{
    const Real *rho, *mx, *my, *mz, *E, *ry;
    double gamma, Rgas, lnA, Ta, q, cv;
    __host__ __device__ double operator()(int c) const
    {
        const double r = rho[c];
        const double ke = 0.5 * (double(mx[c]) * mx[c] + double(my[c]) * my[c] +
                                 double(mz[c]) * mz[c]) / r;
        const double p = (gamma - 1.0) * (double(E[c]) - ke);
        const double T = p / (r * Rgas);
        const double Y = ry[c] / r;
        const double k = exp(lnA - Ta / T);
        const double J = k * (1.0 + q * (Y > 0 ? Y : 0.0) * Ta / (cv * T * T));
        return isfinite(J) ? J : HUGE_VAL; // never dropped by the max
    }
};

// J_c of the two-reactant step: k rho (Y_O + s Y_F) (the depletion
// eigenvalue of the (rhoY_F, rhoY_O) pair) + q k rho Y_F Y_O Ta/(cv T^2)
// (thermal runaway), k = A e^(-Ta/T), Y floored at 0 as the rate floors them.
struct ChemRate2
{
    const Real *rho, *mx, *my, *mz, *E, *rf, *ro;
    double gamma, Rgas, lnA, Ta, q, cv, s;
    __host__ __device__ double operator()(int c) const
    {
        const double r = rho[c];
        const double ke = 0.5 * (double(mx[c]) * mx[c] + double(my[c]) * my[c] +
                                 double(mz[c]) * mz[c]) / r;
        const double p = (gamma - 1.0) * (double(E[c]) - ke);
        const double T = p / (r * Rgas);
        const double yf = fmax(rf[c] / r, 0.0), yo = fmax(ro[c] / r, 0.0);
        const double k = exp(lnA - Ta / T);
        const double J =
            k * r * (yo + s * yf + q * yf * yo * Ta / (cv * T * T));
        return isfinite(J) ? J : HUGE_VAL;
    }
};

__global__ void kFlameColumns(GpuGrid g, const Real* rho, const Real* mx,
                              const Real* E, const Real* ry, const Real* T,
                              const Real* omega, const Real* u, const Real* p,
                              double q, double* out)
{
    const int i = int(std::size_t(blockIdx.x) * blockDim.x + threadIdx.x);
    if (i >= g.nx) return;
    double s[kFlameSums] = {};
    s[9] = HUGE_VAL;
    s[10] = -HUGE_VAL;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j) {
            const std::size_t c = gidx(g, i, j, k);
            const double r = rho[c], y = ry[c];
            s[0] += r;
            s[1] += mx[c];
            s[2] += T[c];
            s[3] += y;
            s[4] += omega[c];
            s[5] += double(E[c]) + q * y;
            s[6] += u[c];
            s[7] += p[c];
            const double Y = y / r;
            s[8] += (Y < -1e-6 || Y > 1 + 1e-6) ? 1.0 : 0.0;
            s[9] = fmin(s[9], Y);
            s[10] = fmax(s[10], Y);
        }
    for (int v = 0; v < kFlameSums; ++v)
        out[std::size_t(i) * kFlameSums + v] = s[v];
}

// One thread per x column, (k, j) ascending, FP64: the velocity moments and
// |grad Y| (central, wrap in y and z; 0 on the open x rows).
__global__ void kTurbColumns(GpuGrid g, const Real* u, const Real* v,
                             const Real* w, const Real* Y, double* out)
{
    const int i = int(std::size_t(blockIdx.x) * blockDim.x + threadIdx.x);
    if (i >= g.nx) return;
    double s[kTurbSums] = {};
    const bool edge = i == 0 || i == g.nx - 1;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j) {
            const std::size_t c = gidx(g, i, j, k);
            const double a = u[c], b = v[c], d = w[c];
            s[0] += a;
            s[1] += b;
            s[2] += d;
            s[3] += a * a;
            s[4] += b * b;
            s[5] += d * d;
            if (edge) continue;
            const Offsets o = offsetsAt(g, i, j, k);
            const double gx = (double(Y[c + o.p[0]]) - Y[c + o.m[0]]) / (2.0 * g.hx);
            const double gy = (double(Y[c + o.p[1]]) - Y[c + o.m[1]]) / (2.0 * g.hy);
            const double gz = (double(Y[c + o.p[2]]) - Y[c + o.m[2]]) / (2.0 * g.hz);
            s[6] += sqrt(gx * gx + gy * gy + gz * gz);
        }
    for (int q = 0; q < kTurbSums; ++q)
        out[std::size_t(i) * kTurbSums + q] = s[q];
}

struct MaxField
{
    const Real* f;
    __host__ __device__ double operator()(int c) const
    {
        const double v = f[c];
        return isfinite(v) ? v : HUGE_VAL; // never dropped by the max
    }
};

struct MinField
{
    const Real* f;
    __host__ __device__ double operator()(int c) const
    {
        const double v = f[c];
        return isfinite(v) ? v : -HUGE_VAL;
    }
};

// T = p / (rho R) from the conserved state, FP64 (stableDt's MaxT).
struct TOf
{
    const Real *rho, *mx, *my, *mz, *E;
    double gamma, Rgas;
    __host__ __device__ double operator()(int c) const
    {
        const double r = rho[c];
        const double ke = 0.5 * (double(mx[c]) * mx[c] + double(my[c]) * my[c] +
                                 double(mz[c]) * mz[c]) / r;
        const double T = (gamma - 1.0) * (double(E[c]) - ke) / (r * Rgas);
        return isfinite(T) ? T : HUGE_VAL;
    }
};

} // namespace

double tflesStableDt(const DeviceComp& d, const DeviceCompState& s,
                     const DeviceScalar& fe, ReduceScratch& r)
{
    const GpuGrid& g = d.grid();
    const Transport& tr = d.transport();
    const std::size_t n = g.size();
    assert(fe.size() == n);
    double mu = tr.muRef;
    if (tr.sutherland && tr.muRef > 0.0)
        mu = tr.mu(internal::reduce(
            r, n,
            TOf{s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(),
                s.E.data(), tr.gamma, tr.Rgas},
            rocprim::maximum<double>(), 0.0));
    double muEff = mu * internal::reduce(r, n, MaxField{fe.data()},
                                         rocprim::maximum<double>(), 0.0);
    if (d.les())
        muEff += internal::reduce(r, n, MaxField{d.mut().data()},
                                  rocprim::maximum<double>(), 0.0);
    if (!(muEff > 0.0)) return HUGE_VAL;
    const double rhoMin = internal::reduce(r, n, MinField{s.rho.data()},
                                           rocprim::minimum<double>(), 1e300);
    double h = std::min<double>(g.hx, g.hy);
    if (g.nz > 1) h = std::min<double>(h, g.hz);
    return 0.25 * h * h * rhoMin / (muEff * tr.diffusivityFactor());
}

hipError_t turbColumnStats(const DeviceComp& d, DeviceBuffer<double>& scratch,
                           std::vector<double>& host)
{
    const GpuGrid& g = d.grid();
    const std::size_t n = std::size_t(g.nx) * kTurbSums;
    if (auto e = internal::ensurePlaneScratch(scratch, n); e != hipSuccess)
        return e;
    kTurbColumns<<<launchBlocks(std::size_t(g.nx)), blockSize>>>(
        g, d.u().data(), d.v().data(), d.w().data(),
        d.speciesY(d.speciesSetup().fuel).data(), scratch.data());
    checkLastKernel("kTurbColumns");
    return internal::fetchPlaneScratch(scratch, n, host);
}

double chemStableDt(const DeviceComp& d, const DeviceCompState& s,
                    ReduceScratch& r)
{
    const SpeciesSetup& sp = d.speciesSetup();
    assert(d.nSpecies() > 0 && s.nY == d.nSpecies());
    if (!std::isfinite(sp.lnA)) return HUGE_VAL; // no reaction
    const Transport& tr = d.transport();
    const double cv = tr.Rgas / (tr.gamma - 1.0);
    if (sp.ox >= 0) {
        const ChemRate2 f2{s.rho.data(), s.mx.data(), s.my.data(),
                           s.mz.data(),  s.E.data(),  s.y[sp.fuel].data(),
                           s.y[sp.ox].data(), tr.gamma, tr.Rgas, sp.lnA,
                           sp.Ta, sp.q, cv, sp.stoich};
        const double J = internal::reduce(r, d.grid().size(), f2,
                                          rocprim::maximum<double>(), 0.0);
        return J > 0 ? kChemDt / J : HUGE_VAL;
    }
    const ChemRate f{s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(),
                     s.E.data(),   s.y[sp.fuel].data(), tr.gamma, tr.Rgas,
                     sp.lnA,       sp.Ta, sp.q, cv};
    const double J = internal::reduce(r, d.grid().size(), f,
                                      rocprim::maximum<double>(), 0.0);
    return J > 0 ? kChemDt / J : HUGE_VAL;
}

hipError_t flameColumnStats(const DeviceComp& d, const DeviceCompState& s,
                            DeviceBuffer<double>& scratch,
                            std::vector<double>& host)
{
    const GpuGrid& g = d.grid();
    const int fuel = d.speciesSetup().fuel;
    const std::size_t n = std::size_t(g.nx) * kFlameSums;
    if (auto e = internal::ensurePlaneScratch(scratch, n); e != hipSuccess)
        return e;
    kFlameColumns<<<launchBlocks(std::size_t(g.nx)), blockSize>>>(
        g, s.rho.data(), s.mx.data(), s.E.data(), s.y[fuel].data(),
        d.T().data(), d.omega().data(), d.u().data(), d.p().data(),
        d.speciesSetup().q, scratch.data());
    checkLastKernel("kFlameColumns");
    return internal::fetchPlaneScratch(scratch, n, host);
}

} // namespace dns::gpu
