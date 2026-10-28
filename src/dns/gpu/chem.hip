#include "dns/gpu/chem.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/plane_sums.hpp"
#include "dns/gpu/internal/reduce.hpp"

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

} // namespace

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
