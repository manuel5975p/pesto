// FP64 host ORACLE of the RANS channel: the steady 1D restriction of THE SAME
// discrete operator the GPU solves on CurvMesh::channelY -- same metrics
// (CurvMetrics, built in double), same face formulas and face coefficients
// (dns/rans.hpp), same isothermal ghost algebra, same body force -- solved by
// Newton with pseudo-transient continuation. A test oracle, not a backend:
// it lets the GPU be checked against a discrete fixed point in seconds instead
// of a 1e5-step march (docs/rans.md).
//
// Wall units as dns channel: delta = 1, u_tau = 1, f = 1, mu = 1/reTau,
// rho = R = 1 at the start, Twall = pRef = channelPRef(reTau, mach), constant
// viscosity, Pr = 0.72, Pr_t = 0.9.
#pragma once

#include "dns/comp.hpp"
#include "dns/grid.hpp"
#include "dns/rans.hpp"

#include <expected>
#include <string>
#include <vector>

namespace dns {

struct Channel1dCase
{
    int ny = 64;               // FLUID rows (dns channel --ny); +2 ghost rows
    double beta = 2.0;         // ChannelStretchY clustering (--stretch)
    double reTau = 180, mach = 0.2;
    RansModel model = RansModel::SaNeg;  // None = laminar (no scalar at all)
    double totalMass = 2.0;    // fluid mass per unit wall area (rho0 2 delta)
    double tol = 1e-11;        // stop at resid < tol
};

struct Channel1dSolution
{
    // Fluid rows 1..ny of the grid, index 0 = the first row off the bottom.
    std::vector<double> y, d, rho, u, T;
    std::vector<double> nut;        // SA: the SPECIFIC scalar nu~
    std::vector<double> k, omega, mut; // SST: specific k, omega; mu_t
    double p = 0, ub = 0;      // uniform pressure; mass-weighted bulk velocity
    double tauWall = 0;        // discrete bottom-wall face stress
    double resU = 0, resE = 0, resT = 0, resid = 0; // docs/rans.md norms
    int iters = 0;
    double ubEst = 0, Twall = 0; // the case's derived constants
    RansModel model = RansModel::None;
    double omegaCoef = 0;      // SST wall: omega_w = omegaCoef / rho_w
};

// The case's grid (nx x (ny+2) x nz, the validator's box): the oracle's own
// discretization uses nx = 2, nz = 1 (every y-face quantity is x/z-free).
// Pre: c valid (ny >= 8, reTau > 0, 0 < mach < 1), nx >= 2, nz >= 1.
Grid channel1dGrid(const Channel1dCase& c, int nx, int nz, double lx, double lz);

// Steady FP64 solution; an error (never an assert) when Newton fails.
std::expected<Channel1dSolution, std::string> solveChannel1d(const Channel1dCase& c);

// The solution on grid g (rows replicated over x and z, ghost rows by the
// isothermal wall algebra) for upload. Pre: g.ny == ny + 2 of the case.
CompState toCompState(const Channel1dSolution& s, const Grid& g);
// Conserved scalar i on grid g with the wall's ghost algebra: SA rho nu~
// (ghost -rho_g nu~_1); SST i = 0 rho k (ghost -rho_g k_1), i = 1 rho omega
// (ghost rho_g (2 omega_w - omega_1)). Empty for laminar.
// Pre: i < ransScalars(s.model).
std::vector<double> toTurb(const Channel1dSolution& s, const Grid& g, int i = 0);

} // namespace dns
