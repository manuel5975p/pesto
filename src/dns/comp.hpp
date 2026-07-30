// 3D compressible Navier-Stokes (ideal gas) on a periodic collocated grid,
// uniform Cartesian or distorted regular (curvilinear, see curvmesh.hpp);
// CPU double-precision reference, fully explicit (no elliptic solve).
// nz == 1 is the exact 2D restriction (z stencils cancel identically).
// Smooth flows only: central KE-consistent fluxes carry no shock capturing.
#pragma once

#include "dns/grid.hpp"
#include "dns/transport.hpp"

#include <expected>
#include <memory>
#include <string>
#include <vector>

namespace dns {

struct CurvMesh;
struct CurvMetrics;

// Conservative state; each array has one value per g.idx().
struct CompState
{
    std::vector<double> rho, mx, my, mz, E;

    // Zero-initialized state sized for g.
    static CompState zeros(const Grid& g);
};

// Explicit compressible solver: Kennedy-Gruber split-form convective fluxes
// (kinetic-energy consistent, non-dissipative), face-based Newtonian viscous
// fluxes (Stokes hypothesis) + Fourier heat flux, SSP-RK3 in time.
struct Comp
{
    Grid g;              // fully periodic
    // The gas: gamma, R, Pr and the viscosity law. Constructed constant-mu by
    // make(); assign a Sutherland Transport (dns/transport.hpp) for a
    // temperature-dependent run.
    Transport tr;

    // Null = uniform Cartesian (the validated path, untouched). Non-null
    // switches rhs/stableDt to the finite-volume curvilinear form on a
    // distorted regular mesh (curvmesh.hpp).
    std::shared_ptr<const CurvMetrics> curv;

    // gamma > 1, mu >= 0. The pre-thermal gas, Transport::constant.
    static Comp make(const Grid& g, double gamma, double mu);

    // Curvilinear variant; mesh defects (inverted cells, bad sizes) are
    // user-triggered and reported, never asserted.
    static std::expected<Comp, std::string> make(const CurvMesh& mesh,
                                                 double gamma, double mu);

    double gamma() const { return tr.gamma; }
    double mu() const { return tr.muRef; }

    // p = (gamma-1)(E - |m|^2 / (2 rho)). PRECONDITION rho > 0, established by
    // stableDt() -- which reports a diverged field rather than asserting -- so
    // this stays branch-free in the inner loops.
    double pressure(double rho, double mx, double my, double mz, double E) const;

    // Conservative rhs dq/dt = -div(F); out arrays sized like s.
    void rhs(const CompState& s, CompState& out) const;

    // One SSP-RK3 step in place; dt within stableDt.
    void step(CompState& s, double dt) const;

    // Largest stable dt: acoustic CFL and viscous limit combined. Returns an
    // error instead of a NaN when the state has diverged (rho <= 0, p <= 0):
    // a user can cause that, so it is std::expected, not an assert.
    std::expected<double, std::string> stableDt(const CompState& s,
                                                double cfl = 0.4) const;
};

} // namespace dns
