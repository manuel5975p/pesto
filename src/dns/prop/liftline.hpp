// Lifting-line propeller analysis, the port of tools/liftline.py: B blades,
// one horseshoe per panel on the radial quarter-chord line, a prescribed
// constant-pitch helical wake per trailing vortex realigned from the solution,
// Kerwin image hub, Vatistas (n = 2) cores, Biot-Savart over the whole vortex
// system -- so tip loss and blade interference come from the vortex system,
// not from a loss factor. Gamma by Newton on a frozen wake (MINPACK hybrd and
// pseudo-transient continuation behind it), Chattot's stall viscosity and a
// homotopy from the stall-saturated linear polar to pick the attached branch.
#pragma once

#include "dns/prop/blade.hpp"
#include "dns/prop/curve.hpp"
#include "dns/prop/polar.hpp"
#include "dns/prop/spec.hpp"

#include <array>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace dns::prop {

using Vec3 = std::array<double, 3>;

// Velocity at p of a unit-strength straight segment a -> b with a Vatistas
// n = 2 core: 1/h becomes h/sqrt(core^4 + h^4). Exactly zero on the
// segment's own line (the control points sit on the bound vortex).
Vec3 segmentVelocity(const Vec3& p, const Vec3& a, const Vec3& b, double core);

// Panels on [rh, R] in metres: nodes (panels + 1), control points (panels),
// chord, beta. Half-cosine (tip-only) clustering under an image hub, full
// cosine for a free root.
struct LiftGeometry
{
    std::vector<double> nodes, cps, chord, beta;
};
LiftGeometry liftGeometry(const Planform& pf, double hubRR, int panels, HubModel hub,
                          double diameter);

// One converged (or flagged) operating point.
struct LiftPoint
{
    std::vector<double> g, ua, ut, va, vt, W, phi, alpha, re, cl, cd;
    std::vector<double> dT, dQ;   // per unit span, all blades
    double T = 0, Q = 0, re75 = 0;
    bool converged = false;
    double u0 = 0, dpitch = 0, core = 0;
    int iters = 0;
};
LiftPoint liftlineSolve(double J, const Case& c, const LiftGeometry& geo,
                        const Polar& polar);

struct LiftCurve
{
    std::vector<CurvePoint> rows;
    std::optional<LiftPoint> span;   // the --dump-span point
};
LiftCurve liftlineCurve(const Case& c, const LiftGeometry& geo, const Polar& polar);

// The --dump-span table (liftline.py's format).
std::string liftlineSpanTable(double J, const LiftGeometry& geo, const LiftPoint& p);

// Biot-Savart against closed forms, then the vortex system against the
// actuator-disk cylinder and the Betz / Prandtl-Goldstein limits.
struct SelfCheck
{
    std::string name;
    bool pass = false;
    std::string message;
};
std::vector<SelfCheck> liftlineSelfTest();

} // namespace dns::prop
