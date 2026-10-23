// The resolved `dns prop` case: what validate() hands the thrust-curve codes.
// Pure data, no I/O -- the planform and polar FILES are read at acquire time.
#pragma once

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace dns::prop {

enum class Method { Bemt, Liftline };
// Lifting-line root: Kerwin image hub (a wall) or a free root end.
enum class HubModel { Image, None };
// Lifting-line wake pitch: circumferential-mean disk velocity or blade-local.
enum class WakePitch { Mean, Local };

std::span<const char* const> methodNames();
std::span<const char* const> hubModelNames();
std::span<const char* const> wakePitchNames();

// The operating point, SI.
struct Operating
{
    double diameter = 0.254;   // m
    double rpm = 6000.0;
    double rho = 1.225;        // kg/m^3
    double nu = 1.5e-5;        // m^2/s
    double sound = 340.0;      // m/s (station / tip Mach, reported only)
    double revPerSec() const { return rpm / 60.0; }
};

struct LiftlineOptions
{
    int panels = 30;            // per blade
    double wakeLength = 10.0;   // helix length, diameters
    int alignIters = 3;         // wake realignments after the seed wake
    double alignTol = 0.01;     // relative wake-pitch change at convergence
    double segDeg = 10.0;       // helix segment angle
    double core = 0.0;          // Vatistas core, tip radii; 0 = auto
    HubModel hub = HubModel::Image;
    WakePitch wakePitch = WakePitch::Mean;
    double stallVisc = 2.0;     // Chattot viscosity, x sawtooth bound; 0 off
};

struct Case
{
    Method method = Method::Bemt;
    std::string planformPath;   // empty only for --selftest
    std::string polarPath;      // empty: the built-in thin-airfoil polar
    double thinCd0 = 0.02;
    double thinAlpha0Deg = 0.0;
    int blades = 3;
    double hubRR = 0.0;         // hub radius / tip radius; 0 = planform's first r/R
    Operating op;
    std::vector<double> js;     // sorted, unique, >= 0
    std::optional<double> dumpSpan;
    int stations = 40;          // BEMT elements
    LiftlineOptions ll;
    bool selftest = false;      // lifting-line kernel self-test, then exit
    std::string outCsv;         // empty: no CSV
    bool quiet = false;
};

} // namespace dns::prop
