// Host side of the time-accurate incompressible FV solver (docs/fvincomp.md):
// the --bc FILE.toml patch table, the structured meshes of `dns tgv --solver
// fv` and `dns flow --solver fv`, and the tables gpu/fv_incomp uploads. FP64,
// FaceMesh numbering, pure (no GPU, no filesystem), deterministic.
//
// NOT simpleFoam: `dns flow --polymesh CASE` alone stays gpu/fv_simple. This is
// the collocated SSP-RK3 + projection march of the MAC solver on any FaceMesh.
#pragma once

#include "dns/facemesh.hpp"
#include "dns/fvprecision.hpp"
#include "dns/ogrid.hpp"
#include "dns/sgs.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace dns {

// A patch's boundary condition (--bc FILE.toml `kind`):
//  wall        no-slip, impermeable; u = u + omega x (Cf - origin)
//  slip        impermeable, no shear
//  inflow      u fixed
//  outflow     p fixed, u zero gradient (backflow: normal part only)
//  convective  du/dt + uc du/dn = 0, p zero gradient, net flux closed on it
//  far         per face by the sign of u.S: inflow, or outflow at p
enum class FvIncompBc : std::uint8_t { Wall, Slip, Inflow, Outflow, Convective, Far };

// The per-face device code; Far resolves to Inflow or Outflow.
enum class FvIncompFaceBc : std::uint8_t { Wall = 0, Slip = 1, Inflow = 2, Outflow = 3, Convective = 4 };

// One patch of --bc. u: wall velocity, inflow / far value. p: outflow / far
// pressure. uc: convective speed. force: the patch enters the cd / cl tally.
struct FvIncompPatch
{
    std::string name;
    FvIncompBc kind{};
    std::array<double, 3> u{}, omega{}, origin{};
    double p = 0, uc = 0;
    bool force = false;
    bool hasT = false; // wall `t`: the fixed temperature of dns conv; absent: adiabatic
    double T = 0;
};

// The --bc table plus its [reference] and [initial] tables. uRef, length: the
// Re scale (nu = uRef length / Re). area: the force reference (0: no cd / cl).
struct FvIncompBcs
{
    std::vector<FvIncompPatch> patches; // polyMesh order after bindFvIncompBcs
    double uRef = 1, length = 1, area = 0;
    std::array<double, 3> u0{};         // uniform initial velocity
    bool hasReference = false, hasInitial = false; // the tables were given
};

// Parses --bc text. Error (exit 3) names the entry: unknown key or kind, a
// missing `kind`, a non-finite number, `omega` on anything but a wall.
std::expected<FvIncompBcs, std::string> parseFvIncompBcs(std::string_view toml);

// Orders b.patches by fm.patches. `empty` patches take none. Error (exit 3):
// a non-empty patch without an entry, an entry naming no patch, an entry on
// an `empty` patch, a wall omega off the emptyAxis on a 2D mesh.
std::expected<void, std::string> bindFvIncompBcs(FvIncompBcs& b, const FaceMesh& fm);

// The --dry-run report of b against the manifest: one line per patch. Same
// errors as bindFvIncompBcs, from the patch names alone.
std::expected<std::string, std::string> describeFvIncompBcs(const FvIncompBcs& b, const PolyManifest& man);

// Run controls of gpu/fv_incomp. tol: |div phi|_2 <= tol |div phi*|_2 per
// pressure solve (phi*: the predicted flux); pressure stops at the arithmetic's
// noise floor when that is larger (gpu/fv_incomp.hpp). nCorr 0: 1 on an
// orthogonal mesh, 2 otherwise. les 0: DNS; else Cs (Smagorinsky) or C_sigma.
struct FvIncompControls
{
    double nu = 0, tol = 1e-5, les = 0;
    int nCorr = 0, maxIter = 200;
    SgsModel sgs = SgsModel::Smagorinsky;
    FvPrecision precision = FvPrecision::Fp64;
    // Temperature (DeviceFvIncomp::make given T0): diffusivity, Boussinesq force
    // betaG (T - Tref) per unit mass.
    double kappa = 0, Tref = 0;
    std::array<double, 3> betaG{};
    // DeviceFvLowMach: the velocity gradient runs every stage, and the initial
    // flux is laid by the caller, never projected (div m is not 0).
    bool lowMach = false;
};

// Tables of DeviceFvIncomp. nB counts NON-empty boundary faces; b in [0, nB)
// is FaceMesh face nI + bFace[b]. Vectors [3 n] are component-major.
struct FvIncompSetup
{
    int nC = 0, nI = 0, nB = 0, emptyAxis = -1;
    // true: every active face has d parallel to S (1e-12): no gradient in the
    // viscous flux, no pressure corrector.
    bool orthogonal = true;
    // true: no face fixes p (the pressure carries the constant null space).
    bool pureNeumann = true;
    bool hasConvective = false;
    double volume = 0, convArea = 0; // total, the convective faces' |S|
    double nonOrthMaxDeg = 0;
    std::vector<int> own, nei;
    std::vector<double> S, d, rO;   // [3 nI]; rO = Cf - C_own in the owner's frame
    std::vector<double> id2, cS;    // [nI] 1/|d|^2, d.S/|d|^2 (compact viscous)
    std::vector<double> wO;         // [nI] OpenFOAM linear weight of the owner
    std::vector<double> a, k;       // [nI] [3 nI] |S|^2/(S.d) (clamped), S - a d
    std::vector<int> bOwn, bFace;
    std::vector<double> bS, bR, bN; // [3 nB] S, Cf - C_own, unit normal
    std::vector<double> bCs, bA, bDn; // [nB] r.S/|r|^2, |S|^2/(S.r), S.r/|S|
    std::vector<double> bMagS;      // [nB]
    std::vector<std::uint8_t> bCode, bForce;
    std::vector<double> bU, bP;     // [3 nB] the fixed velocity, [nB] the fixed p
    std::vector<double> bUc;        // [nB] convective speed
    std::vector<double> bT;         // [nB] the fixed temperature
    std::vector<std::uint8_t> bTfix; // [nB] 1: T fixed, 0: zero heat flux
    std::vector<double> V, C, delta2; // [nC] [3 nC] [nC] filter width^2
    std::vector<int> cellStart;     // nC + 1
    // CSR, ascending face: internal f from its owner 2f, its neighbour 2f + 1,
    // boundary b: -(b + 1).
    std::vector<int> cellEnt;
    // Least-squares weights of the cell gradient on differences (FvGradWeights).
    std::vector<double> gO;         // [3 (nI + nB)]
    std::vector<double> gN;         // [3 nI]
    std::vector<double> slotA;      // pressure off-diagonal per graph slot (-a)
    std::vector<double> diagA;      // [nC] pressure diagonal
};

// The tables of fm under b. Pre: bindFvIncompBcs(b, fm) done (b.patches one
// entry per non-empty patch, order of fm.patches). Error (exit 3): a singular
// least-squares cell, S.d <= 0 (FvFaceGeom).
std::expected<FvIncompSetup, std::string> buildFvIncompSetup(const FaceMesh& fm, const FvIncompBcs& b,
                                                             int threads = 0);

// `dns tgv --solver fv` mesh distortion. None: the uniform box. Sine: the
// smooth CurvMesh::sinusoidal map (amp <= 0.06). Random: every node moved by
// a uniform hash in [-amp, amp] h per axis (z stays planar at nz == 1).
enum class FvPerturb : std::uint8_t { None, Sine, Random };

// Uniform in [-1, 1) from (seed, node, axis): the node hash of FvPerturb::Random.
double fvHashUnit(std::uint64_t seed, std::uint64_t node, int axis);

// The periodic box of g as a FaceMesh (translational cyclics on every axis, z
// `empty` at nz == 1), Morton order. Error (exit 3): the FaceMesh's.
std::expected<FaceMesh, std::string> fvPeriodicBox(const Grid& g, FvPerturb kind, double amp,
                                                   std::uint64_t seed = 0, int threads = 0);

// The structured O-grid's real cells as a FaceMesh: patches `wall` (j-) and
// `far` (j+), z cyclic or `empty` at nz == 1, Morton order. Pre: og from
// buildOGrid (not revolved).
std::expected<FaceMesh, std::string> fvOgridMesh(const OGrid& og, int threads = 0);

// The label of every report: "DNS" / "LES ... Cs ... (model)", the mesh, the
// scheme and the precision. les > 0 names Cs and the model.
std::string fvIncompLabel(const FvIncompControls& c, std::string_view mesh);

} // namespace dns
