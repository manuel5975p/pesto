// FvRef: the CPU FP64 reference of the FV path (docs/mesh.md; masterplan
// §2.3, phase P2). Compressible Navier-Stokes on any FaceMesh with the face
// physics of dns/fvflux.hpp: Kennedy-Gruber mean flux + face viscous terms,
// cell gradients by least squares or Green-Gauss (dns/fvgrad), boundaries as
// face fluxes only, SSP-RK3 with the `ca = 1 - cb` rule, the Sigma-form dt.
// It is the oracle the GPU FV path (P4) and the FP64 implicit residual (P7)
// are pinned against; on a uniform Cartesian FaceMesh it equals dns::Comp.
//
// Pass order (the GPU stage choreography): primitives -> boundary states ->
// cell gradients of u, v, w, T -> [LES mu_t] -> [TFLES F E, E/F, Omega: the step's
// first stage only] -> [Y, omega, boundary Y, grad Y] -> face fluxes F[5][nI+nB]
// (materialized) [and species fluxes, the thickened energy flux added to F_E] ->
// per-cell gather of its CSR faces in ascending face order, divided by V, plus
// additive cell sources. (Independent passes: their order moves no number.)
// Every pass writes only its own index, so the result is byte-identical for any
// thread count.
//
// The state is [nV][nC] with nV fixed at RUN time: 5 mean-flow rows, then the
// species (FvConfig::chem) or the RANS scalars (FvConfig::rans, dns/fvrans.hpp;
// docs/fvcomp.md "RANS"), never both. With a
// closure: primitives -> boundary states -> gradients -> scalar fields ->
// their boundary values -> their gradients -> mu_t and cell terms -> face
// fluxes -> gather -> scalar face fluxes -> point-implicit scalar gather.
#pragma once

#include "dns/facemesh.hpp"
#include "dns/fvflux.hpp"
#include "dns/fvgrad.hpp"
#include "dns/fvles.hpp"
#include "dns/fvmodules.hpp"
#include "dns/fvrans.hpp"
#include "dns/fvshock.hpp"
#include "dns/sgs.hpp"
#include "dns/shockflux.hpp"
#include "dns/tfles.hpp"
#include "dns/transport.hpp"

#include <array>
#include <expected>
#include <optional>
#include <string>
#include <vector>

namespace dns {

// Conservative FV state, [nV][nC] (row v of cell c at q[v * nC + c]).
struct FvState
{
    int nV = kFvMeanVars, nC = 0;
    std::vector<double> q;

    static FvState zeros(int nC, int nV = kFvMeanVars);
    double* var(int v) { return q.data() + std::size_t(v) * std::size_t(nC); }
    const double* var(int v) const { return q.data() + std::size_t(v) * std::size_t(nC); }
};

// The RANS closure of an FV run (docs/fvcomp.md "RANS"): model None = no scalar
// row, the laminar solver. wallDist: per cell, the distance to the nearest Wall
// face (dns/fvcomp_case fvRansWallDistance); d1 of omega_w = the owner's.
// phiInf: freestream specific scalars. transportOnly (TEST ONLY): upwind on the
// mass flux alone. Scalars' state rows: rho nu~ | rho k, rho omega.
struct FvRansConfig
{
    RansModel model = RansModel::None;
    std::vector<double> wallDist{};
    std::array<double, 2> phiInf{};
    double omegaFloor = 1e-10; // counted guard: k >= 0, omega >= floor rho
    bool transportOnly = false;
};

// LES of an FV run (cs > 0): mu_t = rho (C Delta)^2 op(grad u) per cell (dns/sgs.hpp,
// Delta^2 = fvLesDelta2), its face mean into mu, cp/prt scaled into kappa; a Slip face
// takes the owner's mu_t, a Wall face none. Species (docs/fvcomp.md "Species LES and
// TFLES"): + nut_f / Sc_t on every internal face, Sc_t = Pr_t = prt (ONE constant).
struct FvLes
{
    double cs = 0;
    SgsModel sgs = SgsModel::Smagorinsky;
    double prt = 0.9;
};

// The hydrostatic ambient column of an FV run under gravity g (docs/fvcomp.md
// "Hydrostatic ambient"): isothermal at T0, p_a(x) = candleAmbientP(p0, |g|, R T0, h),
// h = (x - x0).up the height above x0, up = -g/|g| (the structured candle's ONE
// definition: g along -x, x0 on the inflow plane, the same numbers). An open face reads
// it at its OWNER'S CENTROID, the face's height in the gravity scheme: a boundary face
// carries no gravity share (fvGravityFace is internal), so its pressure is the owner
// height's, as a Wall's is p_P. At rest the row is the cell's state bit for bit
// (Far: P == F returns F exactly), so the rest is exact on any mesh.
struct FvAmbient
{
    double p0 = 1, T0 = 1;
    std::array<double, 3> x0{};
};

// Shock capture of an FV run (docs/nozzle.md): one post-pass over the internal faces,
// F += |S| D(sigma_f) + eps4 (1 - sigma_f) |u_f.S| D3 (dns/fvflux.hpp fvShockFlux,
// fvD4Rate, fvD4Row), sigma from the pressure sensor along mesh lines (dns/fvshock), once
// per step on the step's input. Sense: the bytes alone, no flux term. force (TEST ONLY):
// bytes 255, sigma 1 everywhere: HLL + Davis, rhoCentralFoam's Kurganov + upwind.
struct FvShock
{
    ShockMode mode = ShockMode::None;
    double eps4 = 0;
    bool force = false;
};

// A boundary table: rows (a row per patch first), and per boundary face its row.
struct FvBcTable
{
    std::vector<FvBc> bc;
    std::vector<int> faceRow;
};

struct FvConfig
{
    Transport tr;
    FvCellGrad grad = FvCellGrad::LeastSquares;
    FvFaceGradRule rule = FvFaceGradRule::AlongD;
    double alpha = 1.0;
    // The FvBc table: one row per FaceMesh::patches entry (Empty exactly on `empty`
    // patches), then per-face rows of the same kind if bcRow says so.
    std::vector<FvBc> bc;
    // Empty: a face reads its patch's row. Else, per boundary face, its row of bc.
    std::vector<int> bcRow{};
    // Acceleration vector. Nonzero: the well-balanced face form of rho g, g.m on the
    // internal faces (fvGravityFace, fvHydroRho). Moving walls are rejected; Far and
    // Outflow patches need `ambient` and rows that are its column (fvAmbientRows).
    // Composes with species, LES, bodyForce (docs/fvcomp.md "Combinations"), not with RANS.
    std::array<double, 3> gravity{};
    // The hydrostatic ambient of the open patches under gravity (FvAmbient). Empty: no
    // open Far / Outflow patch may meet gravity. Set without gravity: an error.
    std::optional<FvAmbient> ambient{};
    FvChem chem{};        // species rows and the step; nY == 0 off
    FvRansConfig rans{};
    // Constant force per volume: rhs of m + f, of E + f.u (the channel's drive).
    std::array<double, 3> bodyForce{};
    FvLes les{};
    // The thickened flame (premixed species only; docs/fvcomp.md "Species LES and
    // TFLES"): F, E, Omega per cell from each step's INPUT state (the walk of
    // fvTflesWalk, dilate steps per direction), held over its three stages.
    // Internal faces: rho D = FE_f mu_f/Pr + (1 - Om_f) nut_f/Sc_t, and the energy
    // flux gains -(FE_f - 1) kappa_f g_T.S + Om_f nut_f (cp/Pr_t) g_T.S (additive);
    // omega -> (E/F) omega. Boundary faces: unthickened. Empty: off.
    std::optional<TflesSetup> tfles{};
    // --turb (M5, dns/fvmodules Turb): the open patches' bcRow rows carry the swept
    // synthetic box, rewritten by the caller between steps. No stage term: the
    // inventory's pairs only.
    bool turb = false;
    FvShock shock{};
};

// The modules cfg switches on (dns/fvmodules; the precision and time axes are the
// device's): what FvRef::make checks against the inventory.
FvModuleSet fvModulesOf(const FvConfig& cfg);

// p_a at x (FvAmbient). g zero: p0. Pre: p0, T0, Rgas > 0.
double fvAmbientP(const FvAmbient& a, const std::array<double, 3>& g, const double x[3], double Rgas);
// The ambient's conserved state at x moving with u: rho_a = p_a/(R T0), rho_a u, p_a/(gamma - 1)
// + rho_a |u|^2/2. The rows and the rest IC both read it: the same bits.
std::array<double, 5> fvAmbientConserved(const FvAmbient& a, const std::array<double, 3>& g, const double x[3],
                                         const double u[3], double gamma, double Rgas);
// t with a row of its own for every face of a Far or Outflow row: Far's qFar the ambient
// at the owner's centroid moving with the row's velocity, Outflow's pOut p_a there; every
// other member and face unchanged. faceRow empty: each face's patch row first.
FvBcTable fvAmbientRows(const FaceMesh& fm, FvBcTable t, const FvAmbient& a, const std::array<double, 3>& g,
                        double gamma, double Rgas);
// The ambient at rest on every cell: rows rho, 0, 0, 0, E (nV rows, the rest 0).
FvState fvAmbientState(const FaceMesh& fm, const FvAmbient& a, const std::array<double, 3>& g, double gamma,
                       double Rgas, int nV);

// Explicit dt limits, both per cell (LTS-ready), both Sigma forms:
//   convective  2 cfl V / sum_f (|u.S| + c |S|)    (cfl 1 = 1D CFL 1)
//   diffusive   diff rho V / (mu(T) max(4/3, gamma/Pr) sum_f corr.S)
// sum_f corr.S is the cell's compact-Laplacian diagonal, so diff 1 puts the
// checkerboard at lambda dt = -2 on ANY mesh and dimension (issue 15's
// lesson: never a per-axis 0.25 h^2). Sums over non-empty faces. A `wedge` side is
// out of the convective sum and its corr.S is the hoop term V sin^2(h)/d_z^2
// (docs/nozzle.md): neither carries an azimuthal transit the axisymmetric field has.
struct FvDtLimits
{
    double cfl = 0.4, diff = 1.0;
};

// Everything one evaluation computes (the GPU pin compares any of it).
struct FvEval
{
    std::array<std::vector<double>, 7> prim; // rho, u, v, w, e, p, T per cell
    std::vector<FvPrim<double>> bnd;         // boundary state per boundary face
    std::array<std::vector<double>, 4> bphi; // its u, v, w, T (the gradients' face values)
    std::array<std::vector<double>, 12> grad; // [field * 3 + axis][nC]; fields u, v, w, T; 0 if inviscid
    std::array<std::vector<double>, 5> F;    // face fluxes, [5][nI + nB]
    std::array<std::vector<double>, 4> Fg;   // gravity's face terms [4][nI] (fvGravityFace), else empty
    // Species (chem.nY rows, else empty): Y [nY][nC], boundary Y [nY][nB], grad Y
    // [3 nY][nC] (row 3 s + axis), the one omega [nC], fluxes [nY][nI + nB].
    std::vector<std::vector<double>> Y, bY, gradY, Fy;
    std::vector<double> omega;
    // RANS only. tfld, tbnd: kFvRansFields (SA nu~, rho; SST k, omega) per cell,
    // per boundary face; tgrad [field * 3 + axis][nC]; tsrc R0, R1, J0, J1 per
    // cell; Ft the scalars' face fluxes [2][nI + nB].
    std::array<std::vector<double>, 2> tfld, tbnd, Ft;
    std::array<std::vector<double>, 6> tgrad;
    std::array<std::vector<double>, 4> tsrc;
    std::vector<double> nut, f1; // nut: RANS or LES mu_t per cell
    // TFLES (FvConfig::tfles): F E, E/F, Omega per cell (the step's coefficients),
    // and their scratch: the dilation's ping-pong, curl u [3][nC], Colin's grid u'.
    std::vector<double> fe, eof, om;
    std::array<std::vector<double>, 2> tfWork;
    std::array<std::vector<double>, 3> curl;
    std::vector<double> uh;
    std::vector<std::uint8_t> sigma; // shock sensor bytes per cell, else empty
};

class FvRef
{
public:
    // Pre: fm outlives the FvRef. Errors (user-triggerable: mesh, patch map,
    // BC values): bc count != patches, Empty <-> `empty` mismatch, invalid BC
    // values, a 2D mesh given out-of-plane BC velocities, bad geometry
    // (fvFaceGeom / fvGradWeights), a RANS closure without mu > 0 or with a
    // wallDist not positive on every cell, a module pair or requirement the inventory
    // refuses (dns/fvmodules, fvModulesOf). threads <= 0: hardware concurrency.
    static std::expected<FvRef, std::string> make(const FaceMesh& fm, FvConfig cfg, int threads = 0);

    const FaceMesh& mesh() const { return *fm_; }
    const FvConfig& config() const { return cfg_; }
    const FvFaceGeom& faceGeom() const { return geom_; }
    const FvGradWeights& gradWeights() const { return gw_; }
    // State rows: 5 + the closure's scalars.
    int nV() const { return nV_; }
    // TFLES only: the sensor walk and edge layers (empty otherwise).
    const FvTflesWalk& tflesWalk() const { return walk_; }
    // --shock only: the sensor's mesh lines (empty otherwise).
    const FvShockLines& shockLines() const { return lines_; }
    // LES or TFLES: Delta^2 per cell (empty otherwise).
    const std::vector<double>& lesDelta2() const { return delta2_; }
    // Per cell sum_f corr . S over active faces: the diffusive dt's denominator.
    const std::vector<double>& diffCoef() const { return diffCoef_; }
    // Row of the FvBc table per boundary face; the patch of that face.
    const std::vector<int>& bcOf() const { return bcOf_; }
    const std::vector<int>& patchOf() const { return patchOf_; }

    // dq/dt = -(1/V) sum_f F + src (src nullptr: none). Pre: s, src sized
    // for the mesh, rho > 0 and p > 0 (stableDt checks).
    void rhs(const FvState& s, FvState& out, const FvState* src = nullptr) const;
    // rhs plus every intermediate.
    void evaluate(const FvState& s, FvEval& ev, FvState& out, const FvState* src = nullptr) const;

    // min over cells of the FvDtLimits bound; per-cell values into local when
    // non-null. RANS, LES: mu_t adds at max(4/3, gamma/Pr_t), SA's bound takes rho
    // max(nu, K)/sigma; TFLES: mu at max(4/3, FE gamma/Pr), FE the cell's of s.
    // Errors: rho or p not positive-finite (a diverged state).
    std::expected<double, std::string> stableDt(const FvState& s, FvDtLimits lim = {},
                                                std::vector<double>* local = nullptr) const;

    // Explicit chemistry bound, the structured chemStableDt: k / max_c J_c, J_c =
    // A e^(-Ta/T) (1 + q Y Ta / (cv T^2)), two reactants A e^(-Ta/T) rho (Y_O + s
    // Y_F + q Y_F Y_O Ta / (cv T^2)), Y floored. +inf without a reaction.
    double chemDt(const FvState& s, double k) const;

    // `steps` SSP-RK3 steps of global dt on persistent workers (one spawn per
    // call); out = ca a + cb b + cr R with ca = 1 - cb. dt within stableDt.
    void march(FvState& s, double dt, int steps, const FvState* src = nullptr) const;
    void step(FvState& s, double dt, const FvState* src = nullptr) const { march(s, dt, 1, src); }

private:
    struct Stage; // one fused stage's operands
    void prims(const double* q, FvEval& ev, std::size_t lo, std::size_t hi) const;
    void bndStates(FvEval& ev, std::size_t lo, std::size_t hi) const;
    void grads(FvEval& ev, std::size_t lo, std::size_t hi) const;
    void faces(FvEval& ev, std::size_t lo, std::size_t hi) const;
    void speciesCells(const double* q, FvEval& ev, std::size_t lo, std::size_t hi) const;
    void speciesBnd(FvEval& ev, std::size_t lo, std::size_t hi) const;
    void speciesGrads(FvEval& ev, std::size_t lo, std::size_t hi) const;
    void speciesFaces(FvEval& ev, std::size_t lo, std::size_t hi) const;
    void gather(const FvEval& ev, const Stage& st, std::size_t lo, std::size_t hi) const;
    void ransFields(const double* q, FvEval& ev, std::size_t lo, std::size_t hi) const;
    void ransBnd(FvEval& ev, std::size_t lo, std::size_t hi) const;
    void ransGrads(FvEval& ev, std::size_t lo, std::size_t hi) const;
    void ransCells(FvEval& ev, std::size_t lo, std::size_t hi) const;
    void ransFaces(FvEval& ev, std::size_t lo, std::size_t hi) const;
    void ransGather(const FvEval& ev, const Stage& st, std::size_t lo, std::size_t hi) const;
    void lesCells(FvEval& ev, std::size_t lo, std::size_t hi) const;
    // Sensor bytes of ev's primitives (serial), the flux post-pass on state q.
    void shockSense(FvEval& ev) const;
    void shockFaces(const double* q, FvEval& ev, std::size_t lo, std::size_t hi) const;
    // TFLES coefficients of state q: raw sensor and curl, a dilation pass per walked
    // axis, Colin's u', the combine. Every pass writes its own cells; sync between.
    void tflesRaw(const double* q, FvEval& ev, std::size_t lo, std::size_t hi) const;
    void tflesDilate(FvEval& ev, int pass, std::size_t lo, std::size_t hi) const;
    void tflesColin(FvEval& ev, std::size_t lo, std::size_t hi) const;
    void tflesCombine(FvEval& ev, std::size_t lo, std::size_t hi) const;
    template <class Sync>
    void tflesFields(const double* q, FvEval& ev, std::size_t c0, std::size_t c1, Sync&& sync) const;
    void sizeEval(FvEval& ev) const;

    const FaceMesh* fm_ = nullptr;
    FvConfig cfg_;
    FvFaceGeom geom_;
    FvGradWeights gw_;
    std::vector<int> bcOf_;        // FvBc row per boundary face
    std::vector<int> patchOf_;     // patch per boundary face
    std::vector<double> diffCoef_; // per cell sum_f corr . S
    FvTflesWalk walk_;             // TFLES
    FvShockLines lines_;           // --shock
    std::vector<double> delta2_;   // LES, TFLES (Colin's h = sqrt Delta^2)
    int workers_ = 1, nV_ = kFvMeanVars;
};

} // namespace dns
