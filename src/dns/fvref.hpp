// FvRef: the CPU FP64 reference of the FV path (docs/mesh.md; masterplan
// §2.3, phase P2). Compressible Navier-Stokes on any FaceMesh with the face
// physics of dns/fvflux.hpp: Kennedy-Gruber mean flux + face viscous terms,
// cell gradients by least squares or Green-Gauss (dns/fvgrad), boundaries as
// face fluxes only, SSP-RK3 with the `ca = 1 - cb` rule, the Sigma-form dt.
// It is the oracle the GPU FV path (P4) and the FP64 implicit residual (P7)
// are pinned against; on a uniform Cartesian FaceMesh it equals dns::Comp.
//
// Pass order (the GPU stage choreography): primitives -> boundary states ->
// [Y, omega, boundary Y] -> cell gradients of u, v, w, T [and Y] -> face fluxes
// F[5][nI+nB] (materialized) [and species fluxes] -> per-cell gather of its CSR
// faces in ascending face order, divided by V, plus additive cell sources.
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
#include "dns/fvrans.hpp"
#include "dns/transport.hpp"

#include <array>
#include <expected>
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
    // internal faces (fvGravityFace, fvHydroRho). Open patches and moving walls are
    // rejected: they have no hydrostatic ambient.
    std::array<double, 3> gravity{};
    FvChem chem{};        // species rows and the step; nY == 0 off
    FvRansConfig rans{};
    // Constant force per volume: rhs of m + f, of E + f.u (the channel's drive).
    std::array<double, 3> bodyForce{};
};

// Explicit dt limits, both per cell (LTS-ready), both Sigma forms:
//   convective  2 cfl V / sum_f (|u.S| + c |S|)    (cfl 1 = 1D CFL 1)
//   diffusive   diff rho V / (mu(T) max(4/3, gamma/Pr) sum_f corr.S)
// sum_f corr.S is the cell's compact-Laplacian diagonal, so diff 1 puts the
// checkerboard at lambda dt = -2 on ANY mesh and dimension (issue 15's
// lesson: never a per-axis 0.25 h^2). Sums over non-empty faces.
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
    std::vector<double> nut, f1;
};

class FvRef
{
public:
    // Pre: fm outlives the FvRef. Errors (user-triggerable: mesh, patch map,
    // BC values): bc count != patches, Empty <-> `empty` mismatch, invalid BC
    // values, a 2D mesh given out-of-plane BC velocities, bad geometry
    // (fvFaceGeom / fvGradWeights), a RANS closure without mu > 0 or with a
    // wallDist not positive on every cell. threads <= 0: hardware concurrency.
    static std::expected<FvRef, std::string> make(const FaceMesh& fm, FvConfig cfg, int threads = 0);

    const FaceMesh& mesh() const { return *fm_; }
    const FvConfig& config() const { return cfg_; }
    const FvFaceGeom& faceGeom() const { return geom_; }
    const FvGradWeights& gradWeights() const { return gw_; }
    // State rows: 5 + the closure's scalars.
    int nV() const { return nV_; }
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
    // non-null. RANS: mu_t adds at max(4/3, gamma/Pr_t), SA's bound takes rho
    // max(nu, K)/sigma. Errors: rho or p not positive-finite (a diverged state).
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
    void sizeEval(FvEval& ev) const;

    const FaceMesh* fm_ = nullptr;
    FvConfig cfg_;
    FvFaceGeom geom_;
    FvGradWeights gw_;
    std::vector<int> bcOf_;        // FvBc row per boundary face
    std::vector<int> patchOf_;     // patch per boundary face
    std::vector<double> diffCoef_; // per cell sum_f corr . S
    int workers_ = 1, nV_ = kFvMeanVars;
};

} // namespace dns
