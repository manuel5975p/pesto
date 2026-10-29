// FvRef: the CPU FP64 reference of the FV path (docs/mesh.md; masterplan
// §2.3, phase P2). Compressible Navier-Stokes on any FaceMesh with the face
// physics of dns/fvflux.hpp: Kennedy-Gruber mean flux + face viscous terms,
// cell gradients by least squares or Green-Gauss (dns/fvgrad), boundaries as
// face fluxes only, SSP-RK3 with the `ca = 1 - cb` rule, the Sigma-form dt.
// It is the oracle the GPU FV path (P4) and the FP64 implicit residual (P7)
// are pinned against; on a uniform Cartesian FaceMesh it equals dns::Comp.
//
// Pass order (the GPU stage choreography): primitives -> boundary states ->
// cell gradients of u, v, w, T -> face fluxes F[5][nI+nB] (materialized) ->
// per-cell gather of its CSR faces in ascending face order, divided by V,
// plus additive cell sources. Every pass writes only its own index, so the
// result is byte-identical for any thread count.
//
// The state is [nV][nC] with nV fixed at RUN time (5 mean-flow rows; RANS
// scalars and species append rows in later phases). P2 accepts nV == 5.
#pragma once

#include "dns/facemesh.hpp"
#include "dns/fvflux.hpp"
#include "dns/fvgrad.hpp"
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

struct FvConfig
{
    Transport tr;
    FvGradScheme grad = FvGradScheme::LeastSquares;
    FvFaceGradRule rule = FvFaceGradRule::AlongD;
    double alpha = 1.0;
    std::vector<FvBc> bc; // one per FaceMesh::patches entry; Empty exactly on `empty` patches
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
};

class FvRef
{
public:
    // Pre: fm outlives the FvRef. Errors (user-triggerable: mesh, patch map,
    // BC values): bc count != patches, Empty <-> `empty` mismatch, invalid BC
    // values, a 2D mesh given out-of-plane BC velocities, bad geometry
    // (fvFaceGeom / fvGradWeights). threads <= 0: hardware concurrency.
    static std::expected<FvRef, std::string> make(const FaceMesh& fm, FvConfig cfg, int threads = 0);

    const FaceMesh& mesh() const { return *fm_; }
    const FvConfig& config() const { return cfg_; }
    const FvFaceGeom& faceGeom() const { return geom_; }
    const FvGradWeights& gradWeights() const { return gw_; }

    // dq/dt = -(1/V) sum_f F + src (src nullptr: none). Pre: s, src sized
    // for the mesh, rho > 0 and p > 0 (stableDt checks).
    void rhs(const FvState& s, FvState& out, const FvState* src = nullptr) const;
    // rhs plus every intermediate.
    void evaluate(const FvState& s, FvEval& ev, FvState& out, const FvState* src = nullptr) const;

    // min over cells of the FvDtLimits bound; per-cell values into local when
    // non-null. Errors: rho or p not positive-finite (a diverged state).
    std::expected<double, std::string> stableDt(const FvState& s, FvDtLimits lim = {},
                                                std::vector<double>* local = nullptr) const;

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
    void gather(const FvEval& ev, const Stage& st, std::size_t lo, std::size_t hi) const;

    const FaceMesh* fm_ = nullptr;
    FvConfig cfg_;
    FvFaceGeom geom_;
    FvGradWeights gw_;
    std::vector<int> bcOf_;        // patch index per boundary face
    std::vector<double> diffCoef_; // per cell sum_f corr . S
    int workers_ = 1;
};

} // namespace dns
