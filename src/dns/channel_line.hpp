// FP64 host restatement of the GPU channel's y-LINE operator in conserved
// variables: the preconditioner of the implicit steady RANS march
// (--steady-scheme implicit, docs/rans.md "Implicit pseudo-time").
//
// It restates what the GPU residual-mode stage computes on one x/z-uniform
// column of CurvMesh::channelY: the Kennedy-Gruber flux and the face-viscous
// flux with mu_t (gpu/internal/comp_flux.hpp faceFluxCurv), the scalar
// upwind convection on the verbatim mass flux, face diffusion and cell terms
// (gpu/rans.hip), the isothermal ghost and the scalar ghosts
// (gpu/internal/wall_ghost.hpp, gpu/comp_channel.hip), mu_t odd / F1 even
// across the ghosts, and the body force. Closure algebra is CALLED from
// dns/rans.hpp and dns/sgs.hpp, never restated.
//
// It only shapes the PATH: the update's right-hand side is always the GPU's
// residual, so a drift between the two costs iterations, never the answer
// (gpu_rans_implicit I1/I2 pin it). This is exact only for the x/z-
// homogeneous RANS channel mean; do not generalize it to a non-homogeneous
// case.
//
// Column layout: rows() grid rows (ghosts 0 and rows()-1 included), each
// fields() = 5 + nT values in DeviceCompState order (rho, rho u, rho v,
// rho w, E, then rho nu~ | rho k, rho omega), row-major [j][v]. The Newton
// unknowns are the 4 + nT fields other than rho w: at w = 0, R_mz is linear
// in w and every other row even in w, so the two decouple exactly and w
// stays 0.
#pragma once

#include "dns/blockband.hpp"
#include "dns/rans.hpp"

#include <span>
#include <vector>

namespace dns {

struct CurvMetrics;
struct Grid;

struct ChannelLineCase
{
    RansModel model = RansModel::SaNeg;
    double gamma = 1.4, Rgas = 1.0, mu = 0, Twall = 0, fBody = 1.0;
    double omegaCoef = 0; // SST: 60 mu / (beta1 d1^2); omega_w = coef / rho_w
};

class ChannelLine
{
public:
    // The line of column (0, 0) of the channel mesh.
    // Pre: m from CurvMetrics::build(CurvMesh::channelY(g, ...)) (x/z-
    // uniform, y faces normal to y), wallDist.size() == g.size(),
    // c.model != None, mu > 0, Twall > 0.
    static ChannelLine make(const CurvMetrics& m, const Grid& g,
                            std::span<const double> wallDist,
                            const ChannelLineCase& c);

    int rows() const { return N_; }
    int fields() const { return 5 + nT_; }
    int unknowns() const { return 4 + nT_; }
    // Half-bandwidth in rows: SA's operator is tridiagonal (mu_t pointwise,
    // every gradient reaches j +- 1); SST's mu_t and F1 carry cell-centre
    // gradients, so j +- 2.
    int band() const { return nT_ == 2 ? 2 : 1; }
    bool sst() const { return nT_ == 2; }
    RansModel model() const { return c_.model; }
    // Field of Newton unknown u: {rho, mx, my, E, t0, t1} -> {0, 1, 2, 4, 5, 6}.
    static int field(int u) { return u < 3 ? u : u + 1; }

    // Ghost rows 0 and rows()-1 from rows 1 and rows()-2: the isothermal
    // ghost (T0 = max(2 Tw - T1, Tw/20), p0 = p1, rho0 = p0/(R T0),
    // m0 = -rho0 u1) and the scalar ghosts rebuilt with the ghost's density.
    // Pre: rho > 0 on rows 1 and rows()-2.
    void wall(std::span<double> q) const;

    // Per-volume rhs of the fluid rows (the GPU residual-mode stage); ghost
    // rows' entries 0. Pre: q wall-consistent, rho > 0, SST omega > 0.
    void residual(std::span<const double> q, std::span<double> R) const;

    // dR/dq over the fluid rows and the unknowns, by (2 band + 1)-colour
    // CENTRAL FD in FP64 (step kImplicitFdStep times each unknown's scale;
    // pseudo_time.hpp says why not forward); the wall runs after every
    // perturbation, so the ghost chain rule is included. n = rows() - 2,
    // B = unknowns(), K = band(). Pre: as residual.
    BlockBand jacobian(std::span<const double> q) const;

    // Fluid mass per unit wall area, sum rho_j vol_j / S_j (the oracle's
    // totalMass definition).
    double mass(std::span<const double> q) const;

    // Pseudo-time step per row: cfl / ((|u| + c) max_a aov_a), the scheme's
    // continuation scale (no diffusive term, deliberately not stableDt).
    // Pre: dtau.size() == rows(). Post: ghost entries 0.
    void pseudoStep(std::span<const double> q, double cfl,
                    std::span<double> dtau) const;

    // q <- q + alpha dq on the fluid rows (dq in the unknown layout; rho w
    // untouched), then every fluid field scaled by M0 / mass(q) and the wall
    // reapplied: the projection onto the IC mass, which leaves every specific
    // quantity (u, T, nu~, k, omega) unchanged.
    // Pre: dq.size() == (rows() - 2) * unknowns(), M0 > 0.
    void advance(std::span<double> q, std::span<const double> dq, double alpha,
                 double M0) const;
    // Every fluid field scaled by M0 / mass(q), wall reapplied.
    void project(std::span<double> q, double M0) const;

    // Finite everywhere, rho > 0 and p > 0 on every row; SST rho k >= 0 and
    // rho omega > 0 on the fluid rows.
    bool admissible(std::span<const double> q) const;

private:
    ChannelLine() = default;

    ChannelLineCase c_;
    int N_ = 0, nT_ = 0;
    double kap_ = 0, cpPrt_ = 0;
    // Per grid row: +y face area and d eta/dy, 1/vol, row height vol/S, wall
    // distance, and the largest per-axis aov (the acoustic width).
    std::vector<double> S_, Jy_, iv_, dy_, d_, aov_;
};

} // namespace dns
