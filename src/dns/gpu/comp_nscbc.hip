#include "dns/gpu/comp_nscbc.hpp"

#include "dns/gpu/internal/comp_flux.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/stage_write.hpp"
#include "dns/gpu/farpot.hpp"
#include "dns/gpu/index.hpp"
#include "dns/remap.hpp"

#include <cassert>
#include <cmath>
#include <memory>
#include <type_traits>

namespace dns::gpu {

namespace {

// Dead-band width (fraction of local sound speed) of the tanh blend between
// the outgoing (one-sided) and incoming (relax) lateral entropy/shear forms;
// removes the discontinuity as un crosses 0. Promote to Config if a scan wants
// it.
constexpr Real kLatBand = Real(0.01);

// Clamp on X = Tp/Lout in the angle-adaptive transverse coefficient: 18 is
// cos(theta) ~ 0.053, about 87 deg, past which a face is not absorbing
// anything anyway and the ratio is dominated by whatever noise is on Lout.
constexpr Real kBetaXMax = Real(18);

// One thread per open-boundary cell (x extremes always; y/z extremes when
// open). Per boundary axis n the characteristic amplitudes in +axis
// orientation (lam1 = un-c, lam2..4 = un, lam5 = un+c; t1/t2 = tangents):
//   L1 = lam1 (dp/dn - rho c dun/dn)   L5 = lam5 (dp/dn + rho c dun/dn)
//   L2 = un (c^2 drho/dn - dp/dn)      L3 = un dv_t1/dn   L4 = un dv_t2/dn
// LODI contribution to the primitive time derivatives:
//   drho/dt = -(L2 + (L1+L5)/2)/c^2    dun/dt = -(L5-L1)/(2 rho c)
//   dv_t/dt = -L3, -L4                 dp/dt = -(L1+L5)/2
// Outgoing amplitudes from first-order one-sided differences; incoming ones
// modeled per face: x-lo stiff inflow relax, x-hi/lateral acoustic anchor
// sig c/L (p - pInf) with the Yoo-Im transverse correction, lateral
// entropy/shear upwinded on sign(outward un) with a gentle freestream relax
// on the inflowing side. Cells on several open faces sum the per-axis LODI
// terms; non-boundary axes keep the exact faceFlux divergence.
//
// The PointRemap R is a template argument (dns/remap.hpp): with any remap
// other than IdentityRemap every one-sided normal derivative, transverse
// derivative and potential-target position is evaluated at PHYSICAL cell
// centres, and the retained transverse flux divergence switches to
// faceFluxCurv on the metric tables cv (the curvilinear interior scheme,
// exactly). The IdentityRemap instantiation keeps the historical uniform
// arithmetic UNTOUCHED (cv unused), so the validated uniform NSCBC runs are
// bit-identical. Comp: the compensated write (internal/stage_write.hpp).
// Prof: per-row x-lo inflow targets (Config::inflowU/Jet/Rho): u target
// inU + pls inJ, entropy target inR (with invRT > 0: p inR). Its own
// instantiation, so the uniform-target kernel is the shipped code.
// Hydro: the ambient hydrostatic incoming waves (Config::hydroLo/Hi: gLo,
// gHi), its own instantiation too.
template <class Remap, bool Comp, bool Prof, bool Hydro = false>
__global__ void kNscbcCells(GpuGrid g, Remap R, CurvView cv, ViscView vv,
                            Real gam, PrimView q,
                            int openY, int openZ, int fr, Real invLx, Real invLy,
                            Real invLz, Real sigma, Real etaIn, Real etaInS,
                            Real omM2,
                            Real betaT, int bAuto, Real sigLat, Real etaLat, Real uT,
                            Real rhoI, Real pI, int pot, Real Q, Real dbl,
                            Real xc, Real yc, Real zc, Real invRT,
                            StageCombine cmb, const Real* linHi,
                            const Real* inU, const Real* inJ, const Real* inR,
                            Real pls, Real gLo, Real gHi,
                            Real* orho, Real* omx, Real* omy, Real* omz,
                            Real* oE)
{
    constexpr bool remapped = !std::is_same_v<Remap, IdentityRemap>;
    // Disjoint boundary enumeration: x faces own their full slabs, y faces
    // the interior-x strips, z faces the interior-x/y core.
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t nAh = std::size_t(g.ny) * g.nz;
    const std::size_t nBh = openY ? std::size_t(g.nx - 2) * g.nz : 0;
    // z-slab spans full j when y is walled (slab B empty there, no overlap);
    // bit-identical to (ny - 2) when openY.
    const std::size_t nCh =
        openZ ? std::size_t(g.nx - 2) * (openY ? g.ny - 2 : g.ny) : 0;
    if (t >= 2 * (nAh + nBh + nCh)) return;
    int i, j, k;
    if (t < 2 * nAh) {
        const bool hi = t >= nAh;
        const std::size_t tt = hi ? t - nAh : t;
        i = hi ? g.nx - 1 : 0;
        j = int(tt % g.ny);
        k = int(tt / g.ny);
    } else if (t < 2 * (nAh + nBh)) {
        const std::size_t tb = t - 2 * nAh;
        const bool hi = tb >= nBh;
        const std::size_t tt = hi ? tb - nBh : tb;
        j = hi ? g.ny - 1 : 0;
        i = 1 + int(tt % (g.nx - 2));
        k = int(tt / (g.nx - 2));
    } else {
        const std::size_t tc = t - 2 * (nAh + nBh);
        const bool hi = tc >= nCh;
        const std::size_t tt = hi ? tc - nCh : tc;
        k = hi ? g.nz - 1 : 0;
        i = 1 + int(tt % (g.nx - 2));
        j = (openY ? 1 : 0) + int(tt / (g.nx - 2));
    }
    const std::size_t c = gidx(g, i, j, k);
    // Fully-solid floor rows at the x planes: the ghost wall owns them; keep
    // the interior rhs instead of a characteristic form built from solid
    // values. fr == -1 = no floor.
    if (j <= fr) return;

    const Real r = q.rho[c], p = q.p[c];
    const Real vc[3] = {q.u[c], q.v[c], q.w[c]};
    const Real c2 = gam * p / r, cs = sqrt(c2);
    const int ijk[3] = {i, j, k};
    const int nn[3] = {g.nx, g.ny, g.nz};
    const Real hh[3] = {g.hx, g.hy, g.hz};
    const Real invL[3] = {invLx, invLy, invLz};
    const int openA[3] = {1, openY, openZ};
    const Real* vf[3] = {q.u, q.v, q.w};

    // Physical cell-centre coordinate along axis d of the UNWRAPPED index
    // triple: a periodic-compatible remap yields the correct periodic-image
    // position by itself, and TanhStretchY's smooth extension covers the
    // wall-ghost rows (same convention as kVorticityZRemap).
    auto coordD = [&](int ii, int jj, int kk, int d) {
        const double X[3] = {(ii + 0.5) * double(g.hx),
                             (jj + 0.5) * double(g.hy),
                             (kk + 0.5) * double(g.hz)};
        double x[3];
        R.map(X, x);
        return x[d];
    };

    auto onFace = [&](int a) {
        return openA[a] && (ijk[a] == 0 || ijk[a] == nn[a] - 1);
    };
    // Transverse derivative along axis d: central, one-sided when the
    // neighbor would wrap across another open face (edge/corner cells).
    auto ct = [&](const Real* f, int d) {
        const int px = d == 0, py = d == 1, pz = d == 2;
        const Real fp = f[gidx(g, i + px, j + py, k + pz)];
        const Real fm = f[gidx(g, i - px, j - py, k - pz)];
        if constexpr (remapped) {
            if (openA[d] && ijk[d] == 0)
                return (fp - f[c]) / Real(coordD(i + px, j + py, k + pz, d) -
                                          coordD(i, j, k, d));
            if (openA[d] && ijk[d] == nn[d] - 1)
                return (f[c] - fm) / Real(coordD(i, j, k, d) -
                                          coordD(i - px, j - py, k - pz, d));
            return (fp - fm) / Real(coordD(i + px, j + py, k + pz, d) -
                                    coordD(i - px, j - py, k - pz, d));
        } else {
            if (openA[d] && ijk[d] == 0) return (fp - f[c]) / hh[d];
            if (openA[d] && ijk[d] == nn[d] - 1) return (f[c] - fm) / hh[d];
            return (fp - fm) * (Real(0.5) / hh[d]);
        }
    };

    // Far-field target: uniform freestream, or (pot) freestream + wake source
    // + displacement doublet with the Bernoulli pressure. tgt == {uT,0,0} and
    // pPt == pI when pot is off, keeping every form below bit-identical.
    Real tgt[3] = {uT, 0, 0};
    Real pPt = pI;
    if (pot) {
        Real cx_ = (Real(i) + Real(0.5)) * g.hx;
        Real cy_ = (Real(j) + Real(0.5)) * g.hy;
        Real cz_ = (Real(k) + Real(0.5)) * g.hz;
        if constexpr (remapped) {
            cx_ = Real(coordD(i, j, k, 0));
            cy_ = Real(coordD(i, j, k, 1));
            cz_ = Real(coordD(i, j, k, 2));
        }
        Real du, dv, dw;
        farPotential(g.nz == 1, cx_ - xc, cy_ - yc, cz_ - zc, Q, dbl,
                     Real(4) * g.hx * g.hx, du, dv, dw);
        tgt[0] += du;
        tgt[1] = dv;
        tgt[2] = dw;
        pPt = pI + Real(0.5) * rhoI *
                       (uT * uT - (tgt[0] * tgt[0] + tgt[1] * tgt[1] +
                                   tgt[2] * tgt[2]));
    }

    Real drdt = 0, dvdt[3] = {0, 0, 0}, dpdt = 0;
    Real trho = 0, tm[3] = {0, 0, 0}, tE = 0;

    for (int a = 0; a < 3; ++a) {
        const int ox = a == 0, oy = a == 1, oz = a == 2;
        if (!onFace(a)) { // exact transverse flux divergence (z cancels at
            if constexpr (remapped) {                             // nz == 1)
                const Flux fh = faceFluxCurv(g, vv, q, cv, i, j, k, a);
                const Flux fl =
                    faceFluxCurv(g, vv, q, cv, i - ox, j - oy, k - oz, a);
                const Real iv = cv.invVol[c]; // fluxes carry the face area
                trho -= (fh.rho - fl.rho) * iv;
                tm[0] -= (fh.mx - fl.mx) * iv;
                tm[1] -= (fh.my - fl.my) * iv;
                tm[2] -= (fh.mz - fl.mz) * iv;
                tE -= (fh.E - fl.E) * iv;
            } else {
                const Flux fh = faceFlux(g, vv, q, i, j, k, a);
                const Flux fl =
                    faceFlux(g, vv, q, i - ox, j - oy, k - oz, a);
                trho -= (fh.rho - fl.rho) / hh[a];
                tm[0] -= (fh.mx - fl.mx) / hh[a];
                tm[1] -= (fh.my - fl.my) / hh[a];
                tm[2] -= (fh.mz - fl.mz) / hh[a];
                tE -= (fh.E - fl.E) / hh[a];
            }
            continue;
        }
        const bool hi = ijk[a] == nn[a] - 1;
        const std::size_t cn =
            gidx(g, i + (hi ? -ox : ox), j + (hi ? -oy : oy), k + (hi ? -oz : oz));
        // (larger coord) - (smaller coord), one cell, both sides; the divisor
        // is the physical centre-to-centre distance on a remapped mesh.
        Real hnrm = hh[a];
        if constexpr (remapped)
            hnrm = Real(fabs(coordD(i + (hi ? -ox : ox), j + (hi ? -oy : oy),
                                    k + (hi ? -oz : oz), a) -
                             coordD(i, j, k, a)));
        // First-order one-sided ON PURPOSE: a 3-point second-order dn() was
        // prototyped and measured 2026-07-30 on the full gpu_nscbc_order angle
        // sweep -- every reflected-energy figure moved in the 4th decimal,
        // some slightly worse. The closure (beta/abc), not this stencil, owns
        // the boundary error; do not re-derive the upgrade.
        auto dn = [&](const Real* f) {
            return (hi ? f[c] - f[cn] : f[cn] - f[c]) / hnrm;
        };
        const int t1 = a == 0 ? 1 : 0, t2 = a == 2 ? 1 : 2;
        const Real un = vc[a];
        const Real dun = dn(vf[a]), dp_ = dn(q.p), dr_ = dn(q.rho);

        // Thermal inflow (invRT = 1/(R TInf), 0 = off): the entropy target is
        // the density the target temperature implies at the LOCAL pressure.
        const Real rTgt = invRT > Real(0) ? p * invRT : rhoI;
        Real L1, L2, L3, L4, L5;
        if (a == 0 && !hi) { // stiff subsonic inflow: only L1 leaves
            Real uIn = tgt[0], rIn = rTgt;
            if constexpr (Prof) {
                const std::size_t row = std::size_t(j) + std::size_t(g.ny) * k;
                uIn = inU[row] + pls * inJ[row];
                rIn = invRT > Real(0) ? p * inR[row] : inR[row];
            }
            // etaIn is sized against the freestream SAG -- an argument about
            // L5 (the normal velocity anchor) alone. The lambda = u family
            // (entropy L2, shear L3/L4) rides its own rate etaInS
            // (--nscbc-etain-shear), which defaults to etaIn: same value,
            // same arithmetic, bit-identical. Softening it makes the inflow
            // transparent to vorticity and entropy without letting the
            // freestream sag.
            L1 = (un - cs) * (dp_ - r * cs * dun);
            L2 = etaInS * cs * c2 * invL[0] * (r - rIn);
            L3 = etaInS * cs * invL[0] * (vc[t1] - tgt[t1]);
            L4 = etaInS * cs * invL[0] * (vc[t2] - tgt[t2]);
            L5 = etaIn * omM2 * r * c2 * invL[0] * (un - uIn);
            if constexpr (Hydro) L5 += (un + cs) * gLo;
        } else { // outflow / lateral far field
            // Yoo-Im transverse correction on the incoming acoustic wave:
            // without partial cancellation of the transverse forcing, wake
            // vortices crossing the outflow pump a growing global mode
            // (measured at Re800: cl amplitude 2 -> 13 instead of the
            // sponge's steady 1.9). Incoming invariant is p -+ rho c un at
            // the hi/lo face, hence the s sign on the Tun term.
            Real Tp = 0, Tun = 0;
            for (int d = 0; d < 3; ++d) {
                if (d == a) continue;
                Tp += vc[d] * ct(q.p, d) + gam * p * ct(vf[d], d);
                Tun += vc[d] * ct(vf[a], d);
            }
            const Real s = hi ? Real(1) : Real(-1);
            const Real sig = a == 0 ? sigma : sigLat;
            const Real om = a == 0 ? omM2 : Real(1); // lateral normal M ~ 0
            // This face's OUTGOING acoustic amplitude, hoisted: with bAuto it
            // is also what tells the face its own angle of incidence.
            const Real Lout = hi ? (un + cs) * (dp_ + r * cs * dun)
                                 : (un - cs) * (dp_ - r * cs * dun);
            const Real Tt = Tp - s * r * cs * Tun;
            // beta_opt = betaT 2/(2 + Tt/Lout): exact for one oblique wave at
            // any angle (comp_nscbc.hpp). X <= 0 gives betaT back BIT-FOR-BIT,
            // which is every normal-incidence face, so this is a refinement of
            // the constant form in the oblique regime and nothing else.
            Real bt = betaT;
            if (bAuto) {
                const Real X = Lout != Real(0) ? Tt / Lout : Real(0);
                bt = betaT * Real(2) /
                     (Real(2) + fmin(fmax(X, Real(0)), kBetaXMax));
            }
            // linHi (gpu/nscbc_modal.hpp) carries the SAME product with the
            // coefficient formed per transverse DCT mode instead of per cell;
            // it owns the x-hi plane alone. Null = evaluate it here, which at
            // bAuto == 0 is the shipped arithmetic bit-for-bit.
            const Real trans = linHi && a == 0 && hi
                                   ? linHi[j + std::size_t(g.ny) * k]
                                   : bt * Tt;
            Real Lin = sig * om * cs * invL[a] * (p - pPt) - trans;
            if constexpr (Hydro) {
                if (a == 0) Lin += (un - cs) * gHi; // x-hi: the outflow
            }
            // Hydro (the candle's open top): the x-hi face takes the lateral
            // blend below -- sustained backflow (the plume's entrainment
            // draws ambient air DOWN through the top) made the one-sided
            // entropy wave a downwind difference: scouts diverged at t 12-14
            // with T 1.3-1.4 T_ad at the outflow, backflow -30 U.
            if (a == 0 && !Hydro) { // x-outflow: entropy/shear leave, one-sided
                // Deliberately NO backflow model here. A lateral-style
                // etaOut relax blend was prototyped and measured 2026-07-30
                // (gpu_comp_nscbc backflow rig, vortex swirl up to 4x the
                // freestream driving real backflow through the face): the
                // residual moved in the 4th decimal at every rate. These
                // amplitudes are PROPORTIONAL to un, so they shut off
                // smoothly through a stagnation on their own -- there is
                // nothing for a blend to fix. Do not re-derive it.
                L2 = un * (c2 * dr_ - dp_);
                L3 = un * dn(vf[t1]);
                L4 = un * dn(vf[t2]);
            } else { // lateral: tanh blend outgoing compute <-> incoming relax
                // (C-infinity across un = 0; both forms vanish at freestream so
                // the fixed point is exact for any weight).
                const Real Lout2 = un * (c2 * dr_ - dp_);
                const Real Lout3 = un * dn(vf[t1]);
                const Real Lout4 = un * dn(vf[t2]);
                const Real Lin2 = etaLat * cs * c2 * invL[a] * (r - rTgt);
                const Real Lin3 = etaLat * cs * invL[a] * (vc[t1] - tgt[t1]);
                const Real Lin4 = etaLat * cs * invL[a] * (vc[t2] - tgt[t2]);
                const Real w =
                    Real(0.5) * (Real(1) + tanh(s * un / (kLatBand * cs)));
                L2 = w * Lout2 + (Real(1) - w) * Lin2;
                L3 = w * Lout3 + (Real(1) - w) * Lin3;
                L4 = w * Lout4 + (Real(1) - w) * Lin4;
            }
            if (hi) {
                L5 = Lout;
                L1 = Lin;
            } else {
                L1 = Lout;
                L5 = Lin;
            }
        }

        drdt += -(L2 + Real(0.5) * (L1 + L5)) / c2;
        dvdt[a] += -(L5 - L1) / (Real(2) * r * cs);
        dvdt[t1] += -L3;
        dvdt[t2] += -L4;
        dpdt += -Real(0.5) * (L1 + L5);
    }

    // The characteristic rhs REPLACES the interior one on this row, and is
    // written through the stage combination the interior cells used (comp.hpp,
    // StageCombine) -- still an rhs overwrite, never a state overwrite.
    const Real rhs[5] = {
        drdt + trho,
        vc[0] * drdt + r * dvdt[0] + tm[0],
        vc[1] * drdt + r * dvdt[1] + tm[1],
        vc[2] * drdt + r * dvdt[2] + tm[2],
        dpdt / (gam - Real(1)) +
            Real(0.5) * (vc[0] * vc[0] + vc[1] * vc[1] + vc[2] * vc[2]) * drdt +
            r * (vc[0] * dvdt[0] + vc[1] * dvdt[1] + vc[2] * dvdt[2]) + tE};
    Real* out[5] = {orho, omx, omy, omz, oE};
    const Real cr = cmb.crAt(c);
    for (int v = 0; v < 5; ++v)
        internal::stagePut<Comp>(cmb, v, c, cr, rhs[v], out[v]);
}

// Shared launch: both public apply overloads funnel here; the remap type
// selects the kernel instantiation.
// The per-row inflow targets as kernel arguments; null u = uniform targets.
struct InflowRows
{
    const Real *u = nullptr, *jet = nullptr, *rho = nullptr;
    double pulse = 0;
};

bool hydroOn(const DeviceCompNscbc::Config& c)
{
    return c.hydroLo != 0 || c.hydroHi != 0;
}

template <class Remap>
void launchNscbc(const GpuGrid& g, const DeviceCompNscbc::Config& c, double u,
                 double cInf, double Q, const DeviceComp& d,
                 const DeviceCompState& s, DeviceCompState& out,
                 const StageCombine& cmb, const Remap& R, const CurvView& cv,
                 const Real* linHi, InflowRows rows = {})
{
    const double M = u / cInf;
    const PrimView q{s.rho.data(), d.u().data(), d.v().data(), d.w().data(),
                     d.e().data(), d.p().data(), d.T().data()};
    const std::size_t nb =
        2 * (std::size_t(g.ny) * g.nz +
             (c.openY ? std::size_t(g.nx - 2) * g.nz : 0) +
             (c.openZ ? std::size_t(g.nx - 2) * (c.openY ? g.ny - 2 : g.ny)
                      : 0));
    const double betaT = c.betaTrans < 0 ? M : c.betaTrans;
    // Doublet strength tracks the ramped freestream: 2D u a^2, 3D u a^3 / 2.
    const double a2 = c.aBody * c.aBody;
    const double dbl = g.nz == 1 ? u * a2 : 0.5 * u * a2 * c.aBody;
    auto launch = [&](auto comp, auto prof, auto hydro) {
        kNscbcCells<Remap, decltype(comp)::value, decltype(prof)::value,
                    decltype(hydro)::value>
            <<<launchBlocks(nb), blockSize>>>(
            g, R, cv, viscView(d.transport()), Real(c.gamma), q, int(c.openY),
            int(c.openZ), c.floorRows,
            Real(1.0 / (g.nx * g.hx)), Real(1.0 / (g.ny * g.hy)),
            Real(1.0 / (g.nz * g.hz)), Real(c.sigma), Real(c.etaIn),
            Real(c.etaInShear < 0 ? c.etaIn : c.etaInShear),
            Real(1.0 - M * M), Real(betaT), int(c.betaAuto), Real(c.sigLat),
            Real(c.etaLat),
            Real(u), Real(c.rhoInf), Real(c.pInf), int(c.potCorrect), Real(Q),
            Real(dbl), Real(c.xc), Real(c.yc), Real(c.zc),
            Real(c.thermalInflow ? 1.0 / (c.Rgas * c.TInf) : 0.0), cmb, linHi,
            rows.u, rows.jet, rows.rho, Real(rows.pulse), Real(c.hydroLo),
            Real(c.hydroHi), out.rho.data(),
            out.mx.data(), out.my.data(), out.mz.data(), out.E.data());
    };
    internal::withCompensation(cmb, [&](auto comp) {
        // Per-row targets exist on the uniform mesh only (make() refuses
        // them otherwise), so only that instantiation is compiled; the
        // hydrostatic waves ride per-row targets only (make() again).
        if constexpr (std::is_same_v<Remap, IdentityRemap>) {
            if (rows.u) {
                if (hydroOn(c))
                    launch(comp, std::true_type{}, std::true_type{});
                else
                    launch(comp, std::true_type{}, std::false_type{});
                return;
            }
        }
        launch(comp, std::false_type{}, std::false_type{});
    });
    checkLastKernel("kNscbcCells");
}

} // namespace

std::expected<DeviceCompNscbc, hipError_t> DeviceCompNscbc::make(const Grid& g,
                                                                 Config c)
{
    if (g.nz == 1) c.openZ = false;
    const double cInf =
        c.rhoInf > 0 && c.pInf > 0 ? std::sqrt(c.gamma * c.pInf / c.rhoInf) : 0;
    const bool ok = g.nx >= 4 && c.rhoInf > 0 && c.pInf > 0 && c.gamma > 1 &&
                    c.sigma > 0 && c.etaIn > 0 &&
                    (c.etaInShear < 0 ||
                     (c.etaInShear > 0 && std::isfinite(c.etaInShear))) &&
                    c.uInf >= 0 && c.uInf < cInf &&
                    (!c.openY || (g.ny >= 4 && c.sigLat > 0 && c.etaLat > 0)) &&
                    (!c.openZ || (g.nz >= 4 && c.sigLat > 0 && c.etaLat > 0)) &&
                    (!c.potCorrect ||
                     (c.aBody >= 0 && std::isfinite(c.aBody) &&
                      std::isfinite(c.xc) && std::isfinite(c.yc) &&
                      std::isfinite(c.zc))) &&
                    c.floorRows >= -1 && c.floorRows <= g.ny - 5 &&
                    (c.floorRows < 0 || (!c.openY && !c.openZ)) &&
                    (!c.thermalInflow || (c.TInf > 0 && c.Rgas > 0)) &&
                    // The modal closure lives on the x-hi plane alone: a
                    // characteristic lateral face is not separable, and a solid
                    // floor cuts the plane the transform assumes whole. The
                    // per-cell form needs none of that and is unrestricted.
                    !(c.betaModal && c.betaAuto) &&
                    (!c.betaModal || (!c.openY && !c.openZ && c.floorRows < 0));
    if (!ok) return std::unexpected(hipErrorInvalidValue);
    const std::size_t rows = std::size_t(g.ny) * g.nz;
    const bool prof = !c.inflowU.empty();
    if (prof && (c.inflowU.size() != rows || c.inflowJet.size() != rows ||
                 c.inflowRho.size() != rows || c.potCorrect))
        return std::unexpected(hipErrorInvalidValue);
    if (hydroOn(c) &&
        (!prof || c.openY || c.openZ || !(c.hydroLo <= 0 && c.hydroHi <= 0) ||
         !std::isfinite(c.hydroLo) || !std::isfinite(c.hydroHi)))
        return std::unexpected(hipErrorInvalidValue);

    std::unique_ptr<NscbcModalAbc> abc;
    if (c.betaModal)
    {
        auto m = NscbcModalAbc::make(GpuGrid::from(g),
                                     {.gamma = c.gamma, .beta = c.betaTrans});
        if (!m) return std::unexpected(m.error());
        abc = std::make_unique<NscbcModalAbc>(std::move(*m));
    }
    DeviceCompNscbc n(g, c, std::move(abc));
    if (prof) {
        auto up = [&](const std::vector<double>& h, DeviceScalar& d) {
            auto b = DeviceScalar::make(rows);
            if (!b) return b.error();
            std::vector<Real> f(h.begin(), h.end());
            d = std::move(*b);
            return d.upload(f.data());
        };
        for (auto [h, d] : {std::pair{&c.inflowU, &n.inU_},
                            std::pair{&c.inflowJet, &n.inJ_},
                            std::pair{&c.inflowRho, &n.inR_}})
            if (auto e = up(*h, *d); e != hipSuccess) return std::unexpected(e);
    }
    return n;
}

DeviceCompNscbc::DeviceCompNscbc(const Grid& g, const Config& c,
                                 std::unique_ptr<NscbcModalAbc> abc)
    : g_(GpuGrid::from(g)), c_(c), u_(c.uInf),
      cInf_(std::sqrt(c.gamma * c.pInf / c.rhoInf)), abc_(std::move(abc))
{
}

void DeviceCompNscbc::apply(const DeviceComp& d, const DeviceCompState& s,
                            DeviceCompState& out, const StageCombine& cmb) const
{
    const Real* lin = nullptr;
    if (abc_)
    {   // Same stage primitives the kernel below reads, one pass earlier.
        abc_->compute(s.rho.data(), d.u().data(), d.v().data(), d.w().data(),
                      d.p().data());
        lin = abc_->plane();
    }
    InflowRows rows;
    if (inU_.size()) rows = {inU_.data(), inJ_.data(), inR_.data(), pulse_};
    launchNscbc(g_, c_, u_, cInf_, Q_, d, s, out, cmb, IdentityRemap{},
                CurvView{}, lin, rows);
}

namespace {

// The curvilinear DeviceComp's metric tables as a kernel view. Returns false
// when d is the Cartesian solver -- the runner pairs a curvilinear overload
// only with the DeviceComp built from the same remap's nodes, so the
// validator makes that unreachable.
bool curvViewOf(const DeviceComp& d, CurvView& cv)
{
    const DeviceCurv* t = d.curvTables();
    assert(t && "curvilinear NSCBC needs the curvilinear DeviceComp");
    if (!t) return false;
    cv = t->view();
    return true;
}

} // namespace

void DeviceCompNscbc::apply(const DeviceComp& d, const DeviceCompState& s,
                            DeviceCompState& out, const StageCombine& cmb,
                            const TanhStretchY& remap) const
{
    assert(!inU_.size() && "per-row inflow targets: uniform mesh only");
    CurvView cv;
    if (!curvViewOf(d, cv)) return;
    launchNscbc(g_, c_, u_, cInf_, Q_, d, s, out, cmb, remap, cv, nullptr);
}

void DeviceCompNscbc::apply(const DeviceComp& d, const DeviceCompState& s,
                            DeviceCompState& out, const StageCombine& cmb,
                            const AxisRemap3& remap) const
{
    assert(!inU_.size() && "per-row inflow targets: uniform mesh only");
    CurvView cv;
    if (!curvViewOf(d, cv)) return;
    launchNscbc(g_, c_, u_, cInf_, Q_, d, s, out, cmb, remap, cv, nullptr);
}

} // namespace dns::gpu
