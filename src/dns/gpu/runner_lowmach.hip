#include "dns/gpu/runner_lowmach.hpp"

#include "dns/interrupt.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/checkpoint.hpp"
#include "dns/gpu/internal/frame_scratch.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/gpu/reductions.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <numbers>
#include <optional>
#include <vector>

namespace dns::gpu {

using internal::fail;
using internal::finishTiming;

namespace {

// runLowMach's loop state: EVERY host variable the time loop carries from
// one step to the next -- the checkpoint's "loop.*" records (fields();
// gpu/internal/checkpoint.hpp; K12: a variable outside it breaks the restart
// byte rule silently, gpu_lowmach_restart is the proof).
struct LowMachLoop
{
    double t = 0.0;
    long steps = 0;
    internal::StatsCadence cad;
    internal::ProfileSchedule prof;
    double dtStab = 0.0, dtChem = 0.0, dtDiff = 0.0, dtPos = 0.0;
    int sinceDt = 1 << 30;
    bool bindChem = false;
    double xfPrev = NAN, tPrev = 0;
    double lastHits = 0, lastMin = 0, lastMax = 1;
    // --turb (M5): the swept distance X = sum uIn dt (FP64 host).
    bool turb = false;
    double sweep = 0;

    explicit LowMachLoop(const LowMachRun& r) : cad{r.statsEvery}, prof(r.profileDt), turb(r.turb != nullptr) {}

    template <class F>
    void fields(F&& f)
    {
        f("t", t);
        f("steps", steps);
        f("cad.lastLog", cad.lastLog);
        f("cad.lastPrint", cad.lastPrint);
        f("prof.next", prof.next);
        f("dtStab", dtStab);
        f("dtChem", dtChem);
        f("dtDiff", dtDiff);
        f("dtPos", dtPos);
        f("sinceDt", sinceDt);
        f("bindChem", bindChem);
        f("xfPrev", xfPrev);
        f("tPrev", tPrev);
        f("lastHits", lastHits);
        f("lastMin", lastMin);
        f("lastMax", lastMax);
        if (turb) f("sweep", sweep);
    }
};

} // namespace

std::expected<RunStats, RunError> runLowMach(const Grid& g, const LowMachState& ic,
                                             const LowMachRun& r, RunOutput& out)
{
    const CheckpointPolicy& cp = r.ckpt;
    auto devm = DeviceLowMach::make(g, r.sp);
    if (!devm) return fail(6, devm.error());
    DeviceLowMach& dev = *devm;
    auto red = ReduceScratch::make();
    if (!red) return fail(6, "GPU allocation failed");
    if (auto e = dev.upload(ic); !e) return fail(6, e.error());
    if (auto h = internal::checkDevice("the initial condition"); !h)
        return std::unexpected(h.error());
    if (cp.restart) {
        if (auto e = dev.load(*cp.restart); !e) return std::unexpected(e.error());
        if (auto h = internal::checkDevice("the restart state"); !h)
            return std::unexpected(h.error());
    }

    const double hx = g.hx, dV = g.cellVolume();
    const double plane = double(g.ny) * g.nz;
    const double Tu = r.sp.TIn;
    const double Tmid = 0.5 * (1 + r.sigma) * Tu;
    const double dTflame = (r.sigma - 1) * Tu;
    const bool open = r.sp.openY || r.sp.openZ;

    std::vector<std::string> cols = {
        "t",      "dt",     "dtChem", "dtDiff",   "SL",     "SLdisp",
        "xf",     "dth",    "Tmax",   "mass",     "massTally", "omegaEdge",
        "yHits",  "uFresh", "pFresh", "TFresh",   "yMin",   "yMax",
        "pJump",  "szDefect", "divErr", "eosErr", "pcgIt"};
    // --equivalence: the mixture fraction's range as deviations from the
    // inflow premix's Z (the CSV's 6 digits could not resolve Z itself).
    if (r.premix2) cols.insert(cols.end(), {"dzMin", "dzMax"});
    // --mode bunsen (S5): the cone's columns (runLowMach's header comment).
    if (r.bunsen) {
        cols = {"t",      "dt",      "dtChem", "dtDiff",   "xTip",   "xBase",
                "tanA",   "tanL",    "tanR",   "rms",      "sinRatio", "SLflank",
                "mFlank", "burn",    "fuelIn", "fuelNom",  "burnRatio", "Tmax",
                "mass",   "massErr", "yHits",  "yMin",     "yMax",   "divErr",
                "eosErr", "pcgIt",   "uMax"};
        if (r.premix2)
            cols.insert(cols.end(), {"xTipOut", "fuelOut", "burnPre", "burnDif",
                                     "preFrac", "zMin", "zMax"});
        if (g.nz > 1) cols.push_back("ST");
        if (open) {
            cols.insert(cols.end(),
                        {"mdotIn", "mdotOut", "mdotLat", "uLatMin", "uLatMax"});
            if (!r.premix2) cols.push_back("fuelOut");
        }
        if (r.rim) cols.insert(cols.end(), {"xRimBase", "rBase", "qRim"});
    }
    if (r.dtCell) cols.push_back("dtPos");
    // M4: the eddy fields (DeviceLowMach::eddyStats).
    const bool lesOn = r.sp.cs > 0, thickOn = r.sp.tfles.has_value();
    if (lesOn) cols.push_back("mutMax");
    if (thickOn) cols.insert(cols.end(), {"FEmax", "Ebar"});
    auto eddyCols = [&](std::vector<double>& row) {
        if (!lesOn && !thickOn) return;
        const LmEddyStats e = dev.eddyStats(*red);
        if (lesOn) row.push_back(e.mutMax);
        if (thickOn) row.insert(row.end(), {e.feMax, e.eBar});
    };
    // S8: the tube's conduction coefficient lambda_w = cp mu(T_w)/Pr (cp =
    // gamma R/(gamma - 1), R = 1 in box units).
    const double lambdaW =
        r.rim ? kGammaAir / (kGammaAir - 1) * r.sp.tr.mu(r.sp.wallT) / r.sp.tr.Pr : 0.0;
    const double zIn =
        r.premix2 ? (r.sp.zS * r.sp.yIn[0] - r.sp.yIn[1] + r.sp.zO2) / r.sp.zDen
                  : 0.0;
    // The fresh reference column: 2 delta_th past the inflow face.
    const int iRef = std::min(g.nx - 1, int(std::lround(2 * r.dth / hx)));
    if (cp.restart) {
        auto at = outputResume(*cp.restart);
        if (!at) return fail(3, "--restart: " + at.error());
        if (auto e = out.resumeAt(*at); !e) return fail(3, e.error());
    }
    if (auto e = out.declareSeries("stats", cols); !e) return fail(4, e.error());
    if (!r.bunsen) {
        const std::string pcols[] = {"t", "x", "T", "rho", "u", "p", "Y", "omega"};
        if (auto e = out.declareSeries("profiles", pcols); !e)
            return fail(4, e.error());
    } else if (r.profileDt > 0) {
        const std::string jcols[] = {"t", "x", "uc", "Tc", "Yc", "J", "P",
                                     "M", "Q", "b", "Qj"};
        if (auto e = out.declareSeries("jet", jcols); !e) return fail(4, e.error());
    }

    DeviceBuffer<double> colScratch;
    std::vector<double> ps;
    auto col = [&](int i, int q) { return ps[std::size_t(i) * kLmSums + q]; };
    struct Meas
    {
        double SL = 0, xf = NAN, dth = 0, Tmax = 0, mass = 0, omegaEdge = 0,
               yHits = 0, uRef = 0, pRef = 0, TRef = 0, pJump = 0, sz = 0;
        double yMin = HUGE_VAL, yMax = -HUGE_VAL;
        double zMin = HUGE_VAL, zMax = -HUGE_VAL;
    };
    auto measure = [&]() -> std::expected<Meas, RunError> {
        dev.updateCells();
        if (dev.columnStats(colScratch, ps) != hipSuccess)
            return fail(6, "low-Mach column stats download failed");
        Meas m;
        double wmax = 0, wedge = 0, gmax = -1;
        int kmax = -1;
        for (int i = 0; i < g.nx; ++i) {
            const double w = col(i, 4) / plane;
            m.SL += w * hx;
            m.mass += col(i, 0) * dV;
            m.yHits += col(i, 8);
            m.yMin = std::min(m.yMin, col(i, 9));
            m.yMax = std::max(m.yMax, col(i, 10));
            m.zMin = std::min(m.zMin, col(i, 11));
            m.zMax = std::max(m.zMax, col(i, 12));
            m.sz = std::max(m.sz, col(i, 5));
            m.Tmax = std::max(m.Tmax, col(i, 2) / plane);
            wmax = std::max(wmax, w);
            if (i == 0 || i == g.nx - 1) wedge = std::max(wedge, w);
            if (i + 1 < g.nx) {
                const double Ta = col(i, 2) / plane, Tb = col(i + 1, 2) / plane;
                if (std::isnan(m.xf) && Ta < Tmid && Tb >= Tmid)
                    m.xf = (i + 0.5 + (Tmid - Ta) / (Tb - Ta)) * hx;
                const double s = (Tb - Ta) / hx;
                if (s > gmax) {
                    gmax = s;
                    kmax = i;
                }
            }
        }
        double peak = gmax;
        if (kmax > 0 && kmax + 2 < g.nx) {
            auto sl = [&](int i) { return (col(i + 1, 2) - col(i, 2)) / plane / hx; };
            const double a = sl(kmax - 1), b = sl(kmax), c = sl(kmax + 1);
            const double den = a - 2 * b + c;
            if (den < 0) peak = b - (c - a) * (c - a) / (8 * den);
        }
        m.dth = peak > 0 ? dTflame / peak : 0;
        m.omegaEdge = wmax > 0 ? wedge / wmax : 0;
        const double ryRef = col(iRef, 3) / plane - r.yFBurnt * (col(iRef, 0) / plane);
        m.SL = ryRef > 0 ? m.SL / ryRef : NAN;
        m.uRef = col(iRef, 6) / plane;
        m.pRef = col(iRef, 7) / plane;
        m.TRef = col(iRef, 2) / plane;
        m.pJump = m.pRef - col(g.nx - 1, 7) / plane;
        return m;
    };

    // --- --mode bunsen (S5) -------------------------------------------------
    // Device scratch for the measurement (allocated on the first row).
    std::optional<DeviceScalar> uC, vC, wC, xiC;
    auto ensureScratch = [&]() -> bool {
        if (uC) return true;
        for (auto* b : {&uC, &vC, &wC, &xiC}) {
            auto m = DeviceScalar::make(g.size());
            if (!m) return false;
            *b = std::move(*m);
        }
        return true;
    };
    std::vector<float> Th, Rh, Wh, YFh, YOh, Uh, Vh, Wzh, Xh, MXh;
    const bool round = r.geom.round;
    const std::size_t plane2 = std::size_t(g.nx) * g.ny;
    // The cone plane of a full field: the span (z) mean (slot; FP64, k
    // ascending -- exact for an extruded field, so nz 8 == 2D bitwise) or the
    // axial plane z = zc, the mean of the two planes either side (round).
    auto conePlane = [&](const std::vector<float>& f, std::vector<float>& m) {
        m.resize(plane2);
        if (round) {
            const std::size_t k0 = plane2 * std::size_t(g.nz / 2 - 1), k1 = k0 + plane2;
            for (std::size_t q = 0; q < plane2; ++q)
                m[q] = float(0.5 * (double(f[q + k0]) + double(f[q + k1])));
            return;
        }
        for (std::size_t q = 0; q < plane2; ++q) {
            double s = 0;
            for (int k = 0; k < g.nz; ++k) s += f[q + plane2 * std::size_t(k)];
            m[q] = float(s / g.nz);
        }
    };
    // The integration field: the slot's span-mean plane (per unit span:
    // rows ny, dA = hx hy) or the round box itself (rows ny nz, dA = hx hy
    // hz).
    const std::size_t nRows = round ? std::size_t(g.ny) * g.nz : std::size_t(g.ny);
    const double dA = round ? hx * g.hy * g.hz : hx * g.hy;
    const double rowArea = round ? g.hy * g.hz : g.hy;
    auto integField = [&](const std::vector<float>& f, std::vector<float>& m) {
        if (round) {
            m = f;
            return;
        }
        conePlane(f, m);
    };
    auto integrate = [&](const std::vector<float>& f) {
        double s = 0;
        for (const float v : f) s += v;
        return s * dA;
    };
    struct MeasB
    {
        ConeMeasure cone;
        double SLflank = NAN, mFlank = NAN, burn = 0, fuelIn = 0, fuelNom = 0,
               fuelOut = 0, Tmax = 0, mass = 0, yHits = 0, uMax = 0;
        double yMin = HUGE_VAL, yMax = -HUGE_VAL;
        double xTipOut = NAN, burnPre = NAN, burnDif = NAN;
        double zMin = HUGE_VAL, zMax = -HUGE_VAL;
        double ST = NAN;
        double rBase = NAN, qRim = NAN; // --rim
    };
    std::vector<float> Tp, Rp, Wp, Up, Vp, Fp, Zp, Ip, Ip2, Wpre, Wdif;
    // The fuel the premixed flame burns per unit fresh mass (SLflank's and
    // ST's reference): C0 1, C2 Y_F1 - Y_F,rem.
    const double dYPre = r.premix2 ? r.premix.yF1() - r.premix.yFrem() : 1.0;
    auto measureB = [&]() -> std::expected<MeasB, RunError> {
        if (!ensureScratch()) return fail(6, "GPU allocation failed (bunsen stats)");
        dev.updateCells();
        if (dev.columnStats(colScratch, ps) != hipSuccess)
            return fail(6, "low-Mach column stats download failed");
        dev.cellVelocity(*uC, *vC, *wC);
        if (r.premix2)
            flameIndexCentred(dev.grid(), dev.Y(0), dev.Y(1), *xiC, r.sp.openY,
                              r.sp.openZ);
        for (auto* v : {&Th, &Rh, &Wh, &YFh, &Uh, &Vh, &Wzh}) v->resize(g.size());
        MXh.resize(dev.mx().size());
        bool ok = dev.T().download(Th.data()) == hipSuccess &&
                  dev.rho().download(Rh.data()) == hipSuccess &&
                  dev.omega().download(Wh.data()) == hipSuccess &&
                  dev.Y(0).download(YFh.data()) == hipSuccess &&
                  uC->download(Uh.data()) == hipSuccess &&
                  vC->download(Vh.data()) == hipSuccess &&
                  wC->download(Wzh.data()) == hipSuccess &&
                  dev.mx().download(MXh.data()) == hipSuccess;
        if (ok && r.premix2) {
            YOh.resize(g.size());
            Xh.resize(g.size());
            ok = dev.Y(1).download(YOh.data()) == hipSuccess &&
                 xiC->download(Xh.data()) == hipSuccess;
        }
        if (!ok) return fail(6, "field download failed");
        MeasB m;
        // Pointwise over the box.
        for (std::size_t c = 0; c < g.size(); ++c) {
            m.Tmax = std::max(m.Tmax, double(Th[c]));
            const double yf = YFh[c];
            m.yMin = std::min(m.yMin, yf);
            m.yMax = std::max(m.yMax, yf);
            m.yHits += (yf < -1e-6 || yf > 1 + 1e-6) ? 1.0 : 0.0;
            m.uMax = std::max(m.uMax, std::sqrt(double(Uh[c]) * Uh[c] +
                                                double(Vh[c]) * Vh[c] +
                                                double(Wzh[c]) * Wzh[c]));
            if (r.premix2) {
                const double yo = YOh[c];
                m.yHits += (yo < -1e-6 || yo > 1 + 1e-6) ? 1.0 : 0.0;
                const double z = r.premix.mixtureFraction(yf, yo);
                m.zMin = std::min(m.zMin, z);
                m.zMax = std::max(m.zMax, z);
            }
        }
        for (int i = 0; i < g.nx; ++i) m.mass += col(i, 0) * dV;
        // --equivalence: the premixed part of omega (w_p(xi), D14) and its
        // complement, per cell; the inner cone is measured on the premixed
        // part (the outer sheet's omega must not qualify an inner flank).
        std::vector<float>& Wcone = r.premix2 ? Wpre : Wh;
        if (r.premix2) {
            Wpre.resize(g.size());
            Wdif.resize(g.size());
            for (std::size_t c = 0; c < g.size(); ++c) {
                const double w = lmPremixedWeight(Xh[c]);
                Wpre[c] = float(double(Wh[c]) * w);
                Wdif[c] = float(double(Wh[c]) * (1.0 - w));
            }
        }
        // The cone.
        conePlane(Th, Tp);
        conePlane(Wcone, Wp);
        conePlane(Rh, Rp);
        conePlane(Uh, Up);
        conePlane(Vh, Vp);
        m.cone = measureCone(Tp, g.nx, g.ny, hx, Tmid, r.geom.yc, Wp,
                             int(std::ceil(0.5 * r.dth / hx)));
        m.mFlank = flankMassFlux(m.cone, Rp, Up, Vp, g.nx, hx);
        if (m.cone.ok && !round) {
            // The flank consumption speed per unit flank length over the fuel
            // the premixed flame burns: rho_u (Y_F1 - Y_F,rem) (C0: 1).
            double w = 0;
            for (int j = 0; j < g.ny; ++j)
                for (int i = m.cone.iLo; i <= m.cone.iHi; ++i)
                    w += Wp[std::size_t(j) * g.nx + i];
            const double cosA = 1.0 / std::hypot(1.0, m.cone.tanA);
            const double len = 2.0 * (m.cone.iHi - m.cone.iLo + 1) * hx / cosA;
            m.SLflank = w * hx * g.hy / (len * dYPre);
        }
        // Integrals (per unit span on the slot).
        integField(Wh, Ip);
        m.burn = integrate(Ip);
        integField(YFh, Fp);
        const double invPr = 1.0 / r.sp.tr.Pr;
        for (std::size_t q = 0; q < nRows; ++q) {
            const double mr = r.sp.rowM[q], yr = r.sp.rowY[0][q];
            const double kd = r.sp.tr.mu(r.sp.rowT[q]) * invPr;
            m.fuelIn += (mr * yr - kd * (double(Fp[q * g.nx]) - yr) / (0.5 * hx)) *
                        rowArea;
            m.fuelNom += mr * yr * rowArea;
        }
        if (r.premix2) {
            // The outflow face: m_nx Y_F of the last cell, per row; the slot's
            // span mean (exact for an extruded field).
            for (std::size_t j = 0; j < (round ? nRows : std::size_t(g.ny)); ++j) {
                double s = 0;
                const int nk = round ? 1 : g.nz;
                for (int k = 0; k < nk; ++k) {
                    const std::size_t row = j + std::size_t(g.ny) * k;
                    s += double(MXh[std::size_t(g.nx) + std::size_t(g.nx + 1) * row]) *
                         double(YFh[std::size_t(g.nx - 1) + std::size_t(g.nx) * row]);
                }
                m.fuelOut += s / nk * rowArea;
            }
            // Premixed / diffusion split (D14) on omega w_p(xi).
            integField(Wpre, Ip);
            integField(Wdif, Ip2);
            m.burnPre = integrate(Ip);
            m.burnDif = integrate(Ip2);
            // The outer tip: the axis' first downward Z = Z_st crossing.
            std::vector<float> zf(g.size());
            for (std::size_t c = 0; c < g.size(); ++c)
                zf[c] = float(r.premix.mixtureFraction(YFh[c], YOh[c]));
            conePlane(zf, Zp);
            const int jl = int(std::floor(r.geom.yc / hx - 0.5)), jr = jl + 1;
            auto ax = [&](int i) {
                return 0.5 * (double(Zp[std::size_t(jl) * g.nx + i]) +
                              double(Zp[std::size_t(jr) * g.nx + i]));
            };
            const double zst = r.premix.zst();
            if (ax(0) >= zst)
                for (int i = 0; i + 1 < g.nx; ++i)
                    if (ax(i) >= zst && ax(i + 1) < zst) {
                        m.xTipOut = (i + 0.5 + (ax(i) - zst) / (ax(i) - ax(i + 1))) * hx;
                        break;
                    }
        }
        if (g.nz > 1 && std::isfinite(m.cone.xTip) && std::isfinite(m.cone.xBase)) {
            const double H = m.cone.xTip - m.cone.xBase, R = 0.5 * r.geom.W;
            const double bp = r.premix2 ? m.burnPre / dYPre : m.burn;
            m.ST = round ? bp / (std::numbers::pi * R * std::hypot(H, R))
                         : bp / (2.0 * std::hypot(H, R));
        }
        if (r.rim) {
            // The base row's flank half-width (the measureCone walk on the
            // cone plane, without the burning test: the base row qualified).
            if (std::isfinite(m.cone.xBase)) {
                const int iB = int(std::floor(m.cone.xBase / hx));
                const int jl = int(std::floor(r.geom.yc / hx - 0.5)), jr = jl + 1;
                auto at = [&](int j) { return double(Tp[std::size_t(j) * g.nx + iB]); };
                double yR = NAN, yL = NAN;
                for (int j = jr; j + 1 < g.ny; ++j)
                    if (at(j) < Tmid && at(j + 1) >= Tmid) {
                        yR = (j + 0.5 + (Tmid - at(j)) / (at(j + 1) - at(j))) * hx;
                        break;
                    }
                for (int j = jl; j - 1 >= 0; --j)
                    if (at(j) < Tmid && at(j - 1) >= Tmid) {
                        yL = (j + 0.5 - (Tmid - at(j)) / (at(j - 1) - at(j))) * hx;
                        break;
                    }
                m.rBase = 0.5 * (yR - yL);
            }
            const double span = !round && g.nz > 1 ? g.nz * g.hz : 1.0;
            m.qRim = lmWallHeat(g, r.sp.solid, Th, lambdaW, r.sp.wallT, r.sp.openY,
                                r.sp.openZ) /
                     span;
        }
        return m;
    };

    // --profile-dt on the burner (S6): the jet's x profiles. Per x column i:
    // uc, Tc, Yc on the axis (the two cells either side of yc on the cone
    // plane: span mean | axial plane), J = int rho u^2, M = int rho u, Q =
    // int u over the cross-section (slot: per unit span, the span mean of
    // the cell products; round: the box's y-z section), b = the distance from
    // the axis to u = uc/2 on the cone plane (linear between cells, the mean
    // of the two sides; NaN if a side never drops to it), P = int p, Qj = Q
    // within 3 b of the axis (the jet's own volume flux).
    std::vector<float> Pj, Pq, Jq, Mq, Qq;
    auto jetProfiles = [&](double t) -> std::expected<void, RunError> {
        if (!ensureScratch()) return fail(6, "GPU allocation failed (jet profiles)");
        dev.updateCells();
        dev.cellVelocity(*uC, *vC, *wC);
        for (auto* v : {&Th, &Rh, &YFh, &Uh, &Pj}) v->resize(g.size());
        if (dev.T().download(Th.data()) != hipSuccess ||
            dev.rho().download(Rh.data()) != hipSuccess ||
            dev.Y(0).download(YFh.data()) != hipSuccess ||
            dev.p().download(Pj.data()) != hipSuccess ||
            uC->download(Uh.data()) != hipSuccess)
            return fail(6, "field download failed (jet profiles)");
        std::vector<float> ru2(g.size()), ru(g.size());
        for (std::size_t c = 0; c < g.size(); ++c) {
            ru2[c] = float(double(Rh[c]) * Uh[c] * Uh[c]);
            ru[c] = float(double(Rh[c]) * Uh[c]);
        }
        conePlane(Uh, Up);
        conePlane(Th, Tp);
        conePlane(YFh, Fp);
        integField(ru2, Jq);
        integField(Pj, Pq);
        integField(ru, Mq);
        integField(Uh, Qq);
        const double dAj = round ? g.hy * g.hz : g.hy;
        const int jl = int(std::floor(r.geom.yc / g.hy - 0.5)), jr = jl + 1;
        for (int i = 0; i < g.nx; ++i) {
            auto pl = [&](const std::vector<float>& f, int j) {
                return double(f[std::size_t(j) * g.nx + i]);
            };
            const double uc = 0.5 * (pl(Up, jl) + pl(Up, jr));
            double J = 0, P = 0, M = 0, Q = 0;
            for (std::size_t q = 0; q < nRows; ++q) {
                const std::size_t c = std::size_t(i) + std::size_t(g.nx) * q;
                J += Jq[c];
                P += Pq[c];
                M += Mq[c];
                Q += Qq[c];
            }
            // The half-width: the first crossing of uc/2 walking outward.
            auto side = [&](int j0, int dj) {
                const double half = 0.5 * uc;
                for (int j = j0; j + dj >= 0 && j + dj < g.ny; j += dj) {
                    const double a = pl(Up, j), b2 = pl(Up, j + dj);
                    if (a >= half && b2 < half) {
                        const double y = (j + 0.5 + dj * (a - half) / (a - b2)) * g.hy;
                        return std::abs(y - r.geom.yc);
                    }
                }
                return double(NAN);
            };
            const double b = 0.5 * (side(jr, 1) + side(jl, -1));
            // The jet's own volume flux: Q over the rows within 3 b of the
            // axis (a wall-issued jet's induced flow outside it runs back
            // toward the wall: the box's Q is not the jet's).
            double Qj = 0;
            for (std::size_t q = 0; std::isfinite(b) && q < nRows; ++q) {
                const double dy = (double(q % std::size_t(g.ny)) + 0.5) * g.hy -
                                  r.geom.yc;
                const double dz =
                    round ? (double(q / std::size_t(g.ny)) + 0.5) * g.hz - r.geom.zc
                          : 0.0;
                if (std::hypot(dy, dz) <= 3 * b)
                    Qj += Qq[std::size_t(i) + std::size_t(g.nx) * q];
            }
            const double row[] = {t,
                                  (i + 0.5) * hx,
                                  uc,
                                  0.5 * (pl(Tp, jl) + pl(Tp, jr)),
                                  0.5 * (pl(Fp, jl) + pl(Fp, jr)),
                                  J * dAj,
                                  P * dAj,
                                  M * dAj,
                                  Q * dAj,
                                  b,
                                  std::isfinite(b) ? Qj * dAj : NAN};
            out.row("jet", row);
        }
        return {};
    };

    const float tempScale = float(dTflame);
    const float omegaScale = float(1.0 / r.dth);
    internal::FrameScratch fs;
    auto writeFrameB = [&](double t) {
        const int nc = r.premix2 ? 6 : 4;
        if (!fs.ensure(g.size(), nc, true)) return;
        dev.updateCells();
        dev.cellVelocity(fs.u(), fs.v(), fs.w());
        speedCentred(dev.grid(), fs.u(), fs.v(), fs.w(), fs.speed());
        const float sScale = float(r.sigma * r.geom.uJet);
        if (!r.premix2) {
            fs.take(0, dev.T(), Tu);
            fs.take(1, dev.Y(0));
            fs.take(2, dev.omega());
            fs.take(3, fs.speed());
            const FrameField ff[4]{{"temp", fs.channel(0), 0, tempScale, false},
                                   {"fuel", fs.channel(1), 0, 1.0f, false},
                                   {"omega", fs.channel(2), 0, omegaScale, false},
                                   {"speed", fs.channel(3), 0, sScale, false}};
            fs.emit(out, t, ff);
            return;
        }
        flameIndexCentred(dev.grid(), dev.Y(0), dev.Y(1), fs.vort(), r.sp.openY,
                          r.sp.openZ);
        fs.take(0, dev.T(), Tu);
        fs.take(1, dev.Y(0));
        fs.take(2, dev.Y(1));
        fs.take(3, dev.omega());
        fs.take(4, fs.vort());
        fs.take(5, fs.speed());
        const FrameField ff[6]{{"temp", fs.channel(0), 0, tempScale, false},
                               {"fuel", fs.channel(1), 0, 1.0f, false},
                               {"oxid", fs.channel(2), 0, 1.0f, false},
                               {"omega", fs.channel(3), 0, omegaScale, false},
                               {"findex", fs.channel(4), 0, 1.0f, true},
                               {"speed", fs.channel(5), 0, sScale, false}};
        fs.emit(out, t, ff);
    };
    auto writeFrame = [&](double t) {
        if (r.bunsen) {
            writeFrameB(t);
            return;
        }
        if (!fs.ensure(g.size(), 3, false)) return;
        dev.updateCells();
        fs.take(0, dev.T(), Tu);
        fs.take(1, dev.Y(0));
        fs.take(2, dev.omega());
        const FrameField ff[3]{{"temp", fs.channel(0), 0, tempScale, false},
                               {"fuel", fs.channel(1), 0, 1.0f, false},
                               {"omega", fs.channel(2), 0, omegaScale, false}};
        fs.emit(out, t, ff);
    };
    auto emitProfiles = [&](double t) {
        for (int i = 0; i < g.nx; ++i) {
            auto cl = [&](int q) { return col(i, q); };
            const double row[] = {t,
                                  (i + 0.5) * hx,
                                  cl(2) / plane,
                                  cl(0) / plane,
                                  cl(6) / plane,
                                  cl(7) / plane,
                                  cl(3) / cl(0),
                                  cl(4) / plane};
            out.row("profiles", row);
        }
    };

    RunStats st;
    LowMachLoop L(r);
    std::vector<double> turbM, turbV, turbW;
    long steps0 = 0;
    if (cp.restart) {
        if (auto e = internal::loadLoop(*cp.restart, L); !e)
            return fail(3, "--restart: " + e.error());
        steps0 = L.steps;
    }
    const double guard = 2 * r.dth;
    const double Lx = g.nx * hx;
    const auto wall0 = std::chrono::steady_clock::now();

    using Clock = std::chrono::steady_clock;
    long ckK = 0;
    double tCk = cp.restart ? L.t : -1.0;
    auto ckAdvance = [&] {
        ckK = long(std::floor(L.t / cp.every)) + 1;
        while (double(ckK) * cp.every <= L.t) ++ckK;
    };
    if (cp.every > 0) ckAdvance();
    const Clock::time_point jobStart =
        cp.wallStart == Clock::time_point{} ? wall0 : cp.wallStart;
    Clock::time_point lastCk = jobStart;
    auto seconds = [](Clock::duration d) {
        return std::chrono::duration<double>(d).count();
    };
    auto writeCheckpoint = [&]() -> std::expected<void, RunError> {
        Checkpoint ck;
        ck.putText("binary", cp.binary);
        ck.putText("solver", kLowMachSolverName);
        ck.putText("case", cp.caseText);
        const std::int64_t grid[4] = {g.nx, g.ny, g.nz, r.sp.nY};
        ck.putI64("grid", grid);
        ck.putF64("t", L.t);
        ck.putI64("steps", L.steps);
        ck.putI64("restarts", cp.restarts);
        internal::saveLoop(ck, L);
        auto at = out.resumePoint();
        if (!at) return fail(4, "checkpoint: " + at.error());
        putOutputResume(ck, *at);
        if (auto e = dev.save(ck); !e) return std::unexpected(e.error());
        if (auto e = ck.write(checkpointPath(cp.prefix)); !e)
            return fail(4, "checkpoint: " + e.error());
        tCk = L.t;
        return {};
    };
    bool stopped = false;

    while (L.t < r.tEnd) {
        if (stopRequested()) {
            if (cp.every > 0 && L.t > tCk) {
                if (auto e = writeCheckpoint(); !e) return std::unexpected(e.error());
                std::printf("checkpoint: t=%.9g steps=%ld (%s; resume with "
                            "--restart %s)\n",
                            L.t, L.steps, stopReason(), cp.prefix.c_str());
                stopped = true;
            }
            break;
        }
        if (auto e = internal::recomputeDt(
                L.sinceDt, L.dtStab, r.dtRecompute, r.dtMargin, 1e-9,
                [&] {
                    const double a = dev.convectiveDt(*red, r.cfl);
                    L.dtDiff = dev.diffusiveDt(*red);
                    L.dtChem = dev.chemDt(*red, r.chemK);
                    double s = std::min(a, L.dtDiff);
                    if (r.dtCell) {
                        L.dtPos = dev.positivityDt(*red, r.posK);
                        s = std::min(s, L.dtPos);
                    }
                    L.bindChem = L.dtChem < s;
                    return std::min(s, L.dtChem);
                });
            !e)
            return fail(7, e.error());
        // The last two steps share the remainder (no step below dtStab/2):
        // the projection reads the FP32 rounding of m* over cr, so a
        // sliver step would print a noisy pressure row at tEnd.
        const double rem = r.tEnd - L.t;
        const double dt = rem <= L.dtStab         ? rem
                          : rem < 2 * L.dtStab    ? 0.5 * rem
                                                  : L.dtStab;
        // --turb: the box column the inflow face sees at the step's start, held over
        // its stages; the box then sweeps past at u_in.
        if (r.turb) {
            lowmachTurbRows(*r.turb, L.sweep, r.sp.mIn, turbM, turbV, turbW);
            if (auto e = dev.setInflowTurb(turbM, turbV, turbW); !e) return fail(6, e.error());
        }
        if (auto e = dev.step(dt); !e) return fail(dev.pcgFailed() ? 8 : 6, e.error());
        L.sweep += r.uIn * dt;
        L.t += dt;
        ++L.steps;
        const double t = L.t;

        const bool statsDue = L.cad.due(L.steps, t, r.tEnd);
        const bool profDue = !r.bunsen && L.prof.due(t, r.tEnd);
        const bool jetDue = r.bunsen && L.prof.due(t, r.tEnd);
        if (r.bunsen && statsDue) {
            auto m = measureB();
            if (!m) return std::unexpected(m.error());
            const double divE = dev.divErr(*red), eosE = dev.eosErr(*red);
            const double tally = dev.massTally();
            if (auto h = internal::checkDevice("a stats row", &*red); !h)
                return std::unexpected(h.error());
            if (!std::isfinite(m->mass) || !std::isfinite(m->Tmax))
                return fail(7, "flame state is not finite (diverged)");
            L.lastHits = m->yHits;
            L.lastMin = m->yMin;
            L.lastMax = m->yMax;
            const ConeMeasure& c = m->cone;
            const double sinA = c.ok ? c.tanA / std::hypot(1.0, c.tanA) : NAN;
            // Open laterals: the boundary sums (per unit span on the slot);
            // fuelOut = x-hi + the laterals, ambient-aware.
            LmBoundarySums bs;
            if (open) {
                bs = dev.boundarySums();
                const double span = !round && g.nz > 1 ? g.nz * g.hz : 1.0;
                bs.mIn /= span;
                bs.mOut /= span;
                bs.mLat /= span;
                m->fuelOut = bs.yOut[0] / span;
            }
            const double net =
                r.premix2 || open ? m->fuelIn - m->fuelOut : m->fuelIn;
            std::vector<double> row = {
                t,         dt,          L.dtChem,   L.dtDiff,
                c.xTip,    c.xBase,     c.ok ? c.tanA : NAN,
                c.ok ? c.tanL : NAN,    c.ok ? c.tanR : NAN,
                c.ok ? c.rms : NAN,     sinA * r.geom.uJet,
                m->SLflank, m->mFlank,  m->burn,    m->fuelIn,
                m->fuelNom, m->burn / net, m->Tmax, m->mass,
                (m->mass - tally) / tally, m->yHits, m->yMin, m->yMax,
                divE,      eosE,        double(dev.pcgIterations()), m->uMax};
            if (r.premix2)
                row.insert(row.end(), {m->xTipOut, m->fuelOut, m->burnPre,
                                       m->burnDif, m->burnPre / m->burn,
                                       m->zMin, m->zMax});
            if (g.nz > 1) row.push_back(m->ST);
            if (open) {
                row.insert(row.end(),
                           {bs.mIn, bs.mOut, bs.mLat, bs.uLatMin, bs.uLatMax});
                if (!r.premix2) row.push_back(m->fuelOut);
            }
            const double xRimBase = c.xBase - r.geom.xRim;
            if (r.rim) row.insert(row.end(), {xRimBase, m->rBase, m->qRim});
            if (r.dtCell) row.push_back(L.dtPos);
            eddyCols(row);
            out.row("stats", row);
            if (L.cad.printDue(L.steps, t, r.tEnd)) {
                std::printf("t=%.3f/%.3f tip=%.3f base=%.3f alpha=%.3f deg "
                            "sin(a)U/SL=%.5f SLflank=%.5f mFlank=%.5f "
                            "burn/in=%.4f Tmax/Tb=%.5f |u|max=%.3f eos=%.2e "
                            "dt=%.2e%s",
                            t, r.tEnd, c.xTip, c.xBase,
                            std::atan(c.tanA) * 180 / std::numbers::pi,
                            sinA * r.geom.uJet, m->SLflank, m->mFlank,
                            m->burn / net, m->Tmax / (r.sigma * Tu), m->uMax,
                            eosE, dt, L.bindChem ? " (chem dt)" : "");
                if (r.premix2)
                    std::printf(" tipOut=%.3f preFrac=%.4f Z=[%+.1e,1%+.1e]",
                                m->xTipOut, m->burnPre / m->burn, m->zMin,
                                m->zMax - 1);
                if (r.rim)
                    std::printf(" rimBase=%.3f rBase=%.3f qRim=%.4g pcgIt=%d",
                                xRimBase, m->rBase, m->qRim, dev.pcgIterations());
                std::printf("\n");
                std::fflush(stdout);
                L.cad.printed(L.steps);
            }
            L.cad.logged(L.steps);
            std::string why;
            // --rim: heights above the rim (0 without one).
            const double x0 = r.rim ? r.geom.xRim : 0.0;
            if (r.inert)
                ; // no cone to anchor (--reaction off)
            else if (c.axisHot)
                why = "the slot axis is hot at the inflow (flash-back)";
            else if (r.rim && std::isfinite(c.xBase) && c.xBase < x0)
                why = std::format("the base x={:.4g} sits below the rim x={:.4g} "
                                  "(flash-back into the tube)",
                                  c.xBase, x0);
            else if (!std::isfinite(c.xTip))
                why = "the slot axis never reaches Tmid (blow-through)";
            else if (c.xTip > Lx - guard)
                why = std::format("the tip x={:.4g} reached the outflow "
                                  "(blow-through)",
                                  c.xTip);
            else if (!std::isfinite(c.xBase) || c.xBase - x0 > 0.5 * (c.xTip - x0))
                why = std::format("the base x={:.4g} lifted past half the tip "
                                  "height {:.4g}{} (blow-off)",
                                  c.xBase, c.xTip, r.rim ? " above the rim" : "");
            else if (!c.ok && g.nz == 1)
                why = std::format("fewer than 4 burning flank rows between base "
                                  "{:.4g} and tip {:.4g} (the cone opened: "
                                  "blow-off)",
                                  c.xBase, c.xTip);
            if (!why.empty())
                return fail(8, std::format("flame not anchored at t={:.4g}: {} "
                                           "(adjust --jet-speed, {})",
                                           t, why,
                                           r.rim ? "--rim, --rim-temp, --coflow-speed"
                                                 : "--pilot-*, --coflow-temp"));
        } else if (statsDue || profDue) {
            auto m = measure();
            if (!m) return std::unexpected(m.error());
            const double divE = dev.divErr(*red), eosE = dev.eosErr(*red);
            const double tally = dev.massTally();
            if (auto h = internal::checkDevice("a stats row", &*red); !h)
                return std::unexpected(h.error());
            if (!std::isfinite(m->mass) || !std::isfinite(m->Tmax) ||
                !std::isfinite(m->xf))
                return fail(7, "flame state is not finite (diverged)");
            L.lastHits = m->yHits;
            L.lastMin = m->yMin;
            L.lastMax = m->yMax;
            if (statsDue) {
                const double dxf = std::isfinite(L.xfPrev) && t > L.tPrev
                                       ? (m->xf - L.xfPrev) / (t - L.tPrev)
                                       : 0.0;
                std::vector<double> row = {
                    t,         dt,          L.dtChem,  L.dtDiff, m->SL,
                    m->uRef - dxf, m->xf,   m->dth,    m->Tmax,  m->mass,
                    tally,     m->omegaEdge, m->yHits, m->uRef,  m->pRef,
                    m->TRef,   m->yMin,     m->yMax,   m->pJump, m->sz,
                    divE,      eosE,        0.0};
                if (r.premix2)
                    row.insert(row.end(), {m->zMin - zIn, m->zMax - zIn});
                if (r.dtCell) row.push_back(L.dtPos);
                eddyCols(row);
                out.row("stats", row);

                L.xfPrev = m->xf;
                L.tPrev = t;
                if (L.cad.printDue(L.steps, t, r.tEnd)) {
                    std::printf("t=%.3f/%.3f SL=%.6f xf=%.4f dth=%.4f "
                                "Tmax/Tb=%.5f pJump=%.6f mass-tally=%.2e "
                                "eos=%.2e yHits=%g dt=%.2e%s",
                                t, r.tEnd, m->SL, m->xf, m->dth,
                                m->Tmax / (r.sigma * Tu), m->pJump,
                                (m->mass - tally) / tally, eosE, m->yHits, dt,
                                L.bindChem ? " (chem dt)" : "");
                    if (r.premix2)
                        std::printf(" Z-Zin=%+.2e..%+.2e", m->zMin - zIn,
                                    m->zMax - zIn);
                    std::printf("\n");
                    std::fflush(stdout);
                    L.cad.printed(L.steps);
                }
                L.cad.logged(L.steps);
            }
            if (profDue) {
                emitProfiles(t);
                L.prof.done(t);
            }
            if (m->xf < guard || m->xf > Lx - guard)
                return fail(8, std::format("flame not stabilized: drifted to "
                                           "the {} face at t={:.4g} (x_f={:.4g}; "
                                           "adjust --inflow-speed)",
                                           m->xf < guard ? "inflow" : "outflow",
                                           t, m->xf));
        }
        if (jetDue) {
            if (auto e = jetProfiles(t); !e) return std::unexpected(e.error());
            L.prof.done(t);
        }
        if (out.wantsAnyFrame(t)) writeFrame(t);
        if (cp.every > 0 && t < r.tEnd && t >= double(ckK) * cp.every) {
            if (auto e = writeCheckpoint(); !e) return std::unexpected(e.error());
            ckAdvance();
            const Clock::time_point now = Clock::now();
            const double elapsed = seconds(now - jobStart);
            const double interval = seconds(now - lastCk);
            lastCk = now;
            if (cp.wallLimit >= 0 &&
                (elapsed >= cp.wallLimit || elapsed + interval > cp.wallLimit)) {
                std::printf("checkpoint: t=%.9g steps=%ld (incomplete; resume "
                            "with --restart %s)\n",
                            t, L.steps, cp.prefix.c_str());
                stopped = true;
                break;
            }
            std::printf("checkpoint: t=%.9g steps=%ld -> %s\n", t, L.steps,
                        checkpointPath(cp.prefix).c_str());
        }
    }
    if (auto h = internal::checkDevice("the end of the run", &*red); !h)
        return std::unexpected(h.error());
    if (cp.active()) out.setCompletion(!stopped && L.t >= r.tEnd, cp.restarts);
    if (L.lastHits > 0)
        std::fprintf(stderr,
                     "dns: warning: %g cells with Y outside [-1e-6, 1+1e-6] "
                     "at the last stats row (counted, never clamped): Y in "
                     "[%.3g, 1%+.3g]\n",
                     L.lastHits, L.lastMin, L.lastMax - 1);
    std::printf("label: %s\n", r.label.c_str());
    st.steps = L.steps;
    finishTiming(st, wall0, out, steps0);
    st.tStop = L.t;
    return st;
}

} // namespace dns::gpu
