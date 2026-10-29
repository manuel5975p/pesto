// The scratch every body-flow runner needs to turn device fields into the
// float planes RunOutput takes, and the one call that hands them over.
//
// WHAT IS HERE. The buffers (cell-centred velocity, the two derived scalars,
// the host staging and the float channels), the download-and-cast loop, and
// the frame() call with its error report. All six runners spelled those
// identically.
//
// WHAT IS NOT. Which derived kernel runs, what the colour scales are, and
// what the channels are called -- those differ per physics and stay at the
// call site, exactly like the loop bodies in run_loop.hpp.
//
// Allocation happens on the FIRST frame. A stats-only run -- every validation
// case, every steady solve -- must not pay for six full-grid arrays it never
// fills; runner_incomp already did this and the other five did not.
//
// GPU-internal header: HIP TUs only.
#pragma once

#include "dns/output.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/derived.hpp"
#include "dns/gpu/device_buffer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optional>
#include <span>
#include <vector>

namespace dns::gpu::internal {

// Full-scale clip of a --temp channel, which renders T - Tinf on the diverging
// map exactly as --speed-map delta does for velocity. The clip is whichever
// thermal excursion the case can actually produce: an isothermal wall's own
// offset, or the stagnation rise (gamma-1)/2 M^2 Tinf that an adiabatic body
// still generates. Shared by the two compressible body runners, which must
// agree -- their frames are compared against each other.
inline float thermalFrameScale(double TInf, double machSq, double gamma,
                               bool isothermalWall, double Twall)
{
    return float(std::max(isothermalWall ? std::abs(Twall - TInf) : 0.0,
                          0.5 * (gamma - 1.0) * machSq * TInf));
}

// The speed channel of a body run: |u| on the sequential [0, speedMax] map, or
// (--speed-map delta) dspeed = |u| - uInf on the diverging +/- 0.6 uInf map.
inline FrameField speedField(const float* data, bool delta, double uInf,
                             float speedMax)
{
    return delta ? FrameField{"dspeed", data, 0, float(0.6 * uInf), true}
                 : FrameField{"speed", data, 0, speedMax, false};
}

class FrameScratch
{
public:
    // Allocates on the first call and is a no-op after. `channels` is how many
    // float planes the caller will fill; `velocity` adds the three cell-centred
    // velocity components a compressible runner derives its fields from (the
    // incompressible one works straight off the MAC field and passes false).
    // Returns false, having reported once, when the device is out of memory --
    // the caller must then skip the frame, not write a partial one.
    // Pre: n > 0, 1 <= channels <= kMaxChannels.
    bool ensure(std::size_t n, int channels, bool velocity)
    {
        if (ready_) return true;
        if (failed_) return false;
        auto grab = [&](std::optional<DeviceScalar>& d) {
            auto a = DeviceScalar::make(n);
            if (!a) return false;
            d = std::move(*a);
            return true;
        };
        const bool ok = grab(vort_) && grab(speed_)
                      && (!velocity || (grab(u_) && grab(v_) && grab(w_)));
        if (!ok)
        {
            std::fprintf(stderr, "frame scratch allocation failed\n");
            failed_ = true;
            return false;
        }
        host_.resize(n);
        for (int i = 0; i < channels; ++i) chan_[i].resize(n);
        n_ = n;
        ready_ = true;
        return true;
    }

    DeviceScalar& u() { return *u_; }
    DeviceScalar& v() { return *v_; }
    DeviceScalar& w() { return *w_; }

    // u/v/w <- the cell-centred velocity m/rho of a conserved state, the
    // first step of every compressible runner's frame.
    // Pre: ensure(..., velocity = true) succeeded.
    void centreFromConserved(const GpuGrid& gg, const DeviceCompState& s)
    {
        primitiveVelocity(gg, s.rho, s.mx, s.my, s.mz, *u_, *v_, *w_);
    }
    DeviceScalar& vort() { return *vort_; }
    DeviceScalar& speed() { return *speed_; }

    // Downloads `src` into channel `ch` as float(src) - off. `off` is the
    // freestream a delta map subtracts, or the reference a diverging
    // temperature map is centred on; 0 leaves the field as it is.
    //
    // The subtraction is done in DOUBLE and rounded once, which is what every
    // caller had except the two delta-speed maps -- those rounded `off` to
    // float first. Identical today because the freestream is pinned to
    // uInf = 1 (config.cpp), exactly representable; if that ever stops being
    // true this rounds better, not the same.
    //
    // A failed download leaves the channel as it was and returns false; the
    // frame is then dropped by emit() rather than written half stale.
    bool take(int ch, const DeviceScalar& src, double off = 0.0)
    {
        if (src.download(host_.data()) != hipSuccess)
        {
            stale_ = true;
            return false;
        }
        auto& out = chan_[ch];
        for (std::size_t c = 0; c < n_; ++c)
            out[c] = float(double(host_[c]) - off);
        return true;
    }

    float* channel(int ch) { return chan_[ch].data(); }

    // Hands the frame to the sink, unless a take() since the last emit failed:
    // that frame is skipped and reported. A failed write is REPORTED (and
    // recorded by the sink for its finish()) and the run continues: a full
    // disk must not lose a run that is otherwise healthy.
    void emit(RunOutput& out, double t, std::span<const FrameField> ff)
    {
        if (stale_)
        {
            stale_ = false;
            std::fprintf(stderr, "frame at t=%.6g skipped: device download "
                                 "failed\n", t);
            return;
        }
        if (auto e = out.frame(t, ff); !e)
            std::fprintf(stderr, "frame write failed: %s\n", e.error().c_str());
    }

    // 5: dns flame at nz > 1 (temp, fuel, omega, vort, speed).
    static constexpr int kMaxChannels = 5;

private:
    std::optional<DeviceScalar> u_, v_, w_, vort_, speed_;
    std::vector<Real> host_;
    std::vector<float> chan_[kMaxChannels];
    std::size_t n_ = 0;
    bool ready_ = false, failed_ = false, stale_ = false;
};

} // namespace dns::gpu::internal
