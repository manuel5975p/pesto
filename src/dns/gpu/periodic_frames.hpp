// Mid-plane speed / |curl| frames for the PERIODIC subcommands (tgv, hit) --
// the former `movie` app. The body-flow runners keep their own frame writers:
// they carry a mask, a display raster and per-physics colour scales, and this
// one has none of that.
//
// NOTE: `movie` also wrote a pressure channel, obtained from a separate
// lap(p) = div(conv U) solve. That channel is NOT carried over.
//
// GPU-only header (it owns device scratch).
#pragma once

#include "dns/config.hpp"
#include "dns/output.hpp"
#include "dns/gpu/derived.hpp"
#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/index.hpp"
#include "dns/gpu/kernels.hpp"

#include <cstdio>
#include <expected>
#include <optional>
#include <string>
#include <vector>

namespace dns::gpu {

class PeriodicFrames
{
public:
    // Borrows the caller's sink. One run, one artefact sink: `hit` also
    // writes stats and aux artefacts, and two RunOutputs on one prefix would
    // both rewrite <prefix>.json from their destructors.
    static std::expected<PeriodicFrames, std::string>
    make(const Grid& g, RunOutput& out)
    {
        PeriodicFrames p;
        p.out_ = &out;
        auto a = DeviceScalar::make(g.size());
        auto b = DeviceScalar::make(g.size());
        if (!a || !b) return std::unexpected("GPU allocation failed");
        p.vort_ = std::move(*a);
        p.speed_ = std::move(*b);
        p.hv_.resize(g.size());
        p.hs_.resize(g.size());
        p.fv_.resize(g.size());
        p.fs_.resize(g.size());
        return p;
    }

    bool wants(double t) const { return out_->wantsFrame(t); }

    // Derives and writes one frame. A failed download skips the frame and a
    // failed write is recorded by the sink; both are reported and the run
    // continues (a frame is never worth a healthy run).
    void write(const GpuGrid& gg, const DeviceField& U, double t)
    {
        vorticityZ(gg, U, *vort_);
        speedMac(gg, U, *speed_);
        if (vort_->download(hv_.data()) != hipSuccess
            || speed_->download(hs_.data()) != hipSuccess)
        {
            std::fprintf(stderr, "frame at t=%.6g skipped: device download "
                                 "failed\n", t);
            return;
        }
        for (std::size_t c = 0; c < hv_.size(); ++c)
        {
            fv_[c] = float(hv_[c]);
            fs_[c] = float(hs_[c]);
        }
        const FrameField ff[2]{{"vort", fv_.data(), 0, 8, true},
                               {"speed", fs_.data(), 0, 2, false}};
        if (auto e = out_->frame(t, ff); !e)
            std::fprintf(stderr, "frame write failed: %s\n", e.error().c_str());
    }

private:
    RunOutput* out_ = nullptr;
    std::optional<DeviceScalar> vort_, speed_;
    std::vector<Real> hv_, hs_;
    std::vector<float> fv_, fs_;
};

} // namespace dns::gpu
