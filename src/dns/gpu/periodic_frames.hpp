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

#include <expected>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

namespace dns::gpu {

// Centre-plane geometry for a periodic box; the caller hands it to
// RunOutput::make.
inline FrameGeometry periodicFrameGeometry(const Grid& g)
{
    FrameGeometry fg;
    fg.ic = g.nx / 2;
    fg.jc = g.ny / 2;
    fg.kc = g.nz / 2;
    fg.D = 2 * std::numbers::pi;
    return fg;
}

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

    void write(const GpuGrid& gg, const DeviceField& U, double t,
               std::size_t n)
    {
        vorticityZ(gg, U, *vort_);
        speedMac(gg, U, *speed_);
        (void)vort_->download(hv_.data());
        (void)speed_->download(hs_.data());
        for (std::size_t c = 0; c < n; ++c)
        {
            fv_[c] = float(hv_[c]);
            fs_[c] = float(hs_[c]);
        }
        const FrameField ff[2]{{"vort", fv_.data(), 0, 8, true},
                               {"speed", fs_.data(), 0, 2, false}};
        (void)out_->frame(t, FrameViews{ff});
    }

private:
    RunOutput* out_ = nullptr;
    std::optional<DeviceScalar> vort_, speed_;
    std::vector<Real> hv_, hs_;
    std::vector<float> fv_, fs_;
};

} // namespace dns::gpu
