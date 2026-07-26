// The single sink for every run artefact: the run .json, the stats CSV, the
// body mask, per-frame .bin planes or volumes, PNG frames, live web frames and
// legacy VTK. Unselected kinds allocate nothing and open no file.
//
// The run .json is written at construction with "frames":0 and REWRITTEN by the
// destructor with the final frame count and ms/step, so a viewer polling a live
// run always sees a well-formed manifest.
#pragma once

#include "dns/runmeta.hpp"
#include "dns/field.hpp"
#include "dns/grid.hpp"

#include <expected>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace dns {

// One derived field of a frame, sampled on the FULL grid in Grid::idx order.
// lo/hi are the fixed colour/quantization range; `diverging` picks the PNG and
// web colour map (diverging maps clip to +/-hi and ignore lo).
struct FrameField
{
    std::string name;
    const float* data = nullptr;
    float lo = 0, hi = 1;
    bool diverging = false;
    // > 0 (diverging only): the PNG renders asinh(v/toe)/asinh(hi/toe) --
    // linear below the toe, logarithmic above, so a high-Re field's decades
    // of vorticity all read. bin/web stay RAW; this is a display transform.
    float asinhToe = 0;
};

// What the caller hands to RunOutput::frame(): the fields of one time level.
struct FrameViews
{
    std::span<const FrameField> fields;
};

// Where the mid-plane / slice / web planes are cut, and the body mask used to
// grey out solid cells. mask is a full-grid fluid fraction (1 fluid, 0 solid)
// or empty for a body-free run.
struct FrameGeometry
{
    std::vector<float> mask;
    int ic = 0, jc = 0, kc = 0;   // centre indices of the three axes
    double D = 1.0;               // body diameter, for the web plane offsets

    // Written window, in full-grid indices: images and volumes cover
    // [i0, i0+w) x [j0, j0+h) x [k0, k0+d). Defaults (w == 0) mean the whole
    // grid; an incompressible run passes the interior so the ghost ring never
    // reaches a frame. ic/jc/kc are full-grid indices regardless.
    int i0 = 0, j0 = 0, k0 = 0;
    int w = 0, h = 0, d = 0;
};

class RunOutput
{
public:
    struct Config
    {
        std::string prefix;
        std::set<Emit> kinds;
        double fdt = 0.15, pdt = 0, wdt = 0;
        int planesPerAxis = 3;
        int pngDown = 1;       // box-average coarsening for PNG frames; 1 = full res
        std::string simJson;   // extra "sim" metadata for the web manifest
    };

    // Creates and truncates every file and directory the run will ever write
    // and fails here if any cannot be created -- before any GPU allocation.
    static std::expected<RunOutput, std::string>
    make(Config, RunMeta, const Grid&, FrameGeometry);

    RunOutput(RunOutput&&) noexcept;
    RunOutput& operator=(RunOutput&&) noexcept;
    ~RunOutput();                            // rewrites the .json, flushes

    // Declares a CSV series. `name` is the file suffix ("stats" ->
    // <prefix>_stats.csv); cols is the header line, written verbatim.
    void declareSeries(std::string_view name, std::span<const std::string> cols);
    [[nodiscard]] std::expected<void, std::string>
    row(std::string_view series, std::span<const double>);

    // True when t has reached the next .bin / png / web cadence.
    bool wantsFrame(double t) const;
    bool wantsPng(double t) const;
    bool wantsWeb(double t) const;
    // The live shared-memory bridge (--emit shm). Same cadence as the web
    // frames; separate predicate because a run can want one without the other.
    bool wantsShm(double t) const;

    // Writes whichever kinds are enabled and due at time t. Advances the
    // cadences it consumed.
    [[nodiscard]] std::expected<void, std::string> frame(double t, const FrameViews&);

    // Cell-centred velocity snapshot (Emit::Vtk only).
    [[nodiscard]] std::expected<void, std::string>
    vtkFrame(const Grid&, const VectorField&);

    // Indexed auxiliary artefacts: <prefix>_<kind>_<n><ext>. `hit` writes its
    // spectra and raw snapshots this way. They go through the sink rather than
    // straight to std::ofstream so a failed write is REPORTED -- the whole
    // point of phase 3c is that a full or read-only destination surfaces
    // immediately, and the old direct writes bypassed that and failed
    // silently, hours into a run.
    //
    // The count is unbounded (one per cadence hit), so these cannot all be
    // created up front; what phase 3c does establish is that the directory is
    // writable at all.
    [[nodiscard]] std::expected<void, std::string>
    auxText(std::string_view kind, int index, std::string_view ext,
            std::string_view content);
    [[nodiscard]] std::expected<void, std::string>
    auxBinary(std::string_view kind, int index, std::string_view ext,
              std::span<const float> data);

    [[nodiscard]] std::expected<void, std::string> flush();

    int frames() const;
    // Recorded in the final .json. Call before destruction.
    void setMsPerStep(double);

private:
    RunOutput();
    struct Impl;
    std::unique_ptr<Impl> p_;
};

// Downloads nothing: argmaxes each component of an already-host-side field and
// prints the offending cells. Called on a non-finite state before exiting 7.
void reportBlowup(const Grid&, const VectorField& host, const Scalar& lambda,
                  double t);

} // namespace dns
