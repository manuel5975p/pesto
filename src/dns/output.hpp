// The single sink for every run artefact: the run .json, the stats CSV, the
// body mask, per-frame .bin planes or volumes, PNG frames, live web frames and
// legacy VTK. Unselected kinds allocate nothing and open no file.
//
// The run .json is written at construction with "frames":0 and REWRITTEN by
// finish() (or, failing that, the destructor) with the final frame count and
// ms/step, so a viewer polling a live run always sees a well-formed manifest.
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

    // Cut on the three centre planes of g (n/2 each; nz/2 is already 0 at
    // nz == 1) with web offsets quoted in units of D; no mask, whole grid.
    static FrameGeometry centred(const Grid& g, double D)
    {
        FrameGeometry fg;
        fg.ic = g.nx / 2;
        fg.jc = g.ny / 2;
        fg.kc = g.nz / 2;
        fg.D = D;
        return fg;
    }
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
    // <prefix>_stats.csv); cols is the header line, written verbatim. Fails
    // when the file cannot be created (the caller exits 4); a no-op success
    // when stats are not emitted.
    [[nodiscard]] std::expected<void, std::string>
    declareSeries(std::string_view name, std::span<const std::string> cols);
    // Appends one row; false when the write failed. The failure is RECORDED
    // and finish() reports it, so a caller in a time loop may ignore this.
    bool row(std::string_view series, std::span<const double>);

    // Replace the body mask, for a run whose body MOVES.
    //
    // Reaches the sinks that read the mask per frame -- the PNG frames and the
    // live shared-memory planes -- so a rotor's images and its cockpit view
    // follow the blade. It does NOT reach the two sinks that write the mask
    // ONCE, by their own contract: `<prefix>_mask.bin` is a single plane
    // written in phase 3c, and the web format carries the mask in its `static`
    // section (docs/viewer-format.md). Both would need a format change to
    // become time-dependent, so for a moving body they hold the mask the run
    // STARTED with. Pre: same size as the mask the run was made with.
    void updateMask(std::span<const float> mask);

    // True when t has reached the next .bin cadence.
    bool wantsFrame(double t) const;
    // True when ANY kind is due -- the guard a runner puts in front of the
    // derived-field kernels, so a frame's GPU work is skipped when nothing
    // would consume it. frame() re-tests each kind itself.
    bool wantsAnyFrame(double t) const;

    // Writes whichever kinds are enabled and due at time t (`fields`: the
    // fields of one time level). Advances the cadences it consumed. A failure
    // is returned AND recorded for finish().
    [[nodiscard]] std::expected<void, std::string>
    frame(double t, std::span<const FrameField> fields);

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

    // The end of a run's output contract: closes every stream checked (a
    // buffered ENOSPC surfaces only at close), rewrites the .json, and returns
    // the FIRST write failure of the whole run, if any -- the caller exits 4
    // on it. Idempotent; the destructor does the unchecked close only when
    // finish() was never called.
    [[nodiscard]] std::expected<void, std::string> finish();

    int frames() const;
    // Recorded in the final .json. Call before finish() / destruction.
    void setMsPerStep(double);

private:
    // Per-kind cadences behind wantsAnyFrame(). The live shared-memory bridge
    // (--emit shm) shares the web cadence but has its own predicate, because a
    // run can want one without the other.
    bool wantsPng(double t) const;
    bool wantsWeb(double t) const;
    bool wantsShm(double t) const;
    // frame() without the failure bookkeeping.
    std::expected<void, std::string> frameKinds(double t,
                                                std::span<const FrameField> fields);

    RunOutput();
    struct Impl;
    std::unique_ptr<Impl> p_;
};

// Downloads nothing: argmaxes each component of an already-host-side field and
// prints the offending cells. Called on a non-finite state before exiting 7.
void reportBlowup(const Grid&, const VectorField& host, const Scalar& lambda,
                  double t);

} // namespace dns
