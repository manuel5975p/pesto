// The edited document, and the bridge to dns_core.
//
// The design rule here is that the UI OWNS NO PHYSICS. Every number it shows
// comes from running the real pipeline:
//
//   Doc fields -> RawConfig -> dns::validate() -> Case
//                           -> dns::buildCaseGeometry() -> sdf + CutGeometry
//                           -> dns::sanitizeCut() -> CutSanitizeStats
//
// so the grid you see IS c.grid, the cut cells you see ARE the apertures the
// runner would consume, and an invalid edit surfaces the validator's own
// message (with its exit code) rather than a second, drifting opinion. That is
// also why the editor cannot show a case the CLI would reject: the same
// function decides.
//
// The build runs on a worker thread -- sampleCut on a 512^3 grid is seconds --
// and results are published under a revision counter, so a stale build is
// discarded rather than shown.
#pragma once

#include "dns/casegeom.hpp"
#include "dns/config.hpp"
#include "dns/remap.hpp"
#include "dns/scene.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace ui {

// What the panels edit. A deliberately small subset of RawConfig: the flags
// that change the GRID or the BODY, which is what this tool is for. Everything
// else (run length, emit set, solver tolerances) belongs to the run, not to
// the geometry, and is left at its default so the exported command line is
// short and honest about what was actually inspected.
struct DocState
{
    int n = 256;
    int ny = 0;          // 0 = follow n
    int nz = 1;          // 1 = native 2D
    double dr = 8.0;
    int physics = 0;     // 0 incompressible, 1 compressible
    double re = 200.0;
    // Run length. The editor does not time-march, but validate() requires it
    // and the exported command line must be runnable as printed.
    double ft = 1.0;
    double mach = 0.2;
    double les = 0.0;    // Cs; 0 = DNS
    int sgs = 0;         // 0 smag, 1 sigma
    int far = 0;         // index into farNames()
    bool cutFlux = false;
    double penalize = 0.0;

    // Focal clustering (compressible only).
    bool focalAuto = false;
    double focalRatio = 4.0;
    double focalBlend = 0.5;
    double focalWake = 5.0;
    double focalPad = 0.25;

    // Body: either one of the built-in kinds, or a scene of objects.
    int bodyKind = 1;    // index into bodyNames()
    std::string meshPath;
    double aspect = 3.5;
    int dimples = 0;

    dns::Scene scene;    // used when bodyKind selects "scene"
};

const char* const* bodyNames();   // nullptr-terminated
int bodyNameCount();
const char* const* farNames();
int farNameCount();
const char* const* objectKindNames();
int objectKindCount();

// One completed build. Everything the viewport and the panels read.
struct Build
{
    uint64_t revision = 0;
    bool ok = false;
    std::string error;     // validator or geometry message, verbatim
    int errorCode = 0;     // the exit code the CLI would return

    // dns::Case has a private constructor -- validate() is the only way to
    // make one, which is the point (CLAUDE.md: "The ONLY way to obtain a Case
    // is validate()"). So it is held by optional and only ever read when ok.
    std::optional<dns::Case> caseOpt;
    const dns::Case& cas() const { return *caseOpt; }

    dns::CaseGeometry geo;
    dns::CutSanitizeStats sanitize{};
    dns::AxisRemap3 remap;   // identity unless --focal is active
    bool remapActive = false;

    // Cell census over the whole grid.
    std::size_t nCells = 0, nSolid = 0, nCut = 0;
    // World box of every non-fluid cell, in REFERENCE coordinates. Measured
    // from the cut rather than taken from CaseGeometry::bbox, which only the
    // scene and mesh paths fill in -- and it is honest for the bodies that
    // span the domain (fence, floor), where a placement-derived box is not.
    double solidBox[6] = {0, 0, 0, 0, 0, 0};
    bool hasSolidBox = false;
    double minLambda = 1.0, minAperture = 1.0;
    double buildSeconds = 0;

    // Resolution verdict, computed from the RESOLVED case (not the inputs).
    double dOverH = 0;       // body diameter in cells (smallest spacing)
    double blCells = 0;      // cells across the laminar BL, (D/h)/sqrt(Re)
    bool bodyResolved = false;   // blCells >= 3 (CLAUDE.md's DNS criterion)
    std::string label;       // "DNS" / "LES (sigma, Cs = 1.35)"
    std::string describe;    // Case::describe(), i.e. the --dry-run text
};

class Doc
{
public:
    Doc();
    ~Doc();

    DocState st;

    // Marks the document dirty; the worker picks it up on the next poll.
    void touch() { ++revision_; }
    uint64_t revision() const { return revision_; }

    // Call once per frame. Starts a build when the document changed and no
    // build is running, and publishes a finished one. Returns true when a NEW
    // build became current (the viewport must rebuild its meshes).
    bool poll();
    bool building() const { return building_; }

    const Build& current() const { return *current_; }

    // --- object editing (scene mode) ---------------------------------------
    int addObject(dns::SceneObjectKind);
    void removeObject(int index);
    int selected = -1;

    // --- export -------------------------------------------------------------
    std::string sceneToml() const;
    std::string commandLine() const;
    // Writes the scene TOML; returns an error string on failure.
    std::string writeScene(const std::string& path) const;

    bool sceneMode() const;

private:
    static void build(const DocState&, uint64_t rev, Build& out);

    std::shared_ptr<Build> current_;
    std::atomic<uint64_t> revision_{1};
    uint64_t builtRevision_ = 0;

    std::thread worker_;
    std::mutex mu_;
    std::condition_variable cv_;
    DocState pending_{};
    uint64_t pendingRev_ = 0;
    std::shared_ptr<Build> done_;
    bool building_ = false;
    bool quit_ = false;
};

} // namespace ui
