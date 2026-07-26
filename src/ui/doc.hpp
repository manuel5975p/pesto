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
#include "ui/units.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace ui {

using ObjId = uint32_t;                 // 0 = none
inline constexpr ObjId kNoObj = 0;

// A scene object plus the identity the EDITOR needs and the scene format has no
// business carrying. `obj` is the dns_core value verbatim: this WRAPS, never
// forks -- a field added here can never reach the solver, because sceneOf()
// only ever copies `obj`.
//
// Identity exists because selection used to be an index: deleting a
// lower-indexed object silently retargeted the selection to a different body.
struct SceneItem
{
    dns::SceneObject obj;
    ObjId id = kNoObj;
};

// What the panels edit. A deliberately small subset of RawConfig: the flags
// that change the GRID or the BODY, which is what this tool is for. Everything
// else (run length, emit set, solver tolerances) belongs to the run, not to
// the geometry, and is left at its default so the exported command line is
// short and honest about what was actually inspected.
struct DocState
{
    // A 96^3 box by default: 3D from the start, and small enough that the cut
    // rebuilds interactively while you drag an object.
    int n = 96;
    int ny = 0;          // 0 = follow n
    int nz = 96;         // 1 = native 2D
    double dr = 8.0;
    int physics = 0;     // 0 incompressible, 1 compressible

    // The PHYSICAL face of the case (units.hpp). Re and Mach are no longer
    // edited directly -- they are derived from these, so the panel cannot show
    // a Reynolds number that disagrees with the metres and m/s next to it.
    int material = 0;    // index into materials(); last entry = custom
    Physical phys;

    double re() const { return phys.re(); }
    double mach() const { return phys.mach(); }

    // Run length. The editor does not time-march, but validate() requires it
    // and the exported command line must be runnable as printed.
    double ft = 1.0;
    // How often the run publishes a live frame (--wdt), in solver time units --
    // the cadence of the shared-memory stream, and therefore the frame rate of
    // everything the viewer shows. The solver's own default is 0.15, which is a
    // handful of frames for a short case; it is an EDITED value here because
    // "the results come too slowly" is otherwise unfixable from the UI.
    double wdt = 0.05;
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
    // Per-body options. Each is only MEANINGFUL for some bodies (see
    // bodyOptions); the value is kept while you switch away and back, but it is
    // never emitted for a body that does not accept it.
    double aspect = 3.5;   // lens
    int dimples = 0;       // cylinder, sphere
    int bumps = 0;         // floor
    double bumpH = 8;      // floor, cells
    double meshScale = 0;  // mesh, 0 = fit extent to D
    bool robust = false;   // mesh

    std::vector<SceneItem> items;   // used when bodyKind selects "scene"
};

// The dns::Scene this document means: the ONE place model/cs/sgs are merged
// from the panel's --les/--sgs onto the object list.
dns::Scene sceneOf(const DocState&);

// The scene as TOML, in exactly the schema dns::parseScene accepts. Free
// function because the build worker must be able to call it without a Doc.
std::string sceneTomlOf(const DocState&);

// Which per-body options a body kind accepts. ONE table, read by both the
// panel (what to show) and toRaw/commandLine (what to emit) -- so a dimple
// count left over from a cylinder cannot ride along to a cube, and the panel
// cannot offer a knob the solver would reject.
struct BodyOpts
{
    bool aspect = false;   // --aspect
    bool dimples = false;  // --dimples
    bool bumps = false;    // --bumps, --bump-h
    bool mesh = false;     // --scale, --robust
};
BodyOpts bodyOptions(int bodyKind);

const char* const* bodyNames();   // nullptr-terminated
int bodyNameCount();
const char* const* farNames();
int farNameCount();

// The far fields that are legal for a given physics model, as indices into
// farNames(). The split is not a UI opinion: validate() rejects `--far tunnel`
// under compressible with "tunnel is an incompressible far field", and
// vice-versa, so offering all five in one control means two of every five
// clicks are invalid by construction. Body-specific restrictions (fence needs
// tunnel, floor needs closed y walls) are NOT encoded here -- those stay the
// validator's to report, because duplicating them is how a second, drifting
// opinion gets born.
std::span<const int> farForPhysics(int physics);
// A valid far index for `physics`, preferring `current` when it is already
// legal. Compressible falls back to nscbc, never sponge (CLAUDE.md: sponge is
// never a default, anywhere).
int remapFar(int current, int physics);
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
    // Scene mode with no objects. NOT an error here: it is the state you pass
    // through while building a scene, and refusing to render the grid would
    // make the editor unusable exactly when you need it. `dns --scene` DOES
    // reject it (exit 3), so it is surfaced as a loud warning and the export
    // refuses to write -- see writeScene, which pre-flights the parser.
    bool sceneEmpty = false;
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
    // `seed` is where the user clicked, in units of D from the reference
    // placement; absent means "find a free spot near the selection".
    ObjId addObject(dns::SceneObjectKind,
                    const std::optional<std::array<double, 3>>& seed = {});
    void removeObject(ObjId);
    void removeSelection();
    ObjId duplicateObject(ObjId);
    bool renameObject(ObjId, std::string_view desired);
    // N-1 further copies at a fixed pitch (in D). Returns how many were made:
    // copies that would leave the domain are dropped, not forced.
    int arrayObjects(ObjId, int count, double dx, double dy, double dz);
    // Deltas applied to every selected object, in units of D.
    void nudgeSelection(double dx, double dy, double dz);
    void scaleSelection(double factor);

    // Selection is by id, never by index.
    enum class Sel { Replace, Add, Toggle, Range };
    void select(ObjId, Sel = Sel::Replace);
    void clearSelection();
    bool isSelected(ObjId) const;
    ObjId primary() const;   // the inspector's subject: last clicked
    std::span<const ObjId> selection() const { return sel_; }
    int indexOf(ObjId) const;
    SceneItem* find(ObjId);
    const SceneItem* find(ObjId) const;
    ObjId idOfLabel(std::string_view) const;
    ObjId idAt(int index) const;

    // Sanitize to the charset, then disambiguate with a numeric suffix until
    // dns::checkSceneNames would accept it.
    std::string uniqueName(std::string_view desired, ObjId skip = kNoObj) const;

    // A free spot for a new object of diameter `d`, in units of D. Uses the
    // last successful build's grid and dns::sceneBoxInside/sceneObjectBox --
    // never a private copy of the boundary rule.
    struct Placement3
    {
        double x = 0, y = 0, z = 0, d = 1.0;
        bool crowded = false;   // had to overlap: the domain is full
        bool shrunk = false;    // d was reduced to make it fit
    };
    Placement3 proposePlacement(dns::SceneObjectKind, double d,
                                const std::optional<std::array<double, 3>>& seed = {},
                                ObjId ignore = kNoObj) const;

    // A transient message for the status strip. NOT part of the document, so
    // it never lands in the undo stack.
    const std::string& notice() const { return notice_; }
    void clearNotice() { notice_.clear(); }

    // --- undo ---------------------------------------------------------------
    // An interaction transaction: beginEdit captures the pre-state, endEdit
    // commits ONE entry iff the revision moved. Nested begins are refcounted.
    void beginEdit(std::string_view label);
    void endEdit();
    bool undo();
    bool redo();
    bool canUndo() const { return !undo_.empty(); }
    bool canRedo() const { return !redo_.empty(); }
    std::string_view undoLabel() const;

    // --- export -------------------------------------------------------------
    std::string sceneToml() const { return sceneTomlOf(st); }
    std::string commandLine() const;
    // The same command as argv, for actually executing it (ui/runner.hpp).
    std::vector<std::string> commandArgs(const std::string& exe,
                                         const std::string& scenePath) const;
    // Writes the scene TOML; returns an error string on failure. Pre-flights
    // the parser, so it can never write a file the CLI would reject.
    std::string writeScene(const std::string& path) const;
    // Opens a scene .toml through dns::loadScene -- the CLI's own path.
    std::string openScene(const std::string& path);

    bool sceneMode() const;

private:
    static void build(const DocState&, uint64_t rev, Build& out);
    void pruneSelection();

    // RAII transaction: every mutating Doc method opens one, so undo works
    // without discipline at the call sites.
    struct Edit
    {
        Doc& d;
        Edit(Doc& doc, std::string_view label) : d(doc) { d.beginEdit(label); }
        ~Edit() { d.endEdit(); }
    };

    std::shared_ptr<Build> current_;
    std::atomic<uint64_t> revision_{1};
    uint64_t builtRevision_ = 0;

    std::vector<ObjId> sel_;
    // Lives OUTSIDE DocState so it stays monotone across undo: if it were
    // snapshotted, undoing a delete and then adding would mint an id that a
    // state on the redo stack also uses.
    ObjId nextId_ = 1;
    std::string notice_;

    struct Snapshot { DocState st; std::vector<ObjId> sel; std::string label; };
    std::vector<Snapshot> undo_, redo_;
    Snapshot preEdit_;
    uint64_t preRev_ = 0;
    int editDepth_ = 0;
    static constexpr std::size_t kUndoMax = 200;
    void commitEdit();

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
