#include "ui/doc.hpp"

#include "dns/geometry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <charconv>
#include <cctype>
#include <fstream>

namespace ui {
namespace {

// Order must match DocState::bodyKind. "scene" is last because it is the one
// entry that turns the object list on.
const char* const kBodyNames[] = {"none",  "cylinder", "sphere", "cube", "lens",
                                  "fence", "floor",    "mesh",   "scene", nullptr};
const char* const kFarNames[] = {"tunnel", "fringe", "sponge", "nscbc", "periodic",
                                 nullptr};
const char* const kObjectKinds[] = {"cylinder", "sphere", "cube", "lens", "mesh",
                                    nullptr};

std::string num(double v, int dec = 4)
{
    char b[64];
    std::snprintf(b, sizeof b, "%.*g", dec, v);
    return b;
}

} // namespace

const char* const* bodyNames() { return kBodyNames; }
int bodyNameCount() { return 9; }
const char* const* farNames() { return kFarNames; }
int farNameCount() { return 5; }

// Indices into kFarNames: 0 tunnel, 1 fringe, 2 sponge, 3 nscbc, 4 periodic.
// Measured against the validator, not guessed -- `dns flow --dry-run` over the
// full body x far x physics matrix rejects the cross terms with "tunnel is an
// incompressible far field" / "sponge is a compressible far field". Periodic is
// the only one both accept.
std::span<const int> farForPhysics(int physics)
{
    static constexpr int kIncomp[] = {0, 1, 4};
    static constexpr int kComp[] = {3, 2, 4};   // nscbc first: it is the default
    return physics == 1 ? std::span<const int>(kComp) : std::span<const int>(kIncomp);
}

int remapFar(int current, int physics)
{
    for (int i : farForPhysics(physics))
        if (i == current) return current;
    // Compressible falls back to NSCBC, never sponge: CLAUDE.md is explicit
    // that sponge is never a default anywhere, in any physics or mode -- it is
    // an escape hatch the user asks for by name.
    return physics == 1 ? 3 : 0;
}

BodyOpts bodyOptions(int bodyKind)
{
    switch (bodyKind) {
    case 1: return {.dimples = true};                  // cylinder -> 2D disks
    case 2: return {.dimples = true};                  // sphere -> 3D caps
    case 4: return {.aspect = true};                   // lens
    case 6: return {.bumps = true};                    // floor trip spikes
    case 7: return {.mesh = true};                     // mesh: scale, robust
    default: return {};                                // none/cube/fence/scene
    }
}
const char* const* objectKindNames() { return kObjectKinds; }
int objectKindCount() { return 5; }

bool Doc::sceneMode() const { return st.bodyKind == 8; }

Doc::Doc()
{
    current_ = std::make_shared<Build>();
    // One default object so scene mode is never an empty screen -- and an
    // empty scene is an exit-3 error from the parser, by design.
    dns::SceneObject o;
    o.label = dns::defaultSceneName(o.kind, 0);
    st.items.push_back({o, nextId_++});
    sel_.push_back(st.items.front().id);

    worker_ = std::thread([this] {
        for (;;) {
            DocState job;
            uint64_t rev = 0;
            {
                std::unique_lock lk(mu_);
                cv_.wait(lk, [&] { return quit_ || pendingRev_ != 0; });
                if (quit_) return;
                job = pending_;
                rev = pendingRev_;
                pendingRev_ = 0;
            }
            auto out = std::make_shared<Build>();
            build(job, rev, *out);
            {
                std::lock_guard lk(mu_);
                done_ = out;
            }
        }
    });
}

Doc::~Doc()
{
    {
        std::lock_guard lk(mu_);
        quit_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

// DocState -> RawConfig. Free, because both the (const) accessor and the
// worker thread need it and the worker must not touch a Doc.
// The convection half of rawFrom. Split out because it shares almost nothing
// with the body-flow case: no body, no far field, no --dr, and its Rayleigh
// number comes from the SI face rather than a --re.
static dns::RawConfig rawConvFrom(const DocState& s)
{
    dns::RawConfig r;
    r.sub = dns::Subcommand::Conv;
    r.out = "conv";
    r.n = s.n;
    r.ny = s.ny > 0 ? s.ny : s.n;
    r.nz = s.nz;
    r.physics = s.convPhysics == 1 ? dns::Physics::Compressible
                                   : dns::Physics::Incompressible;
    r.rayleigh = s.convRayleigh();
    r.prandtl = s.prandtl;
    r.convAspect = s.convAspect;
    r.hotPatch = s.hotPatch;
    r.sides = s.convSides;
    r.tt = s.convTt;
    r.kick = s.convKick;
    r.wdt = s.wdt;
    r.given = {"--n", "--ny", "--nz", "--physics", "--rayleigh", "--prandtl",
               "--aspect", "--tt", "--kick", "--wdt"};
    if (s.hotPatch < 1.0) r.given.insert("--hot-patch");
    if (s.convSides == 1) r.given.insert("--sides");
    // Compressible-only knobs. The Boussinesq branch REJECTS these by name
    // rather than ignoring them, so offering them there would be offering an
    // error -- see the validator's conv branch.
    if (s.convPhysics == 1) {
        r.mach = s.convMach;
        r.dtemp = s.convDtemp();
        r.visc = s.visc;
        r.given.insert("--mach");
        r.given.insert("--dtemp");
        if (s.visc == 1) {
            r.tinf = s.convTmean;
            r.given.insert("--visc");
            r.given.insert("--tinf");
        }
    }
    return r;
}

static dns::RawConfig rawFrom(const DocState& s)
{
    if (s.isConv()) return rawConvFrom(s);
    dns::RawConfig r;
    r.sub = dns::Subcommand::Flow;
    r.out = "flow";
    r.n = s.n;
    r.nz = s.nz;
    r.dr = s.dr;
    r.re = s.re();
    r.ft = s.ft;
    r.mach = s.mach();
    r.physics = s.physics == 1 ? dns::Physics::Compressible
                               : dns::Physics::Incompressible;
    r.far = dns::FarField(s.far);
    r.cutFlux = s.cutFlux;
    r.penalize = s.penalize;
    r.sgs = s.sgs == 1 ? dns::SgsModel::Sigma : dns::SgsModel::Smagorinsky;
    // aspect/dimples/bumps/scale/robust are NOT set here: they are per-body and
    // are applied below, gated on bodyOptions(). Assigning them unconditionally
    // is what let a cylinder's dimple count follow you to a cube.

    // `given` is how validate() tells "left at the default" from "set to the
    // default" (--ny follows --n unless typed). The editor always means what
    // it shows, so every field it exposes is marked given.
    r.given = {"--n", "--nz", "--dr", "--re", "--ft", "--physics", "--far"};
    if (s.ny > 0) {
        r.ny = s.ny;
        r.given.insert("--ny");
    }
    if (s.physics == 1) r.given.insert("--mach");
    if (s.penalize > 0) r.given.insert("--penalize");

    if (s.bodyKind == 8) {
        r.scene = "<editor>";   // recorded, never opened (phase 1.5 is ours)
        r.given.insert("--scene");
    } else {
        r.body = s.bodyKind == 7 ? s.meshPath : kBodyNames[s.bodyKind];
        r.given.insert("--body");
    }

    // Per-body options, gated on the ONE capability table. A value the current
    // body does not accept is discarded rather than passed: keeping it would
    // either be rejected by the validator or, worse, silently honoured.
    const BodyOpts bo = bodyOptions(s.bodyKind);
    if (bo.aspect) {
        r.aspect = s.aspect;
        r.given.insert("--aspect");
    }
    if (bo.dimples && s.dimples > 0) {
        r.dimples = s.dimples;
        r.given.insert("--dimples");
    }
    if (bo.bumps && s.bumps > 0) {
        r.bumps = s.bumps;
        r.bumpH = s.bumpH;
        r.given.insert("--bumps");
        r.given.insert("--bump-h");
    }
    if (bo.mesh) {
        if (s.meshScale > 0) {
            r.scale = s.meshScale;
            r.given.insert("--scale");
        }
        if (s.robust) {
            r.robust = true;
            r.given.insert("--robust");
        }
    }

    if (s.les > 0 && s.bodyKind != 8) {
        r.les = s.les;
        r.given.insert("--les");
        r.given.insert("--sgs");
    }

    // Thermal knobs (compressible only, and the validator says so for every
    // one of them). Marked `given` only when they are actually engaged: a
    // document that leaves them alone must export the command it always did,
    // because that is what keeps the measured anchors valid.
    if (s.physics == 1) {
        r.prandtl = s.prandtl;
        r.given.insert("--prandtl");
        if (s.visc == 1) {
            r.visc = 1;
            r.tinf = s.tinf;
            r.given.insert("--visc");
            r.given.insert("--tinf");
        }
        if (s.wallIso) {
            r.wallIso = 1;
            r.twall = s.twall;
            r.given.insert("--wall");
            r.given.insert("--twall");
        }
        if (s.nscbcTinflow) {
            r.nscbcTinflow = true;
            r.given.insert("--nscbc-tinflow");
        }
        if (s.tempFrames) {
            r.tempFrames = true;
            r.given.insert("--temp");
        }
    }

    if (s.physics == 1 && s.focalAuto) {
        r.focal = "auto";
        r.focalRatio = s.focalRatio;
        r.focalBlend = s.focalBlend;
        r.focalWake = s.focalWake;
        r.focalPad = s.focalPad;
        r.given.insert("--focal");
        r.given.insert("--focal-ratio");
        r.given.insert("--focal-blend");
    }
    return r;
}

void Doc::build(const DocState& s, uint64_t rev, Build& out)
{
    const auto t0 = std::chrono::steady_clock::now();
    out.revision = rev;

    dns::RawConfig r = rawFrom(s);
    const bool scene = s.bodyKind == 8;
    dns::Scene sc;
    if (scene) {
        // THE SERIALIZE-PARSE GATE. The editor's scene is DEFINED as
        // parseScene(sceneToml()): the document is serialized and run back
        // through the real parser before anything else sees it. Microseconds
        // against seconds of sampleCut, and it buys three things that no
        // UI-side rule could:
        //   - an empty scene, a bad name, a mesh with no path all surface as
        //     the CLI's own exit-3 message, verbatim;
        //   - sceneToml() cannot silently rot into a schema the parser rejects,
        //     because every build exercises the round trip;
        //   - "write scene.toml" plus the printed --scene command are provably
        //     a runnable pair.
        if (s.items.empty()) {
            // The one state the gate must NOT fail on: an empty scene is where
            // every scene starts and where deleting the last object leaves you.
            // Failing here loses the grid, the domain box and the ability to
            // add anything back -- a dead end. Build it as a bodyless case and
            // flag it instead.
            out.sceneEmpty = true;
            sc = sceneOf(s);
        } else {
            auto parsed = dns::parseScene(sceneTomlOf(s));
            if (!parsed) {
                out.ok = false;
                out.error = parsed.error();
                out.errorCode = 3;   // the class the CLI gives a bad scene file
                return;
            }
            sc = std::move(*parsed);
        }
        if (auto m = dns::applySceneModel(sc, r); !m) {
            out.ok = false;
            out.error = dns::format(m.error());
            out.errorCode = m.error().code;
            return;
        }
    }

    auto cas = dns::validate(r);
    if (!cas) {
        out.ok = false;
        out.error = dns::format(cas.error());
        out.errorCode = cas.error().code;
        return;
    }
    out.caseOpt = *cas;
    out.describe = cas->describe();

    if (s.isConv()) {
        // A convection cell has no body, so there is no sdf to sample and no
        // cut to sanitize -- the whole geometry stage is skipped, which is also
        // why a conv document rebuilds instantly at any resolution. The cut is
        // filled all-fluid so every viewport layer that walks it finds nothing
        // solid rather than walking an empty array.
        const dns::Grid& g = out.cas().grid;
        out.geo.cut = dns::CutGeometry{dns::Scalar(g.size(), 1.0),
                                       dns::Scalar(g.size(), 1.0),
                                       dns::Scalar(g.size(), 1.0),
                                       dns::Scalar(g.size(), 1.0)};
        out.geo.D = out.cas().convH;   // the layer height IS the length scale
        out.nCells = g.size();
        out.remap.fx = dns::Focal1D::identity(double(g.nx) * g.hx);
        out.remap.fy = dns::Focal1D::identity(double(g.ny) * g.hy);
        out.remap.fz = dns::Focal1D::identity(double(g.nz) * g.hz);
        out.remapActive = false;

        // The resolution question here is not a body boundary layer but the
        // THERMAL one at the plates, delta = H / (2 Nu). Nu is not known before
        // the run, so this uses the standard a-priori Nu ~ 0.1 Ra^(1/3) -- which
        // runs ~20% high against this repo's own measurements (Nu 7.98 at
        // Ra 1e6, 10.56 at 2e6), and therefore errs toward reporting the layer
        // as thinner and the grid as coarser than it is.
        const double nuEst =
            out.cas().rayleigh > 0
                ? std::max(1.0, 0.1 * std::cbrt(out.cas().rayleigh))
                : 1.0;
        const double dTh = out.cas().convH / (2.0 * nuEst);
        out.dOverH = out.cas().convH / g.hy;   // cells across the layer
        out.blCells = dTh / g.hy;
        // Five points across the thermal layer is the usual floor for a
        // resolved Rayleigh-Benard DNS; below it the wall gradient -- and so
        // the Nusselt number -- is a property of the grid.
        out.bodyResolved = out.blCells >= 5.0;
        out.label = "DNS";
        out.ok = true;
        out.buildSeconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - t0).count();
        return;
    }

    auto geo = dns::buildCaseGeometry(*cas, scene ? &sc : nullptr);
    if (!geo) {
        out.ok = false;
        out.error = geo.error();
        out.errorCode = 3;   // geometry build touches the filesystem: INPUT
        return;
    }
    out.geo = std::move(*geo);

    // The runners sanitize before solving (apps/dns_main.cpp), so the cut the
    // viewer draws is the sanitized one -- otherwise it would show slivers the
    // solver never sees.
    out.sanitize = dns::sanitizeCut(out.cas().grid, out.geo.cut);

    const dns::Grid& g = out.cas().grid;
    out.nCells = g.size();
    double minL = 1.0, minA = 1.0;
    int bi0 = g.nx, bi1 = -1, bj0 = g.ny, bj1 = -1, bk0 = g.nz, bk1 = -1;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t id = g.idx(i, j, k);
                const double l = out.geo.cut.lambda[id];
                if (l < 1.0) {
                    if (l <= 0) ++out.nSolid;
                    else {
                        ++out.nCut;
                        minL = std::min(minL, l);
                    }
                    bi0 = std::min(bi0, i); bi1 = std::max(bi1, i);
                    bj0 = std::min(bj0, j); bj1 = std::max(bj1, j);
                    bk0 = std::min(bk0, k); bk1 = std::max(bk1, k);
                }
                const double a = std::min({out.geo.cut.ax[id], out.geo.cut.ay[id],
                                           out.geo.cut.az[id]});
                if (a > 0) minA = std::min(minA, a);
            }
    if (bi1 >= 0) {
        out.hasSolidBox = true;
        out.solidBox[0] = bi0 * g.hx;
        out.solidBox[1] = bj0 * g.hy;
        out.solidBox[2] = bk0 * g.hz;
        out.solidBox[3] = (bi1 + 1) * g.hx;
        out.solidBox[4] = (bj1 + 1) * g.hy;
        out.solidBox[5] = (bk1 + 1) * g.hz;
    }
    out.minLambda = minL;
    out.minAperture = minA;

    // Focal remap, built exactly as runner_comp does.
    const double axisL[3] = {double(g.nx) * g.hx, double(g.ny) * g.hy,
                             double(g.nz) * g.hz};
    dns::Focal1D* f[3] = {&out.remap.fx, &out.remap.fy, &out.remap.fz};
    for (int a = 0; a < 3; ++a)
        *f[a] = out.cas().focalActive && out.cas().focalHi[a] > out.cas().focalLo[a]
                    ? dns::Focal1D::make(axisL[a], out.cas().focalLo[a],
                                         out.cas().focalHi[a], out.cas().focalW[a],
                                         out.cas().focalRatio)
                    : dns::Focal1D::identity(axisL[a]);
    out.remapActive = out.cas().focalActive;

    // Resolution verdict. h is the FINEST spacing present -- with --focal that
    // is the plateau spacing, which is the one the body actually sits on.
    double h = std::min(g.hx, g.hy);
    if (g.nz > 1) h = std::min(h, g.hz);
    if (out.cas().focalActive)
        for (int a = 0; a < 3; ++a)
            if (out.cas().focalHFine[a] > 0) h = std::min(h, out.cas().focalHFine[a]);
    out.dOverH = out.geo.D > 0 ? out.geo.D / h : out.cas().placement.D / h;
    out.blCells = out.cas().re > 0 ? out.dOverH / std::sqrt(out.cas().re) : 0;
    out.bodyResolved = out.blCells >= 3.0;

    if (out.cas().lesLabel) {
        out.label = std::string("LES (") +
                    (out.cas().sgsModel == dns::SgsModel::Sigma ? "sigma" : "smag") +
                    ", Cs = " + num(out.cas().cs) + ")";
    } else {
        out.label = "DNS";
    }

    out.ok = true;
    out.buildSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

bool Doc::poll()
{
    std::shared_ptr<Build> got;
    {
        std::lock_guard lk(mu_);
        if (done_) {
            got = done_;
            done_.reset();
            building_ = false;
        }
        if (!building_ && builtRevision_ != revision_) {
            pending_ = st;
            pendingRev_ = revision_;
            builtRevision_ = revision_;
            building_ = true;
            cv_.notify_one();
        }
    }
    if (got) {
        current_ = got;
        return true;
    }
    return false;
}

// --- the scene as dns_core sees it -------------------------------------------

dns::Scene sceneOf(const DocState& st)
{
    dns::Scene sc;
    // The scene, not --les, carries the model in scene mode: that is the rule
    // applySceneModel() enforces, and going through it means the editor cannot
    // produce a scene the CLI would reject as conflicting.
    if (st.les > 0) {
        sc.model = dns::SceneModel::Les;
        sc.cs = st.les;
        sc.sgs = st.sgs == 1 ? dns::SgsModel::Sigma : dns::SgsModel::Smagorinsky;
    } else {
        sc.model = dns::SceneModel::Dns;
        sc.cs = 0;
    }
    sc.objects.reserve(st.items.size());
    for (const auto& it : st.items) sc.objects.push_back(it.obj);
    return sc;
}

namespace {

// Shortest representation that round-trips bit-exactly, so
// parseScene(sceneToml(x)) recovers the very doubles that were edited -- which
// is what makes the round-trip test an equality test, not a tolerance test.
std::string tomlNum(double v)
{
    char buf[40];
    auto [end, ec] = std::to_chars(buf, buf + sizeof buf, v);
    std::string s(buf, end);
    if (ec != std::errc{}) return "0.0";
    // TOML types a bare integer as an integer node; the parser wants a float.
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos
        && s.find("inf") == std::string::npos && s.find("nan") == std::string::npos)
        s += ".0";
    return s;
}

} // namespace

// Exactly the schema dns::parseScene accepts -- key for key. The previous
// version emitted `model = "dns"` (a scalar, not a [model] table), `label`
// (the key is `name`) and `mesh` (the key is `path`), so "write scene.toml"
// plus the printed `--scene scene.toml` were a pair that could not be run.
// Doc::build() now parses this back on every build, so a drift here is a
// visible error in the editor rather than a broken file discovered later.
std::string sceneTomlOf(const DocState& st)
{
    std::string s = "# written by dns-ui\n\n[model]\n";
    if (st.les > 0) {
        s += "type = \"les\"\n";
        s += "cs = " + tomlNum(st.les) + "\n";
        s += std::string("sgs = \"") + (st.sgs == 1 ? "sigma" : "smag") + "\"\n";
    } else {
        s += "type = \"dns\"\n";
    }
    for (const auto& it : st.items) {
        const dns::SceneObject& o = it.obj;
        s += "\n[[objects]]\n";
        s += std::string("kind = \"") + dns::name(o.kind) + "\"\n";
        if (!o.label.empty()) s += "name = \"" + o.label + "\"\n";
        // Always explicit: this is a file a human will edit, and an explicit
        // name also makes the parser's index-based defaulting a no-op on our
        // output -- which is what keeps reordering from renaming anything.
        s += "x = " + tomlNum(o.x) + "\n";
        s += "y = " + tomlNum(o.y) + "\n";
        s += "z = " + tomlNum(o.z) + "\n";
        s += "d = " + tomlNum(o.d) + "\n";
        if (o.kind == dns::SceneObjectKind::Lens)
            s += "aspect = " + tomlNum(o.aspect) + "\n";
        if (o.kind == dns::SceneObjectKind::Mesh) {
            s += "path = \"" + o.meshPath + "\"\n";
            if (o.scale > 0) s += "scale = " + tomlNum(o.scale) + "\n";
            if (o.pitch != 0) s += "pitch = " + tomlNum(o.pitch) + "\n";
            if (o.yaw != 0) s += "yaw = " + tomlNum(o.yaw) + "\n";
            if (o.robust) s += "robust = true\n";
        }
    }
    return s;
}

// --- identity and selection --------------------------------------------------

int Doc::indexOf(ObjId id) const
{
    for (std::size_t i = 0; i < st.items.size(); ++i)
        if (st.items[i].id == id) return int(i);
    return -1;
}

ObjId Doc::idAt(int index) const
{
    return index >= 0 && index < int(st.items.size()) ? st.items[size_t(index)].id
                                                      : kNoObj;
}

SceneItem* Doc::find(ObjId id)
{
    const int i = indexOf(id);
    return i < 0 ? nullptr : &st.items[size_t(i)];
}

const SceneItem* Doc::find(ObjId id) const
{
    const int i = indexOf(id);
    return i < 0 ? nullptr : &st.items[size_t(i)];
}

ObjId Doc::idOfLabel(std::string_view label) const
{
    for (const auto& it : st.items)
        if (it.obj.label == label) return it.id;
    return kNoObj;
}

bool Doc::isSelected(ObjId id) const
{
    return std::find(sel_.begin(), sel_.end(), id) != sel_.end();
}

ObjId Doc::primary() const { return sel_.empty() ? kNoObj : sel_.back(); }

void Doc::clearSelection() { sel_.clear(); }

void Doc::select(ObjId id, Sel how)
{
    if (id == kNoObj) return;
    switch (how) {
    case Sel::Replace:
        sel_.assign(1, id);
        break;
    case Sel::Add:
        if (!isSelected(id)) sel_.push_back(id);
        break;
    case Sel::Toggle:
        if (isSelected(id)) std::erase(sel_, id);
        else sel_.push_back(id);
        break;
    case Sel::Range: {
        const int a = sel_.empty() ? indexOf(id) : indexOf(sel_.front());
        const int b = indexOf(id);
        if (a < 0 || b < 0) return;
        const ObjId anchor = sel_.empty() ? id : sel_.front();
        sel_.assign(1, anchor);
        for (int i = std::min(a, b); i <= std::max(a, b); ++i)
            if (st.items[size_t(i)].id != anchor) sel_.push_back(st.items[size_t(i)].id);
        break;
    }
    }
}

// Drops ids that no longer exist. Runs after every structural mutation, which
// is what makes deletion safe: the selection never referred to an index.
void Doc::pruneSelection()
{
    std::erase_if(sel_, [&](ObjId id) { return indexOf(id) < 0; });
}

// --- naming ------------------------------------------------------------------

std::string Doc::uniqueName(std::string_view desired, ObjId skip) const
{
    std::string base;
    for (char c : desired)
        base += (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')
                    ? c
                    : '_';
    if (base.empty()) base = "object";
    auto taken = [&](const std::string& n) {
        for (const auto& it : st.items)
            if (it.id != skip && it.obj.label == n) return true;
        return false;
    };
    if (!taken(base)) return base;
    // Strip a trailing number so "cyl3" retries as "cyl4", not "cyl3-1".
    std::size_t cut = base.size();
    while (cut > 0 && std::isdigit(static_cast<unsigned char>(base[cut - 1]))) --cut;
    const std::string stem = cut ? base.substr(0, cut) : base;
    for (int n = 1; n < 100000; ++n) {
        const std::string cand = stem + std::to_string(n);
        if (!taken(cand)) return cand;
    }
    return base;
}

// --- placement ---------------------------------------------------------------

Doc::Placement3 Doc::proposePlacement(dns::SceneObjectKind kind, double d,
                                      const std::optional<std::array<double, 3>>& seed,
                                      ObjId ignore) const
{
    Placement3 out;
    out.d = d;
    const std::array<double, 3> s0 = seed.value_or(std::array<double, 3>{0, 0, 0});
    out.x = s0[0];
    out.y = s0[1];
    out.z = s0[2];

    const Build& b = *current_;
    // No usable domain yet: do not invent geometry. The build will report
    // whatever is actually wrong.
    if (!b.ok) return out;

    const dns::Grid& g = b.cas().grid;
    const dns::Placement& p = b.cas().placement;
    const bool flat = g.nz == 1;

    auto boxOf = [&](const dns::SceneObject& o) {
        return dns::sceneObjectBox(g, p, o);
    };
    auto probe = [&](double x, double y, double z, double dd) {
        dns::SceneObject o;
        o.kind = kind;
        o.x = x; o.y = y; o.z = z; o.d = dd;
        return o;
    };
    // Clearance is not a physics rule -- overlapping objects are legal
    // (docs/scene.md) -- it exists so per-object force columns survive:
    // apps/dns_main.cpp tests CELL-SNAPPED index boxes, so two world boxes
    // that merely touch can still snap to overlapping index boxes.
    const double gap = std::max(0.25 * d * p.D, 2.0 * std::max(g.hx, g.hy));
    auto clears = [&](const dns::SceneObject& cand) {
        const auto cb = boxOf(cand);
        for (const auto& it : st.items) {
            if (it.id == ignore) continue;
            const auto ob = boxOf(it.obj);
            const bool disjoint =
                cb[3] + gap < ob[0] || cb[0] - gap > ob[3] ||
                cb[4] + gap < ob[1] || cb[1] - gap > ob[4] ||
                (!flat && (cb[5] + gap < ob[2] || cb[2] - gap > ob[5]));
            if (!disjoint) return false;
        }
        return true;
    };

    double dMax = d;
    for (const auto& it : st.items) dMax = std::max(dMax, it.obj.d);
    // Lattice pitch = the two HALF extents plus the clearance. Summing the two
    // full diameters would reserve twice the space each pair needs and make the
    // domain look full at half occupancy.
    const double step = 0.5 * (d + dMax) + gap / p.D + 1e-3;

    // Ring search outward, downstream (+x) first: a second body in a flow
    // domain usually belongs in the wake.
    static constexpr double kDirs[8][2] = {{1, 0},  {1, 1},   {1, -1}, {0, 1},
                                           {0, -1}, {-1, 1},  {-1, -1}, {-1, 0}};
    // Two origins, in order: the seed (place it near what you were working on)
    // and then the reference centre. Seeding ONLY from the seed makes each add
    // chain outward from the last one until the chain reaches the wall, and the
    // domain then looks full while its middle is still empty.
    const std::array<double, 3> origins[2] = {s0, {0.0, 0.0, 0.0}};

    auto search = [&](const std::array<double, 3>& c, bool needClear) {
        for (int ring = 0; ring <= 16; ++ring) {
            const int zSteps = flat ? 0 : 4;
            for (int zi = 0; zi <= zSteps; ++zi) {
                const int zk = zi == 0 ? 0 : (zi % 2 ? (zi + 1) / 2 : -(zi / 2));
                for (int di = 0; di < (ring == 0 ? 1 : 8); ++di) {
                    const double x = c[0] + kDirs[di][0] * step * ring;
                    const double y = c[1] + kDirs[di][1] * step * ring;
                    const double z = c[2] + double(zk) * step;
                    const auto cand = probe(x, y, z, out.d);
                    if (!dns::sceneBoxInside(g, boxOf(cand))) continue;
                    if (needClear && !clears(cand)) continue;
                    out.x = x; out.y = y; out.z = z;
                    return true;
                }
            }
        }
        return false;
    };

    for (int attempt = 0; attempt < 3; ++attempt) {
        for (const auto& c : origins)
            if (search(c, true)) return out;
        // Nothing clears anywhere: admit an overlap rather than fail. Overlaps
        // are legal geometry (docs/scene.md); only the per-object force tally
        // is lost, and the caller says so.
        for (const auto& c : origins)
            if (search(c, false)) {
                out.crowded = true;
                return out;
            }
        out.d *= 0.6;      // not even in bounds: shrink and retry
        out.shrunk = true;
    }
    return out;
}

// --- object editing ----------------------------------------------------------

ObjId Doc::addObject(dns::SceneObjectKind k,
                     const std::optional<std::array<double, 3>>& seed)
{
    Edit tx(*this, "add object");
    dns::SceneObject o;
    o.kind = k;
    std::optional<std::array<double, 3>> useSeed = seed;
    if (!useSeed) {
        if (const SceneItem* p = find(primary()))
            useSeed = std::array<double, 3>{p->obj.x, p->obj.y, p->obj.z};
    }
    const Placement3 pl = proposePlacement(k, o.d, useSeed);
    o.x = pl.x;
    o.y = pl.y;
    o.z = pl.z;
    o.d = pl.d;
    o.label = uniqueName(dns::defaultSceneName(k, st.items.size()));
    if (pl.crowded)
        notice_ = "domain is full: placed overlapping, per-object force columns "
                  "will be suppressed";
    else if (pl.shrunk)
        notice_ = "no room at 1.0 D: placed at " + num(pl.d) + " D";
    const ObjId id = nextId_++;
    st.items.push_back({o, id});
    select(id);
    touch();
    return id;
}

void Doc::removeObject(ObjId id)
{
    const int i = indexOf(id);
    if (i < 0) return;
    Edit tx(*this, "remove object");
    st.items.erase(st.items.begin() + i);
    pruneSelection();
    touch();
}

void Doc::removeSelection()
{
    if (sel_.empty()) return;
    Edit tx(*this, "remove objects");
    const std::vector<ObjId> doomed = sel_;
    for (ObjId id : doomed) {
        const int i = indexOf(id);
        if (i >= 0) st.items.erase(st.items.begin() + i);
    }
    pruneSelection();
    touch();
}

ObjId Doc::duplicateObject(ObjId id)
{
    const SceneItem* src = find(id);
    if (!src) return kNoObj;
    Edit tx(*this, "duplicate object");
    dns::SceneObject o = src->obj;
    const Placement3 pl = proposePlacement(o.kind, o.d,
                                           std::array<double, 3>{o.x, o.y, o.z}, id);
    o.x = pl.x;
    o.y = pl.y;
    o.z = pl.z;
    o.d = pl.d;
    o.label = uniqueName(src->obj.label);
    const ObjId nid = nextId_++;
    st.items.push_back({o, nid});
    select(nid);
    touch();
    return nid;
}

bool Doc::renameObject(ObjId id, std::string_view desired)
{
    SceneItem* it = find(id);
    if (!it) return false;
    const std::string want = uniqueName(desired, id);
    if (want == it->obj.label) return true;
    Edit tx(*this, "rename object");
    it->obj.label = want;
    touch();
    return want == desired;
}

int Doc::arrayObjects(ObjId id, int count, double dx, double dy, double dz)
{
    const SceneItem* src = find(id);
    if (!src || count < 2) return 0;
    Edit tx(*this, "array objects");
    const dns::SceneObject base = src->obj;
    const Build& b = *current_;
    int made = 0;
    for (int i = 1; i < count; ++i) {
        dns::SceneObject o = base;
        o.x = base.x + dx * i;
        o.y = base.y + dy * i;
        o.z = base.z + dz * i;
        // A copy that would leave the domain is DROPPED, not forced: forcing
        // it would fail the whole build, which is the bug this replaces.
        if (b.ok && !dns::sceneBoxInside(b.cas().grid,
                                         dns::sceneObjectBox(b.cas().grid,
                                                             b.cas().placement, o)))
            continue;
        o.label = uniqueName(base.label);
        st.items.push_back({o, nextId_++});
        ++made;
    }
    if (made + 1 < count)
        notice_ = "array clipped to " + std::to_string(made + 1) + " of "
                  + std::to_string(count) + ": the rest would leave the domain";
    if (made) touch();
    return made;
}

void Doc::nudgeSelection(double dx, double dy, double dz)
{
    if (sel_.empty()) return;
    for (ObjId id : sel_)
        if (SceneItem* it = find(id)) {
            it->obj.x += dx;
            it->obj.y += dy;
            it->obj.z += dz;
        }
    touch();
}

void Doc::scaleSelection(double factor)
{
    if (sel_.empty() || !(factor > 0)) return;
    for (ObjId id : sel_)
        if (SceneItem* it = find(id)) it->obj.d = std::max(1e-6, it->obj.d * factor);
    touch();
}

// --- undo --------------------------------------------------------------------

void Doc::beginEdit(std::string_view label)
{
    if (editDepth_++ > 0) return;
    preEdit_ = {st, sel_, std::string(label)};
    preRev_ = revision_.load();
}

void Doc::endEdit()
{
    if (editDepth_ == 0 || --editDepth_ > 0) return;
    commitEdit();
}

// Commit the PRE-state, and only if the revision actually moved: touch() is
// post-hoc, so the snapshot taken at beginEdit is the one that can be undone
// back to. Using the revision counter avoids having to define equality over
// DocState.
void Doc::commitEdit()
{
    if (revision_.load() == preRev_) return;
    undo_.push_back(std::move(preEdit_));
    if (undo_.size() > kUndoMax) undo_.erase(undo_.begin());
    redo_.clear();
}

bool Doc::undo()
{
    if (undo_.empty()) return false;
    redo_.push_back({st, sel_, undo_.back().label});
    st = std::move(undo_.back().st);
    sel_ = std::move(undo_.back().sel);
    undo_.pop_back();
    pruneSelection();
    touch();
    return true;
}

bool Doc::redo()
{
    if (redo_.empty()) return false;
    undo_.push_back({st, sel_, redo_.back().label});
    st = std::move(redo_.back().st);
    sel_ = std::move(redo_.back().sel);
    redo_.pop_back();
    pruneSelection();
    touch();
    return true;
}

std::string_view Doc::undoLabel() const
{
    return undo_.empty() ? std::string_view{} : std::string_view(undo_.back().label);
}

// --- open --------------------------------------------------------------------

std::string Doc::openScene(const std::string& path)
{
    auto sc = dns::loadScene(path);
    if (!sc) return sc.error();
    Edit tx(*this, "open scene");
    st.items.clear();
    for (const auto& o : sc->objects) st.items.push_back({o, nextId_++});
    st.bodyKind = 8;
    if (sc->model == dns::SceneModel::Les) {
        st.les = sc->cs;
        st.sgs = sc->sgs == dns::SgsModel::Sigma ? 1 : 0;
    } else {
        st.les = 0;
    }
    clearSelection();
    if (!st.items.empty()) select(st.items.front().id);
    touch();
    return {};
}

std::string Doc::writeScene(const std::string& path) const
{
    // Pre-flight through the real parser: never write a file the CLI would
    // reject, and never print a --scene command that cannot run.
    const std::string s = sceneToml();
    if (auto chk = dns::parseScene(s); !chk) return chk.error();
    std::ofstream f(path, std::ios::binary);
    if (!f) return "cannot write " + path;
    f.write(s.data(), std::streamsize(s.size()));
    if (!f) return "write failed: " + path;
    return {};
}

// The command as ARGV, which is what actually gets executed. commandLine()
// joins this for display, so the printed command and the run cannot diverge --
// and a mesh path with a space in it survives, which a re-split string would
// not. `scenePath` is what --scene should point at (the file the editor wrote).
std::vector<std::string> Doc::commandArgs(const std::string& exe,
                                          const std::string& scenePath) const
{
    auto add2 = [](std::vector<std::string>& v, const char* flag,
                   const std::string& val) {
        v.push_back(flag);
        v.push_back(val);
    };
    if (st.isConv()) {
        std::vector<std::string> c{exe, "conv"};
        auto addc = [&](const char* f, const std::string& v) { add2(c, f, v); };
        addc("--physics", st.convPhysics == 1 ? "compressible"
                                              : "incompressible");
        addc("--n", std::to_string(st.n));
        addc("--ny", std::to_string(st.ny > 0 ? st.ny : st.n));
        addc("--nz", std::to_string(st.nz));
        addc("--aspect", num(st.convAspect));
        addc("--rayleigh", num(st.convRayleigh()));
        addc("--prandtl", num(st.prandtl));
        if (st.convPhysics == 1) {
            addc("--mach", num(st.convMach));
            addc("--dtemp", num(st.convDtemp()));
            if (st.visc == 1) {
                addc("--visc", "sutherland");
                addc("--tinf", num(st.convTmean));
            }
        }
        if (st.hotPatch < 1.0) addc("--hot-patch", num(st.hotPatch));
        if (st.convSides == 1) addc("--sides", "closed");
        addc("--kick", num(st.convKick));
        addc("--tt", num(st.convTt));
        addc("--wdt", num(st.wdt));
        return c;
    }

    std::vector<std::string> a{exe, "flow"};
    auto add = [&](const char* flag, const std::string& v) {
        add2(a, flag, v);
    };
    add("--n", std::to_string(st.n));
    if (st.ny > 0) add("--ny", std::to_string(st.ny));
    add("--nz", std::to_string(st.nz));
    add("--dr", num(st.dr));
    add("--re", num(st.re()));
    add("--ft", num(st.ft));
    // The live cadence. Always emitted: it is what decides the frame rate the
    // viewer sees, and a printed command that dropped it would not reproduce
    // the run that was watched. --wdt is the web/shm interval; --fdt (files) is
    // left alone because the editor's runs write no frames to disk.
    add("--wdt", num(st.wdt));
    if (st.physics == 1) {
        add("--physics", "compressible");
        add("--mach", num(st.mach()));
    }
    add("--far", kFarNames[st.far]);
    if (sceneMode()) add("--scene", scenePath);
    else add("--body", st.bodyKind == 7 ? st.meshPath : kBodyNames[st.bodyKind]);

    // Same capability gate as rawFrom(): the command that runs must be the
    // command that was inspected, so it cannot carry a flag the body does not
    // take.
    const BodyOpts bo = bodyOptions(st.bodyKind);
    if (bo.aspect) add("--aspect", num(st.aspect));
    if (bo.dimples && st.dimples > 0) add("--dimples", std::to_string(st.dimples));
    if (bo.bumps && st.bumps > 0) {
        add("--bumps", std::to_string(st.bumps));
        add("--bump-h", num(st.bumpH));
    }
    if (bo.mesh) {
        if (st.meshScale > 0) add("--scale", num(st.meshScale));
        if (st.robust) a.push_back("--robust");
    }
    if (st.les > 0 && !sceneMode()) {
        add("--les", num(st.les));
        add("--sgs", st.sgs == 1 ? "sigma" : "smag");
    }
    // Thermal, same gating as rawFrom(): only what is engaged is printed, so
    // the command reproduces the case that was inspected and nothing more.
    if (st.physics == 1) {
        add("--prandtl", num(st.prandtl));
        if (st.visc == 1) {
            add("--visc", "sutherland");
            add("--tinf", num(st.tinf));
        }
        if (st.wallIso) {
            add("--wall", "isothermal");
            add("--twall", num(st.twall));
        }
        if (st.nscbcTinflow) a.push_back("--nscbc-tinflow");
        if (st.tempFrames) a.push_back("--temp");
    }
    if (st.cutFlux) a.push_back("--cut-flux");
    if (st.penalize > 0) add("--penalize", num(st.penalize));
    if (st.physics == 1 && st.focalAuto) {
        add("--focal", "auto");
        add("--focal-ratio", num(st.focalRatio));
        add("--focal-blend", num(st.focalBlend));
    }
    return a;
}

std::string Doc::commandLine() const
{
    const auto a = commandArgs("build/dns", "scene.toml");
    std::string s;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (i) s += ' ';
        // Quote only what needs it: the command is meant to be pasted.
        s += a[i].find(' ') == std::string::npos ? a[i] : "'" + a[i] + "'";
    }
    return s;
}

} // namespace ui
