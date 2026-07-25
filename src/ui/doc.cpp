#include "ui/doc.hpp"

#include "dns/geometry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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
const char* const* objectKindNames() { return kObjectKinds; }
int objectKindCount() { return 5; }

bool Doc::sceneMode() const { return st.bodyKind == 8; }

Doc::Doc()
{
    current_ = std::make_shared<Build>();
    // One default object so the scene mode is never an empty screen.
    st.scene.objects.push_back(dns::SceneObject{});

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
static dns::RawConfig rawFrom(const DocState& s)
{
    dns::RawConfig r;
    r.sub = dns::Subcommand::Flow;
    r.out = "flow";
    r.n = s.n;
    r.nz = s.nz;
    r.dr = s.dr;
    r.re = s.re;
    r.ft = s.ft;
    r.mach = s.mach;
    r.physics = s.physics == 1 ? dns::Physics::Compressible
                               : dns::Physics::Incompressible;
    r.far = dns::FarField(s.far);
    r.cutFlux = s.cutFlux;
    r.penalize = s.penalize;
    r.sgs = s.sgs == 1 ? dns::SgsModel::Sigma : dns::SgsModel::Smagorinsky;
    r.aspect = s.aspect;
    r.dimples = s.dimples;

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
    if (s.dimples > 0) r.given.insert("--dimples");

    if (s.bodyKind == 8) {
        r.scene = "<editor>";   // recorded, never opened (phase 1.5 is ours)
        r.given.insert("--scene");
    } else {
        r.body = s.bodyKind == 7 ? s.meshPath : kBodyNames[s.bodyKind];
        r.given.insert("--body");
        if (s.bodyKind == 4) r.given.insert("--aspect");
    }

    if (s.les > 0 && s.bodyKind != 8) {
        r.les = s.les;
        r.given.insert("--les");
        r.given.insert("--sgs");
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
    dns::Scene sc = s.scene;
    if (scene) {
        // The scene, not --les, carries the model in scene mode: that is the
        // rule applySceneModel() enforces, and going through it means the
        // editor cannot produce a scene the CLI would reject as conflicting.
        if (s.les > 0) {
            sc.model = dns::SceneModel::Les;
            sc.cs = s.les;
            sc.sgs = s.sgs == 1 ? dns::SgsModel::Sigma : dns::SgsModel::Smagorinsky;
        } else {
            sc.model = dns::SceneModel::Dns;
            sc.cs = 0;
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

int Doc::addObject(dns::SceneObjectKind k)
{
    dns::SceneObject o;
    o.kind = k;
    // Offset each new object downstream so two objects are never coincident
    // (a union sdf of two identical bodies is indistinguishable from one).
    o.x = 1.5 * double(st.scene.objects.size());
    st.scene.objects.push_back(o);
    touch();
    return int(st.scene.objects.size()) - 1;
}

void Doc::removeObject(int index)
{
    if (index < 0 || index >= int(st.scene.objects.size())) return;
    st.scene.objects.erase(st.scene.objects.begin() + index);
    if (selected >= int(st.scene.objects.size())) selected = int(st.scene.objects.size()) - 1;
    touch();
}

std::string Doc::sceneToml() const
{
    std::string s = "# written by dns-ui\n";
    if (st.les > 0) {
        s += "model = \"les\"\n";
        s += "cs = " + num(st.les) + "\n";
        s += std::string("sgs = \"") + (st.sgs == 1 ? "sigma" : "smag") + "\"\n";
    } else {
        s += "model = \"dns\"\n";
    }
    for (const auto& o : st.scene.objects) {
        s += "\n[[objects]]\n";
        s += std::string("kind = \"") + dns::name(o.kind) + "\"\n";
        if (!o.label.empty()) s += "label = \"" + o.label + "\"\n";
        if (o.x != 0) s += "x = " + num(o.x) + "\n";
        if (o.y != 0) s += "y = " + num(o.y) + "\n";
        if (o.z != 0) s += "z = " + num(o.z) + "\n";
        s += "d = " + num(o.d) + "\n";
        if (o.kind == dns::SceneObjectKind::Lens)
            s += "aspect = " + num(o.aspect) + "\n";
        if (o.kind == dns::SceneObjectKind::Mesh) {
            s += "mesh = \"" + o.meshPath + "\"\n";
            if (o.scale > 0) s += "scale = " + num(o.scale) + "\n";
            if (o.robust) s += "robust = true\n";
        }
    }
    return s;
}

std::string Doc::writeScene(const std::string& path) const
{
    std::ofstream f(path, std::ios::binary);
    if (!f) return "cannot write " + path;
    const std::string s = sceneToml();
    f.write(s.data(), std::streamsize(s.size()));
    if (!f) return "write failed: " + path;
    return {};
}

std::string Doc::commandLine() const
{
    std::string s = "build/dns flow";
    s += " --n " + std::to_string(st.n);
    if (st.ny > 0) s += " --ny " + std::to_string(st.ny);
    s += " --nz " + std::to_string(st.nz);
    s += " --dr " + num(st.dr);
    s += " --re " + num(st.re);
    s += " --ft " + num(st.ft);
    if (st.physics == 1) {
        s += " --physics compressible --mach " + num(st.mach);
    }
    s += std::string(" --far ") + kFarNames[st.far];
    if (sceneMode()) s += " --scene scene.toml";
    else s += std::string(" --body ") + (st.bodyKind == 7 ? st.meshPath
                                                          : kBodyNames[st.bodyKind]);
    if (st.bodyKind == 4) s += " --aspect " + num(st.aspect);
    if (st.dimples > 0) s += " --dimples " + std::to_string(st.dimples);
    if (st.les > 0 && !sceneMode()) {
        s += " --les " + num(st.les);
        s += std::string(" --sgs ") + (st.sgs == 1 ? "sigma" : "smag");
    }
    if (st.cutFlux) s += " --cut-flux";
    if (st.penalize > 0) s += " --penalize " + num(st.penalize);
    if (st.physics == 1 && st.focalAuto) {
        s += " --focal auto --focal-ratio " + num(st.focalRatio);
        s += " --focal-blend " + num(st.focalBlend);
    }
    return s;
}

} // namespace ui
