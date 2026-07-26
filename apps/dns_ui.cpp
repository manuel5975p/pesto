// dns-ui: the interactive obstacle editor and grid viewer.
//
// The second binary in the tree, and the only exception to the one-binary rule
// in CLAUDE.md, because it solves nothing: it is a PRE-RUN inspector. Place
// bodies, choose a resolution, and see the cut cells the solver would actually
// receive -- built by the same validate() / buildCaseGeometry() / sanitizeCut()
// the runners call, so what it shows and what `dns flow` computes cannot
// disagree.
//
// Exit codes follow the same contract as `dns`: 0 ok, 5 no usable GPU (here,
// no Vulkan device or no surface), 3 input (no font).
#include "ui/app.hpp"
#include "ui/guicheck.hpp"
#include "ui/field.hpp"
#include "ui/gizmo.hpp"
#include "ui/pickobj.hpp"
#include "ui/livemesh.hpp"
#include "ui/runner.hpp"

#include "dns/shmframes.hpp"
#include "ui/view.hpp"

#include <atomic>
#include <csignal>
#include <fcntl.h>
#include <sys/mman.h>
#include <filesystem>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>

namespace {

// Headless self-check: everything below the window. Each case runs the
// document build (validate -> buildCaseGeometry -> sanitizeCut) and the view
// geometry (remapped grid lines, marching squares/tetrahedra, cut-cell boxes)
// and prints what came out. No Vulkan, no window, no font -- so it runs over
// ssh and as a ctest.
//
// It is a smoke test, not a validation tier: it asserts that each case builds
// and produces geometry, never that a coefficient lands in a band. Numbers
// that ARE pinned belong in tests/, against dns_core directly.
bool checkOne(const char* title, ui::DocState st, bool wantSurface)
{
    ui::Doc doc;
    doc.st = st;
    doc.touch();
    for (int i = 0; i < 4000 && !doc.current().ok && !doc.current().errorCode; ++i) {
        doc.poll();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const ui::Build& b = doc.current();
    std::printf("\n== %s\n", title);
    if (!b.ok) {
        std::printf("   FAILED (exit %d): %s\n", b.errorCode, b.error.c_str());
        return false;
    }
    const dns::Grid& g = b.cas().grid;
    std::printf("   %s  grid %d x %d x %d (%zu cells)  h %.5g\n", b.label.c_str(),
                g.nx, g.ny, g.nz, g.size(), g.hx);
    std::printf("   D %.5g  D/h %.4g  BL cells %.3g (%s)\n", b.geo.D, b.dOverH,
                b.blCells, b.bodyResolved ? "resolved" : "under-resolved");
    std::printf("   cut %zu, solid %zu, minLambda %.4g, minAperture %.4g\n", b.nCut,
                b.nSolid, b.minLambda, b.minAperture);
    std::printf("   sanitize: cells %zu, faces %zu, pockets %zu\n",
                b.sanitize.cellsSolidified, b.sanitize.facesClosed,
                b.sanitize.pocketsClosed);

    ui::ViewOptions o;
    o.showSolid = true;
    o.showApertures = true;
    if (g.nz > 1) o.sliceIndex = g.nz / 2;
    ui::ViewGeometry vg;
    buildViewGeometry(b, o, vg);
    std::printf("   view: %zu line, %zu surface, %zu cell verts (%zu cut, %zu solid"
                "%s)\n",
                vg.lines.size(), vg.surface.size(), vg.cells.size(),
                vg.stats.cutDrawn, vg.stats.solidDrawn,
                vg.stats.capped ? ", CAPPED" : "");

    bool ok = true;
    if (vg.lines.empty()) {
        std::printf("   FAILED: no grid geometry\n");
        ok = false;
    }
    if (b.nCut == 0 && st.bodyKind != 0) {
        std::printf("   FAILED: body has no cut cells\n");
        ok = false;
    }
    // In 2D the body outline is emitted as lines (marching squares), so only
    // the 3D cases are expected to produce surface triangles.
    if (wantSurface && vg.surface.empty()) {
        std::printf("   FAILED: no surface triangles\n");
        ok = false;
    }
    return ok;
}

// The SI <-> box mapping (ui/units.hpp). Pure arithmetic, so it is checked
// directly rather than inferred from a built case: if Re stops being
// uInf*D/nu the panels lie, and every downstream number lies with them.
bool checkUnits()
{
    int bad = 0;
    auto near = [&](const char* what, double got, double want, double tol) {
        const bool ok = std::abs(got - want) <= tol * std::abs(want);
        std::printf("   %-22s %-12.6g %s\n", what, got, ok ? "ok" : "FAILED");
        if (!ok) {
            std::printf("      wanted %.6g\n", want);
            ++bad;
        }
    };

    std::printf("\n== units\n");
    ui::Physical p;   // the default: 1 mm in air at 3 m/s
    near("Re", p.re(), 3.0 * 0.001 / 1.516e-5, 1e-12);
    near("Mach", p.mach(), 3.0 / 343.2, 1e-12);

    // Metres per box unit: the realised body is 0.7854 box units across in the
    // default cylinder case, and it IS the physical body, so 1 mm / 0.7854.
    near("m per box", p.metresPerBox(0.7854), 0.001 / 0.7854, 1e-12);
    near("m per box (0)", p.metresPerBox(0.0), 0.0, 1e-12);

    // A flow-through is L/uInf in whichever units L is expressed.
    near("flow-through", p.flowThroughSeconds(6.2832, 0.7854),
         6.2832 * (0.001 / 0.7854) / 3.0, 1e-12);

    // Switching material must move Re by exactly the viscosity ratio.
    ui::Physical w = p;
    const auto mats = ui::materials();
    w.nu = mats[1].nu;   // water
    w.c = mats[1].c;
    near("Re in water", w.re(), p.re() * (1.516e-5 / 1.004e-6), 1e-9);

    // Guard rails: a zero denominator must give 0, never NaN, because these
    // are formatted into panel text every frame.
    ui::Physical z;
    z.nu = 0;
    z.c = 0;
    near("Re (nu = 0)", z.re(), 0.0, 1e-12);
    near("Mach (c = 0)", z.mach(), 0.0, 1e-12);
    return bad == 0;
}


// Waits for the background build to settle, then reports it.
const ui::Build& settle(ui::Doc& doc)
{
    for (int i = 0; i < 4000; ++i) {
        doc.poll();
        if (doc.current().revision == doc.revision() - 0) {
        }
        if (!doc.building() && doc.current().ok) break;
        if (doc.current().errorCode) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return doc.current();
}

// Force a full rebuild and wait for it, whatever the outcome.
const ui::Build& rebuild(ui::Doc& doc)
{
    const uint64_t want = doc.revision();
    for (int i = 0; i < 4000; ++i) {
        doc.poll();
        if (doc.current().revision >= want && !doc.building()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return doc.current();
}

// The document layer: object placement, identity, naming and the scene round
// trip. Each block is a regression for a defect that shipped.
bool checkDoc()
{
    int bad = 0;
    auto ok = [&](bool cond, const char* what) {
        std::printf("   %-52s %s\n", what, cond ? "ok" : "FAILED");
        if (!cond) ++bad;
    };
    std::printf("\n== document\n");

    // 1. MANY OBJECTS. addObject used to place object i at x = 1.5*i, so with
    //    the default --dr 8 box the FIFTH object reached the domain boundary,
    //    buildSceneGeometry failed, and every bit of geometry vanished.
    {
        ui::Doc doc;
        doc.st.bodyKind = 8;
        doc.touch();
        rebuild(doc);
        bool everyBuildOk = true;
        int made = 0;
        for (int i = 0; i < 20; ++i) {
            const ui::ObjId id = doc.addObject(dns::SceneObjectKind::Cylinder);
            const ui::Build& b = rebuild(doc);
            if (!b.ok) {
                everyBuildOk = false;
                std::printf("      object %d: %s\n", i + 2, b.error.c_str());
                break;
            }
            if (id != ui::kNoObj) ++made;
        }
        ok(everyBuildOk, "20 added objects: every build stays valid");
        ok(made == 20, "20 objects were actually created");

        // Pairwise disjoint, so per-object force columns survive.
        const ui::Build& b = doc.current();
        bool disjoint = true;
        if (b.ok) {
            const auto& items = doc.st.items;
            for (size_t i = 0; i < items.size() && disjoint; ++i)
                for (size_t j = i + 1; j < items.size() && disjoint; ++j) {
                    const auto A = dns::sceneObjectBox(b.cas().grid,
                                                       b.cas().placement,
                                                       items[i].obj);
                    const auto B = dns::sceneObjectBox(b.cas().grid,
                                                       b.cas().placement,
                                                       items[j].obj);
                    if (!(A[3] < B[0] || A[0] > B[3] || A[4] < B[1] ||
                          A[1] > B[4] || A[5] < B[2] || A[2] > B[5]))
                        disjoint = false;
                }
        }
        ok(disjoint, "placed objects do not overlap each other");
    }

    // 2. CROWDING. Far more objects than fit: the build must still never fail.
    {
        ui::Doc doc;
        doc.st.bodyKind = 8;
        doc.touch();
        rebuild(doc);
        bool everyOk = true;
        for (int i = 0; i < 40; ++i) {
            doc.addObject(dns::SceneObjectKind::Cylinder);
            if (!rebuild(doc).ok) { everyOk = false; break; }
        }
        ok(everyOk, "40 objects in a full domain still build");
    }

    // 3. IDENTITY. Deleting a lower-indexed object used to silently retarget
    //    the selection to a different body.
    {
        ui::Doc doc;
        doc.st.bodyKind = 8;
        doc.touch();
        rebuild(doc);
        const ui::ObjId a = doc.addObject(dns::SceneObjectKind::Cylinder);
        const ui::ObjId c = doc.addObject(dns::SceneObjectKind::Sphere);
        rebuild(doc);
        doc.select(c);
        doc.removeObject(a);
        ok(doc.primary() == c, "removing a lower object keeps the selection");
        ok(doc.find(a) == nullptr, "the removed object is gone");
        ok(doc.find(c) != nullptr, "the selected object survives");
    }

    // 4. NAMES. Every object gets a unique, parser-legal name.
    {
        ui::Doc doc;
        doc.st.bodyKind = 8;
        doc.touch();
        rebuild(doc);
        for (int i = 0; i < 5; ++i) doc.addObject(dns::SceneObjectKind::Cylinder);
        ok(dns::checkSceneNames(ui::sceneOf(doc.st)).has_value(),
           "auto-named objects satisfy dns::checkSceneNames");
        const ui::ObjId id = doc.st.items.back().id;
        doc.renameObject(id, "a b/c");
        ok(dns::validSceneName(doc.find(id)->obj.label),
           "an illegal rename is sanitized to a legal name");
        const std::string first = doc.st.items.front().obj.label;
        doc.renameObject(id, first);
        ok(doc.find(id)->obj.label != first,
           "renaming onto a taken name is disambiguated");
    }

    // 5. ROUND TRIP. sceneToml() used to emit `model = "dns"` (a scalar, not a
    //    [model] table), `label` (the key is `name`) and `mesh` (it is `path`),
    //    so the written file plus the printed --scene command could not run.
    {
        ui::Doc doc;
        doc.st.bodyKind = 8;
        doc.st.items.clear();
        auto add = [&](dns::SceneObjectKind k, const char* nm, double x, double d) {
            dns::SceneObject o;
            o.kind = k;
            o.label = nm;
            o.x = x;
            o.d = d;
            doc.st.items.push_back({o, ui::ObjId(doc.st.items.size() + 1)});
            return &doc.st.items.back().obj;
        };
        add(dns::SceneObjectKind::Cylinder, "front", -1.25, 0.8);
        add(dns::SceneObjectKind::Lens, "wing-2", 0.5, 1.0)->aspect = 3.25;
        auto* m = add(dns::SceneObjectKind::Mesh, "hull", 2.0, 1.5);
        m->meshPath = "body.stl";
        m->scale = 2.5;
        m->robust = true;
        doc.st.les = 1.35;
        doc.st.sgs = 1;   // sigma

        const std::string toml = doc.sceneToml();
        auto parsed = dns::parseScene(toml);
        ok(parsed.has_value(), "sceneToml() parses with dns::parseScene");
        if (parsed) {
            bool same = parsed->objects.size() == doc.st.items.size() &&
                        parsed->model == dns::SceneModel::Les &&
                        parsed->cs == 1.35 &&
                        parsed->sgs == dns::SgsModel::Sigma;
            for (size_t i = 0; same && i < parsed->objects.size(); ++i) {
                const auto& a = parsed->objects[i];
                const auto& b = doc.st.items[i].obj;
                same = a.kind == b.kind && a.label == b.label && a.x == b.x &&
                       a.y == b.y && a.z == b.z && a.d == b.d &&
                       a.aspect == b.aspect && a.meshPath == b.meshPath &&
                       a.scale == b.scale && a.robust == b.robust;
            }
            ok(same, "every field survives the round trip bit-exactly");
        }
        // Idempotence: re-emitting a parsed scene gives the same bytes.
        if (parsed) {
            ui::DocState st2 = doc.st;
            st2.items.clear();
            for (const auto& o : parsed->objects)
                st2.items.push_back({o, ui::ObjId(st2.items.size() + 1)});
            ok(ui::sceneTomlOf(st2) == toml, "sceneToml is idempotent");
        }
    }

    // 6. EMPTY SCENE is a state you must be able to LEAVE. Deleting the last
    //    object used to fail the whole build, which lost the grid, the domain
    //    box and -- because click-to-place was gated on a valid build -- any
    //    way to add an object back. It is now a warned, recoverable state.
    {
        ui::Doc doc;
        doc.st.bodyKind = 8;
        doc.st.items.clear();
        doc.touch();
        const ui::Build& b = rebuild(doc);
        ok(b.ok, "an empty scene still builds (the grid stays visible)");
        ok(b.sceneEmpty, "an empty scene is flagged");
        ok(b.nCut == 0 && b.nSolid == 0, "an empty scene has no solid cells");
        // ...but it must NOT be exportable: the CLI rejects it with exit 3.
        const std::string err = doc.writeScene("/dev/null");
        ok(!err.empty() && err.find("at least one") != std::string::npos,
           "writing an empty scene is refused with the parser's message");
        // And the way out works.
        const ui::ObjId id = doc.addObject(dns::SceneObjectKind::Cylinder);
        const ui::Build& b2 = rebuild(doc);
        ok(id != ui::kNoObj && b2.ok && !b2.sceneEmpty,
           "adding an object recovers from the empty scene");
        ok(doc.writeScene("/dev/null").empty(),
           "the recovered scene is exportable again");
    }

    // 7. UNDO. A gesture is one entry, and undo restores byte-identical TOML.
    {
        ui::Doc doc;
        doc.st.bodyKind = 8;
        doc.touch();
        rebuild(doc);
        const std::string before = doc.sceneToml();
        for (int i = 0; i < 6; ++i) doc.addObject(dns::SceneObjectKind::Cube);
        const std::string after = doc.sceneToml();
        ok(after != before, "adding objects changed the document");
        int undone = 0;
        while (doc.canUndo() && undone < 50) { doc.undo(); ++undone; }
        ok(doc.sceneToml() == before, "undo restores the document exactly");
        ok(undone == 6, "one undo entry per add, not per touch");
        while (doc.canRedo()) doc.redo();
        ok(doc.sceneToml() == after, "redo restores the edited document");
    }

    // 8. ARRAY + DUPLICATE.
    {
        ui::Doc doc;
        doc.st.bodyKind = 8;
        doc.touch();
        rebuild(doc);
        const ui::ObjId id = doc.st.items.front().id;
        const int made = doc.arrayObjects(id, 3, 0.0, 1.5, 0.0);
        ok(made == 2, "array of 3 makes 2 further copies");
        ok(rebuild(doc).ok, "the arrayed scene builds");
        ok(dns::checkSceneNames(ui::sceneOf(doc.st)).has_value(),
           "arrayed copies get unique names");
        const size_t n = doc.st.items.size();
        doc.duplicateObject(id);
        ok(doc.st.items.size() == n + 1 && rebuild(doc).ok,
           "duplicate lands somewhere the build accepts");
    }

    // 9. OPEN WHAT THE REPO SHIPS. The editor could previously only ever
    //    start from scratch; these are the scenes it must be able to edit.
#ifdef DNS_EXAMPLES_DIR
    {
        static const char* const kExamples[] = {
            "tandem_cylinders.toml", "cylinder_array_les.toml",
            "mixed_shapes.toml",     "sphere_pair_3d.toml"};
        for (const char* nm : kExamples) {
            ui::Doc doc;
            const std::string path = std::string(DNS_EXAMPLES_DIR) + "/" + nm;
            const std::string err = doc.openScene(path);
            const bool opened = err.empty() && !doc.st.items.empty();
            ok(opened, (std::string("opens examples/") + nm).c_str());
            if (!opened) {
                std::printf("      %s\n", err.c_str());
                continue;
            }
            // And re-emitting it must still be something the parser accepts.
            ok(dns::parseScene(doc.sceneToml()).has_value(),
               (std::string("re-emits examples/") + nm).c_str());
        }
    }
#endif

    // 9. OPEN WHAT THE REPO SHIPS. The editor could previously only ever
    //    start from scratch; these are the scenes it must be able to edit.
#ifdef DNS_EXAMPLES_DIR
    {
        static const char* const kExamples[] = {
            "tandem_cylinders.toml", "cylinder_array_les.toml",
            "mixed_shapes.toml",     "sphere_pair_3d.toml"};
        for (const char* nm : kExamples) {
            ui::Doc doc;
            const std::string path = std::string(DNS_EXAMPLES_DIR) + "/" + nm;
            const std::string err = doc.openScene(path);
            const bool opened = err.empty() && !doc.st.items.empty();
            ok(opened, (std::string("opens examples/") + nm).c_str());
            if (!opened) {
                std::printf("      %s\n", err.c_str());
                continue;
            }
            // And re-emitting it must still be something the parser accepts.
            ok(dns::parseScene(doc.sceneToml()).has_value(),
               (std::string("re-emits examples/") + nm).c_str());
        }
    }
#endif

    return bad == 0;
}

// The run button spawns `dns` as a CHILD PROCESS -- dns-ui links dns_core only
// and solves nothing itself. What is checked here is the process plumbing:
// output capture, exit codes, and that a run can be stopped.
bool checkRunner()
{
    int bad = 0;
    auto ok = [&](bool cond, const char* what) {
        std::printf("   %-52s %s\n", what, cond ? "ok" : "FAILED");
        if (!cond) ++bad;
    };
    std::printf("\n== runner\n");

    auto wait = [](ui::Runner& r, int maxMs) {
        for (int i = 0; i < maxMs / 5 && r.running(); ++i) {
            r.poll();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        r.poll();
    };

    {   // stdout and stderr are both captured, in order, and exit codes arrive.
        ui::Runner r;
        ok(r.start({"/bin/sh", "-c", "echo one; echo two 1>&2; exit 7"}, ".").empty(),
           "starts a child process");
        wait(r, 4000);
        const auto lines = r.lines();
        ok(!r.running(), "the run finishes");
        ok(r.exitCode() == 7, "the child's exit code is reported");
        bool sawOne = false, sawTwo = false;
        for (const auto& l : lines) {
            if (l == "one") sawOne = true;
            if (l == "two") sawTwo = true;
        }
        ok(sawOne && sawTwo, "stdout AND stderr are captured");
        ok(r.seconds() >= 0, "the elapsed time is measured");
    }

    {   // A run that would never end must be stoppable.
        ui::Runner r;
        ok(r.start({"/bin/sh", "-c", "while true; do sleep 1; done"}, ".").empty(),
           "starts a long run");
        ok(r.running(), "the long run is in flight");
        r.stop();
        ok(!r.running(), "stop() ends it");
    }

    {   // A command that does not exist must fail as a run, not a crash.
        ui::Runner r;
        r.start({"/nonexistent/dns", "flow"}, ".");
        wait(r, 4000);
        ok(!r.running() && r.exitCode() == 127,
           "a missing binary exits 127 rather than hanging");
    }

    {   // Two runs at once are refused rather than racing.
        ui::Runner r;
        r.start({"/bin/sh", "-c", "sleep 2"}, ".");
        const std::string err = r.start({"/bin/sh", "-c", "true"}, ".");
        ok(!err.empty(), "a second concurrent run is refused");
        r.stop();
    }

    return bad == 0;
}

// The live bridge: solver -> shared memory -> viewer, with no disk between.
// Publisher and reader are exercised against each other here, which is the only
// place the seqlock and the byte layout are checked at all.
bool checkLive()
{
    int bad = 0;
    auto ok = [&](bool cond, const char* what) {
        std::printf("   %-52s %s\n", what, cond ? "ok" : "FAILED");
        if (!cond) ++bad;
    };
    std::printf("\n== live bridge\n");

    const int W = 12, H = 8, D = 5;
    std::vector<dns::ShmField> fields(2);
    std::snprintf(fields[0].name, sizeof fields[0].name, "vort");
    fields[0].lo = -2;
    fields[0].hi = 2;
    fields[0].diverging = 1;
    std::snprintf(fields[1].name, sizeof fields[1].name, "speed");
    fields[1].lo = 0;
    fields[1].hi = 1;

    std::vector<dns::ShmFrames::PlaneSpec> specs{{2, 0.0f, W, H}, {1, 0.0f, W, D}};

    dns::ShmFrames pub;
    const std::string name = "dns_uicheck";
    const auto made = pub.create(name, fields, specs, W, H, D);
    ok(made.has_value(), "the publisher creates a segment");
    if (!made) {
        std::printf("      %s\n", made.error().c_str());
        return false;
    }
    pub.setGrid(W + 2, H + 2, D + 2, 1, 1, 1, W, H, D, 0.25, 0.5, 2.0, 0, 0, 0, 1);

    ui::LiveField rd;
    ok(rd.attach(name).empty(), "the reader attaches to it");
    ok(rd.attached() && rd.fields().size() == 2, "both fields are visible");
    ok(rd.running(), "the run reads as live");

    // Nothing published yet.
    ok(!rd.poll(), "no frame before the first publish");

    // Publish a frame whose values encode their own index, so a wrong stride or
    // a wrong plane shows up as a wrong number rather than as a blur.
    std::vector<float> xy0(std::size_t(W) * H), xy1(std::size_t(W) * H);
    std::vector<float> xz0(std::size_t(W) * D), xz1(std::size_t(W) * D);
    std::vector<float> mask(std::size_t(W) * H, 1.0f);
    std::vector<float> vol(std::size_t(W) * H * D);
    for (std::size_t i = 0; i < xy0.size(); ++i) {
        xy0[i] = float(i);
        xy1[i] = -float(i);
    }
    for (std::size_t i = 0; i < xz0.size(); ++i) xz0[i] = float(i) * 2.0f;
    for (std::size_t i = 0; i < vol.size(); ++i) vol[i] = float(i % 7);
    mask[3] = 0.0f;   // one solid cell

    pub.publish(1.25, {{xy0.data(), xy1.data()}, {xz0.data(), xz1.data()}},
                {mask.data(), nullptr}, {vol.data(), vol.data()});

    rd.wantVolume(true);
    ok(rd.poll(), "the reader sees the published frame");
    ok(rd.frame() == 1 && std::abs(rd.time() - 1.25) < 1e-9,
       "frame counter and time arrive");
    ok(rd.planes().size() == 2, "both planes arrive");
    if (rd.planes().size() == 2) {
        const auto& p0 = rd.planes()[0];
        ok(p0.axis == 2 && p0.w == W && p0.h == H, "plane 0 keeps its axis and size");
        ok(p0.value.size() == std::size_t(W) * H && p0.value[5] == 5.0f,
           "plane 0 carries the selected field's values");
        ok(p0.mask.size() == p0.value.size() && p0.mask[3] == 0.0f,
           "the body mask arrives with the plane");
    }
    ok(rd.volume().size() == std::size_t(W) * H * D, "the 3D volume arrives");
    ok(rd.i0() == 1 && rd.j0() == 1 && rd.k0() == 1,
       "the window origin arrives (planes would be a ghost cell off without it)");

    // Selecting the second field must re-read the SAME frame, not wait.
    rd.select(1);
    ok(rd.poll(), "selecting another field re-reads the frame");
    ok(!rd.planes().empty() && rd.planes()[0].value[5] == -5.0f,
       "the second field's values are distinct");

    // A second publish bumps the counter.
    pub.publish(2.5, {{xy0.data(), xy1.data()}, {xz0.data(), xz1.data()}},
                {mask.data(), nullptr}, {vol.data(), vol.data()});
    ok(rd.poll() && rd.frame() == 2, "a second frame is picked up");
    ok(!rd.poll(), "polling again with no new frame is a no-op");

    // THE SEQLOCK, under real contention. A publisher thread writes frames in
    // which every sample equals the frame number; the reader polls. If the lock
    // were wrong, a copy would mix two frames and the plane would not be
    // uniform -- which is precisely the corruption a mutex-free bridge risks.
    {
        dns::ShmFrames cpub;
        const std::string cname = "dns_uicheck_seq";
        if (cpub.create(cname, fields, {{2, 0.0f, W, H}}, 0, 0, 0)) {
            ui::LiveField crd;
            if (crd.attach(cname).empty()) {
                std::atomic<bool> stop{false};
                std::atomic<int> written{0};
                std::thread writer([&] {
                    std::vector<float> buf(std::size_t(W) * H);
                    for (int f = 1; f <= 2000 && !stop; ++f) {
                        std::fill(buf.begin(), buf.end(), float(f));
                        cpub.publish(double(f), {{buf.data(), buf.data()}},
                                     {nullptr}, {});
                        written = f;
                    }
                    stop = true;
                });
                int seen = 0, torn = 0;
                for (int i = 0; i < 200000 && !stop; ++i)
                    if (crd.poll() && !crd.planes().empty()) {
                        const auto& v = crd.planes()[0].value;
                        const float first = v.empty() ? 0.0f : v[0];
                        for (float x : v)
                            if (x != first) {
                                ++torn;
                                break;
                            }
                        ++seen;
                    }
                writer.join();
                ok(torn == 0, "no torn frame under a concurrent writer");
                ok(seen > 0, "the reader saw frames while the writer ran");
            }
        }
    }

    pub.finish();
    ok(!rd.running(), "finishing the run is visible to the reader");
    ok(rd.publisherPid() == int(getpid()), "the publisher's pid is readable");

    // The mesh builders: pure CPU, so they are checked here too.
    ui::LiveMeshOptions o;
    o.showPlane = true;
    o.plane = 2;
    std::vector<ui::Vtx3> verts;
    ui::LiveMeshStats st;
    ui::buildLivePlane(rd.view(), o, true, -2, 2, verts, st);
    ok(st.quads > 0 && verts.size() == st.quads * 6,
       "the cut plane builds six vertices per quad");
    const std::size_t planeVerts = verts.size();
    o.showFog = true;
    o.fogFloor = 0.0f;
    ui::LiveMeshStats fst;
    ui::buildLiveFog(rd.view(), o, true, -2, 2, ui::Vec3{0, 0, 1}, verts, fst);
    ok(fst.slices > 0 && verts.size() > planeVerts, "the fog builds slices");

    // Slicing an OFF-CENTRE plane out of the volume. The volume was filled with
    // (index % 7), so a wrong slice index or a wrong axis shows up as the wrong
    // number rather than as a plausible-looking picture.
    {
        // A volume whose value encodes its k index, so the slice is checkable.
        std::vector<float> kvol(std::size_t(W) * H * D);
        for (int k = 0; k < D; ++k)
            for (int j = 0; j < H; ++j)
                for (int i = 0; i < W; ++i)
                    kvol[std::size_t(k) * W * H + std::size_t(j) * W + i] = float(k);
        pub.publish(9.0, {{xy0.data(), xy1.data()}, {xz0.data(), xz1.data()}},
                    {mask.data(), nullptr}, {kvol.data(), kvol.data()});
        rd.select(0);
        rd.wantVolume(true);
        ok(rd.poll(), "a frame with a per-field volume arrives");
        ok(rd.hasVolume(), "the run reports a published volume");

        ui::LiveMeshOptions so;
        so.showPlane = true;
        so.plane = 2;            // xy: slicing moves along k
        so.hideSolid = false;
        // The colour of a quad is a pure function of the sliced value, so
        // comparing two slices' meshes compares the slices.
        auto meshAt = [&](float at) {
            so.planeAt = at;
            std::vector<ui::Vtx3> v;
            ui::LiveMeshStats s3;
            ui::buildLivePlane(rd.view(), so, false, 0, float(D - 1), v, s3);
            return std::pair{v, s3};
        };
        auto [v0, s0] = meshAt(0.0f);
        auto [v1, s1] = meshAt(1.0f);
        ok(s0.slicedFromVolume && s1.slicedFromVolume,
           "an off-centre plane is sliced from the volume");
        ok(s0.sliceIndex == 0 && s1.sliceIndex == D - 1,
           "planeAt 0 and 1 hit the first and last slice");
        ok(!v0.empty() && !v1.empty() && v0[0].c.r != v1[0].c.r,
           "the two slices carry different data");
        // k = 0 is the bottom of the range, k = D-1 the top.
        ok(v0[0].c.r < v1[0].c.r, "the slice index maps to the field value");
        // And the plane MOVES: its z coordinate must differ.
        ok(v0[0].p[2] != v1[0].p[2], "the plane is drawn where it was sliced");

        // --- the fog stack ----------------------------------------------
        // Alpha compositing is only correct BACK TO FRONT, and the volume
        // encodes its own k index, so "which slice came out first" is a
        // readable number rather than a judgement about a picture. Drawn the
        // other way round the fog reads inside-out, which is what "the fog
        // looks off" turned out to be.
        {
            bool asc = false;
            const int axis = ui::fogOrder(ui::Vec3{0, 0, 1}, asc);
            ok(axis == 2 && !asc,
               "looking down +z slices along z, from the far end");
            bool asc2 = false;
            ok(ui::fogOrder(ui::Vec3{0, 0, -1}, asc2) == 2 && asc2,
               "looking down -z reverses the traversal");

            ui::LiveMeshOptions fo;
            fo.showPlane = false;
            fo.showFog = true;
            fo.fogFloor = 0.0f;
            fo.fogSlices = D;
            fo.hideSolid = false;
            auto fogWith = [&](const ui::Vec3& dir, int slices) {
                fo.fogSlices = slices;
                std::vector<ui::Vtx3> fv;
                ui::LiveMeshStats fs;
                ui::buildLiveFog(rd.view(), fo, false, 0, float(D - 1), dir, fv, fs);
                return std::pair{fv, fs};
            };
            auto [fwd, fwdS] = fogWith(ui::Vec3{0, 0, 1}, D);
            auto [bwd, bwdS] = fogWith(ui::Vec3{0, 0, -1}, D);
            ok(!fwd.empty() && !bwd.empty(), "the fog builds both ways");
            // The first vertex emitted belongs to the first slice drawn, and
            // that slice must be the FURTHEST one along the view direction.
            ok(fwd.front().p[2] > fwd.back().p[2],
               "+z view draws the far slice first");
            ok(bwd.front().p[2] < bwd.back().p[2],
               "-z view draws the far slice first");
            ok(fwdS.fogAxis == 2 && fwdS.fogAscending == false &&
                   bwdS.fogAscending == true,
               "the mesh records the octant it was built for");

            // Thinning the stack must not thin the FOG: one slice standing in
            // for `step` cells carries the opacity those cells would have.
            auto meanAlpha = [](const std::vector<ui::Vtx3>& v) {
                double acc = 0;
                for (const auto& x : v) acc += x.c.a;
                return v.empty() ? 0.0 : acc / double(v.size());
            };
            auto [few, fewS] = fogWith(ui::Vec3{0, 0, 1}, D / 4);
            ok(fewS.fogStep > fwdS.fogStep, "fewer slices means a coarser step");
            ok(meanAlpha(few) > meanAlpha(fwd) * 1.5,
               "a thinned stack compensates its per-slice opacity");

            // A HARD threshold draws its own contour: the fog comes out as
            // nested shells. The soft ramp is what makes it fog, and it shows
            // up as alphas strictly between nothing and the full value.
            fo.fogSlices = D;
            fo.fogFloor = 0.3f;
            fo.fogSoft = 0.3f;
            std::vector<ui::Vtx3> soft;
            ui::LiveMeshStats softS;
            ui::buildLiveFog(rd.view(), fo, false, 0, float(D - 1),
                             ui::Vec3{0, 0, 1}, soft, softS);
            int partial = 0;
            unsigned char hi8 = 0;
            for (const auto& x : soft) hi8 = std::max(hi8, x.c.a);
            for (const auto& x : soft)
                if (x.c.a > 0 && x.c.a < hi8) ++partial;
            ok(partial > 0, "the threshold ramps instead of cutting");
        }
    }

    // Relative colouring: a magnitude re-centred on a reference speed, so the
    // picture shows slower-than-tunnel vs faster-than-tunnel rather than "how
    // fast". This is a pure mapping question, so it is checked directly.
    {
        // A plane whose speed is BELOW the reference on one half and above on
        // the other, with the mean exactly at the reference.
        const int PW = 8, PH = 4;
        std::vector<float> slow(std::size_t(PW) * PH);
        for (int j = 0; j < PH; ++j)
            for (int i = 0; i < PW; ++i)
                slow[std::size_t(j) * PW + i] = (i < PW / 2) ? 0.5f : 1.5f;
        std::vector<float> m2(slow.size(), 1.0f);
        std::vector<dns::ShmFrames::PlaneSpec> sp2{{2, 0.0f, PW, PH}};
        dns::ShmFrames p2;
        if (p2.create("dns_uirel", fields, sp2, 0, 0, 0)) {
            p2.setGrid(PW + 2, PH + 2, 3, 1, 1, 1, PW, PH, 1, 0.1, 0.1, 0.1, 0, 0,
                       0, 1);
            p2.publish(1.0, {{slow.data(), slow.data()}}, {m2.data()}, {});
            ui::LiveField r2;
            if (r2.attach("dns_uirel").empty() && r2.poll()) {
                ui::LiveMeshOptions ro;
                ro.showPlane = true;
                ro.hideSolid = false;
                ro.refMode = ui::RefMode::Mean;
                std::vector<ui::Vtx3> rv;
                ui::LiveMeshStats rst;
                ui::buildLivePlane(r2.view(), ro, false, 0, 1.6f, rv, rst);
                ok(std::abs(rst.refUsed - 1.0f) < 1e-5f,
                   "the mean reference is measured from the fluid cells");
                ok(rst.quads == std::size_t(PW) * PH, "every cell is drawn");
                // The slow half must be BLUE (b > r), the fast half RED.
                bool blueSlow = false, redFast = false;
                for (std::size_t q = 0; q + 5 < rv.size(); q += 6) {
                    const float x = rv[q].p[0];
                    const auto& c = rv[q].c;
                    if (x < 0.1f + PW * 0.05f) blueSlow = blueSlow || c.b > c.r;
                    else redFast = redFast || c.r > c.b;
                }
                ok(blueSlow, "slower than the reference reads blue");
                ok(redFast, "faster than the reference reads red");

                // An explicit tunnel reference shifts the split.
                ro.refMode = ui::RefMode::Freestream;
                ro.refValue = 0.5f;   // now NOTHING is slower than the reference
                rv.clear();
                ui::LiveMeshStats rst2;
                ui::buildLivePlane(r2.view(), ro, false, 0, 1.6f, rv, rst2);
                ok(std::abs(rst2.refUsed - 0.5f) < 1e-6f,
                   "the tunnel reference is used verbatim");
                bool anyBelow = false;
                for (std::size_t q = 0; q + 5 < rv.size(); q += 6)
                    if (rv[q].c.b > rv[q].c.r + 20) anyBelow = true;
                ok(!anyBelow, "nothing reads blue when nothing is below the reference");

                // Contrast saturates sooner without changing the reference.
                ro.refMode = ui::RefMode::Mean;
                ro.contrast = 4.0f;
                rv.clear();
                ui::LiveMeshStats rst3;
                ui::buildLivePlane(r2.view(), ro, false, 0, 1.6f, rv, rst3);
                ok(rst3.spanUsed < rst.spanUsed && std::abs(rst3.refUsed - 1.0f) < 1e-5f,
                   "contrast narrows the span, not the reference");

                // A uniform field must not explode into noise.
                std::fill(slow.begin(), slow.end(), 0.7f);
                p2.publish(2.0, {{slow.data(), slow.data()}}, {m2.data()}, {});
                (void)r2.poll();
                rv.clear();
                ui::LiveMeshStats rst4;
                ro.contrast = 1.0f;
                ui::buildLivePlane(r2.view(), ro, false, 0, 1.6f, rv, rst4);
                ok(rst4.spanUsed > 0 && !rv.empty(),
                   "a uniform field does not divide by zero");
            }
        }
    }

    // Rewind. Every decoded frame is kept as a snapshot of its PLANES, so a run
    // can be scrubbed back through while it is still going. Volumes are
    // deliberately not kept -- one is 512 MB -- so a rewound view must report no
    // volume rather than quietly reusing the live one.
    {
        const int PW = 6, PH = 4;
        std::vector<dns::ShmFrames::PlaneSpec> sp{{2, 0.0f, PW, PH}};
        dns::ShmFrames hp;
        if (hp.create("dns_uihist", fields, sp, 0, 0, 0)) {
            hp.setGrid(PW + 2, PH + 2, 3, 1, 1, 1, PW, PH, 1, 0.1, 0.1, 0.1, 0, 0,
                       0, 1);
            ui::LiveField hr;
            if (hr.attach("dns_uihist").empty()) {
                std::vector<float> buf(std::size_t(PW) * PH);
                for (int f = 1; f <= 5; ++f) {
                    std::fill(buf.begin(), buf.end(), float(f));
                    hp.publish(double(f), {{buf.data(), buf.data()}}, {nullptr}, {});
                    ok(hr.poll(), "the reader takes the frame");
                }
                ok(hr.history().size() == 5, "every decoded frame is kept");
                ok(hr.history().back().frame == 5 && hr.history().front().frame == 1,
                   "the timeline runs oldest to newest");
                // A rewound view is the OLD data, at the same grid geometry.
                const ui::LiveView old = hr.viewOf(hr.history()[1]);
                ok(old.planes && !old.planes->empty() &&
                       old.planes->front().value[0] == 2.0f,
                   "rewinding reads the frame it was parked on");
                ok(!old.hasVolume(), "a rewound frame carries no volume");
                ok(old.hx == hr.view().hx && old.i0 == hr.view().i0,
                   "rewinding keeps the grid geometry");

                // The budget is a hard ceiling: the oldest frames go first, and
                // the newest is never evicted.
                hr.setHistoryBudget(1);
                ok(hr.history().size() == 1, "the budget evicts from the front");
                ok(hr.history().back().frame == 5, "the newest frame survives");

                // Switching the coloured field re-decodes the same frames as a
                // different quantity: two series must not be spliced.
                hr.setHistoryBudget(64u << 20);
                std::fill(buf.begin(), buf.end(), 9.0f);
                hp.publish(6.0, {{buf.data(), buf.data()}}, {nullptr}, {});
                (void)hr.poll();
                hr.select(1);
                (void)hr.poll();
                ok(hr.history().size() == 1,
                   "changing the field restarts the timeline");
            }
        }
    }

    // hide-solid must drop exactly the masked cells, and nothing else.
    o.showFog = false;
    o.hideSolid = false;
    verts.clear();
    ui::LiveMeshStats sAll;
    ui::buildLivePlane(rd.view(), o, true, -2, 2, verts, sAll);
    o.hideSolid = true;
    verts.clear();
    ui::LiveMeshStats sCut;
    ui::buildLivePlane(rd.view(), o, true, -2, 2, verts, sCut);
    ok(sAll.quads == std::size_t(W) * H, "every cell becomes a quad");
    ok(sCut.quads == sAll.quads - 1, "hide-solid drops exactly the masked cell");

    return bad == 0;
}

// The manipulation layer. All pure functions, so the part that actually
// matters -- that the grabbed point stays under the cursor -- is testable
// without a window.
bool checkGizmo()
{
    int bad = 0;
    auto ok = [&](bool cond, const char* what) {
        std::printf("   %-52s %s\n", what, cond ? "ok" : "FAILED");
        if (!cond) ++bad;
    };
    std::printf("\n== manipulation\n");

    const float vpW = 900, vpH = 600;
    auto mkCam = [&](bool ortho) {
        ui::Camera c;
        c.target = {2.0f, 3.0f, 1.0f};
        c.dist = 6.0f;
        c.yaw = -0.9f;
        c.pitch = 0.5f;
        c.ortho = ortho;
        return c;
    };

    // 1. worldToScreen must be the exact inverse of Camera::ray. This pins the
    //    NDC / y-flip convention; get it wrong and every gizmo hit test is
    //    silently mirrored vertically.
    for (int mode = 0; mode < 2; ++mode) {
        const ui::Camera c = mkCam(mode == 1);
        const ui::Mat4 mvp = c.viewProj(vpW / vpH);
        float worst = 0;
        for (int i = 0; i < 12; ++i) {
            const ui::Vec3 p{2.0f + 0.4f * i, 3.0f - 0.3f * i, 1.0f + 0.2f * i};
            ui::Vec2 px;
            if (!ui::worldToScreen(mvp, p, vpW, vpH, px)) continue;
            ui::Vec3 o, d;
            c.ray(px.x, px.y, vpW, vpH, o, d);
            // Distance from the point to the ray.
            const ui::Vec3 w = p - o;
            const float t = ui::dot(w, d) / ui::dot(d, d);
            worst = std::max(worst, ui::length(w - d * t));
        }
        ok(worst < 1e-3f, mode ? "worldToScreen inverts Camera::ray (ortho)"
                               : "worldToScreen inverts Camera::ray (perspective)");
    }

    // 2. gizmoPick finds the handles where they are drawn.
    {
        const ui::Camera c = mkCam(false);
        const ui::Mat4 mvp = c.viewProj(vpW / vpH);
        ui::Vec3 right, up;
        c.basis(right, up);
        const ui::Vec3 origin{2.0f, 3.0f, 1.0f};
        const bool en[3] = {true, true, true};
        const auto f = ui::makeGizmoFrame(origin, c.eye(), c.forward(), right, up,
                                          c.worldPerPixelAt(origin, vpH), 1.0f, en);
        ui::GizmoStyle st;
        ui::Vec2 originPx;
        ui::worldToScreen(mvp, origin, vpW, vpH, originPx);
        ok(ui::gizmoPick(f, st, mvp, vpW, vpH, originPx).part == ui::GizmoPart::Screen,
           "the centre of the gizmo is the screen handle");
        // Midway along the x shaft.
        ui::Vec2 tip;
        ui::worldToScreen(mvp, origin + f.axis[0] * (st.shaftPx * f.worldPerPx),
                          vpW, vpH, tip);
        const ui::Vec2 mid{(originPx.x + tip.x) * 0.5f, (originPx.y + tip.y) * 0.5f};
        ok(ui::gizmoPick(f, st, mvp, vpW, vpH, mid).part == ui::GizmoPart::AxisX,
           "the x shaft picks as AxisX");
        // Far away: nothing.
        ok(ui::gizmoPick(f, st, mvp, vpW, vpH, {5.0f, 5.0f}).part ==
               ui::GizmoPart::None,
           "empty space picks nothing");
        // A disabled axis is unreachable from ANY pixel.
        const bool en2[3] = {true, true, false};
        const auto f2 = ui::makeGizmoFrame(origin, c.eye(), c.forward(), right, up,
                                           c.worldPerPixelAt(origin, vpH), 1.0f, en2);
        bool sawZ = false;
        for (int y = 0; y < 60 && !sawZ; ++y)
            for (int x = 0; x < 90 && !sawZ; ++x) {
                const auto h = ui::gizmoPick(f2, st, mvp, vpW, vpH,
                                             {x * 10.0f, y * 10.0f});
                if (h.part == ui::GizmoPart::AxisZ) sawZ = true;
            }
        ok(!sawZ, "a disabled axis is unpickable everywhere");
    }

    // 3. THE property: the grabbed point stays under the cursor. Choose a
    //    target on the axis, find the pixel it projects to, drag there, and the
    //    solve must land on it. This is "the gizmo works", as an assertion.
    for (int mode = 0; mode < 2; ++mode) {
        const ui::Camera c = mkCam(mode == 1);
        const ui::Mat4 mvp = c.viewProj(vpW / vpH);
        ui::Vec3 right, up;
        c.basis(right, up);
        const ui::Vec3 origin{2.0f, 3.0f, 1.0f};
        const bool en[3] = {true, true, true};
        const auto f = ui::makeGizmoFrame(origin, c.eye(), c.forward(), right, up,
                                          c.worldPerPixelAt(origin, vpH), 1.0f, en);
        ui::Vec2 originPx;
        ui::worldToScreen(mvp, origin, vpW, vpH, originPx);

        auto rayAt = [&](const ui::Vec2& px) {
            ui::Vec3 o, d;
            c.ray(px.x, px.y, vpW, vpH, o, d);
            return ui::Ray{o, d};
        };

        ui::GizmoDrag g;
        // Grab a point PART WAY along the shaft, not the origin: if the anchor
        // were ignored the object would jump so its centre met the cursor.
        ui::Vec2 grabPx;
        const ui::Vec3 grabWorld = origin + f.axis[0] * 0.8f;
        ui::worldToScreen(mvp, grabWorld, vpW, vpH, grabPx);
        ok(ui::gizmoBeginDrag(g, f, ui::GizmoPart::AxisX, rayAt(grabPx)),
           mode ? "axis grab begins (ortho)" : "axis grab begins (perspective)");
        // Not moving the cursor must not move the object.
        const ui::Vec3 held = ui::gizmoSolve(g, rayAt(grabPx));
        ok(ui::length(held - origin) < 1e-4f,
           "a grab with no motion leaves the object exactly put");
        // Now aim at a known target on the axis.
        const ui::Vec3 want = origin + f.axis[0] * 1.7f;
        ui::Vec2 wantPx;
        ui::worldToScreen(mvp, want + (grabWorld - origin), vpW, vpH, wantPx);
        const ui::Vec3 got = ui::gizmoSolve(g, rayAt(wantPx));
        ok(ui::length(got - want) < 2e-3f,
           "the grabbed point tracks the cursor exactly");

        // No drift: a random walk that returns to the start returns the origin.
        ui::Vec2 px = grabPx;
        unsigned seed = 12345;
        for (int i = 0; i < 400; ++i) {
            seed = seed * 1103515245u + 12345u;
            px.x = grabPx.x + float((seed >> 16) % 400) - 200.0f;
            seed = seed * 1103515245u + 12345u;
            px.y = grabPx.y + float((seed >> 16) % 400) - 200.0f;
            (void)ui::gizmoSolve(g, rayAt(px));
        }
        const ui::Vec3 back = ui::gizmoSolve(g, rayAt(grabPx));
        ok(ui::length(back - origin) < 1e-4f,
           "400 moves and back: no drift (the solve is absolute)");
    }

    // 4. Snapping.
    {
        ui::SnapSettings sn;
        sn.mode = ui::SnapSettings::Mode::Grid;
        sn.h[0] = sn.h[1] = sn.h[2] = 0.25;
        const bool freeAll[3] = {true, true, true};
        const ui::Vec3 base{0, 0, 0};
        const ui::Vec3 got = ui::snapOrigin(sn, base, {0.31f, -0.62f, 1.13f}, freeAll);
        // -0.62/0.25 = -2.48, which rounds to -2 -> -0.5 (not -0.75).
        ok(std::abs(got.x - 0.25f) < 1e-5f && std::abs(got.y + 0.5f) < 1e-5f &&
               std::abs(got.z - 1.25f) < 1e-5f,
           "grid snap lands on multiples of h");
        ok(ui::length(ui::snapOrigin(sn, base, got, freeAll) - got) < 1e-6f,
           "grid snap is idempotent");
        // A constrained axis passes through untouched.
        const bool onlyX[3] = {true, false, false};
        const ui::Vec3 c2 = ui::snapOrigin(sn, base, {0.31f, -0.62f, 1.13f}, onlyX);
        ok(c2.y == -0.62f && c2.z == 1.13f,
           "a constrained axis is not snapped (it is already exact)");
        // Fraction snapping is relative to the grab.
        sn.mode = ui::SnapSettings::Mode::Fraction;
        sn.D = 2.0;
        sn.fraction = 0.5;   // 1.0 world units
        const ui::Vec3 b2{0.3f, 0, 0};
        const ui::Vec3 fr = ui::snapOrigin(sn, b2, {1.4f, 0, 0}, freeAll);
        ok(std::abs(fr.x - 1.3f) < 1e-5f, "fraction snap is relative to the grab");
        sn.mode = ui::SnapSettings::Mode::Free;
        ok(ui::snapOrigin(sn, base, {0.31f, 0, 0}, freeAll).x == 0.31f,
           "free snap changes nothing");
    }

    // 5. Analytic picking: the defect this replaces is an AABB that spans the
    //    whole domain depth for a cylinder.
    {
        dns::Grid g{10, 10, 10, 0.1, 0.1, 0.1};
        dns::Placement ref{0.5, 0.5, 0.5, 0.2};
        dns::SceneObject cyl;
        cyl.kind = dns::SceneObjectKind::Cylinder;
        cyl.d = 1.0;   // radius 0.1 about (0.5, 0.5)
        float t = 0;
        // Straight down z through the centre: a hit.
        ok(ui::rayObject(cyl, ref, g, nullptr, {{0.5f, 0.5f, -1.0f}, {0, 0, 1}}, t),
           "a ray down the cylinder axis hits it");
        // Offset by more than the radius: a miss. The AABB would also miss
        // here, but the next case is the one that matters.
        ok(!ui::rayObject(cyl, ref, g, nullptr, {{0.8f, 0.5f, -1.0f}, {0, 0, 1}}, t),
           "a ray outside the radius misses");
        // Diagonally past the corner of the AABB but outside the circle: the
        // box test would have said yes.
        const float c = 0.5f + 0.099f;
        ok(!ui::rayObject(cyl, ref, g, nullptr, {{c, c, -1.0f}, {0, 0, 1}}, t),
           "the AABB corner is not part of the cylinder");
        // A lens is an ellipse, not the box its aspect implies.
        dns::SceneObject lens = cyl;
        lens.kind = dns::SceneObjectKind::Lens;
        lens.aspect = 3.5;
        ok(ui::rayObject(lens, ref, g, nullptr, {{0.8f, 0.5f, -1.0f}, {0, 0, 1}}, t),
           "the lens is wide in x");
        ok(!ui::rayObject(lens, ref, g, nullptr, {{0.8f, 0.56f, -1.0f}, {0, 0, 1}}, t),
           "...but not in the corner of its box");
        // Nearest first: two spheres on the same ray.
        dns::SceneObject s1;
        s1.kind = dns::SceneObjectKind::Sphere;
        s1.d = 1.0;
        dns::SceneObject s2 = s1;
        s2.z = 2.0;    // further along +z, in units of D
        float t1 = 0, t2 = 0;
        const ui::Ray down{{0.5f, 0.5f, -1.0f}, {0, 0, 1}};
        ok(ui::rayObject(s1, ref, g, nullptr, down, t1) &&
               ui::rayObject(s2, ref, g, nullptr, down, t2) && t1 < t2,
           "the nearer of two stacked objects has the smaller t");
    }

    // 6. Click cycling reaches an occluded object.
    {
        const int hits[3] = {0, 1, 2};
        ok(ui::cycleSelection(hits, -1, false) == 0, "a fresh click takes the nearest");
        ok(ui::cycleSelection(hits, 0, true) == 1, "a second click at the same spot advances");
        ok(ui::cycleSelection(hits, 1, true) == 2, "...and again");
        ok(ui::cycleSelection(hits, 2, true) == 0, "...and wraps");
        ok(ui::cycleSelection(hits, 2, false) == 0, "moving the cursor restarts at the nearest");
    }

    return bad == 0;
}

// RUN and STOP, end to end with the REAL solver: the command the editor builds,
// spawned the way the button spawns it, publishing into the segment the viewer
// attaches to, and then stopped. Skipped when no dns binary is next to us.
bool checkRunStop()
{
    int bad = 0;
    auto ok = [&](bool cond, const char* what) {
        std::printf("   %-52s %s\n", what, cond ? "ok" : "FAILED");
        if (!cond) ++bad;
    };
    std::printf("\n== run / stop (end to end)\n");

    namespace fs = std::filesystem;
    std::error_code ec;
    std::string exe;
    for (const char* c : {"build-cuda/dns", "build/dns", "./dns"})
        if (fs::exists(c, ec)) {
            exe = fs::absolute(c, ec).string();
            break;
        }
    if (exe.empty()) {
        std::printf("   SKIP: no dns binary beside the check\n");
        return true;
    }

    // A long run, so the test is about start/stop and not about the run ending
    // by itself.
    const std::vector<std::string> argv{
        exe,   "flow",     "--n",    "48",   "--nz",   "1",     "--dr",  "8",
        "--re", "100",     "--ft",   "500",  "--far",  "tunnel", "--body", "cylinder",
        "--out", "uicheck", "--emit", "shm"};

    ui::Runner r;
    const std::string err = r.start(argv, ".");
    ok(err.empty(), "the editor's command starts");
    if (!err.empty()) {
        std::printf("      %s\n", err.c_str());
        return false;
    }

    // Wait for the segment the viewer attaches to.
    ui::LiveField lf;
    bool attached = false;
    for (int i = 0; i < 400 && !attached; ++i) {
        r.poll();
        if (!r.running()) break;
        attached = lf.attach("dns_uicheck").empty();
        if (!attached) std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    ok(attached, "the run publishes a segment the viewer can attach to");
    ok(r.running(), "the run is still in flight");

    bool gotFrame = false;
    for (int i = 0; i < 400 && attached && !gotFrame; ++i) {
        if (lf.poll()) gotFrame = true;
        else std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    ok(gotFrame, "live frames arrive from the real solver");
    ok(!gotFrame || !lf.planes().empty(), "the frame carries a plane");

    // ...and KEEP arriving. A viewer that shows one frame and then freezes is
    // the failure this asserts against: the counter must advance, and the
    // pixels must actually change as the wake develops.
    const std::uint64_t f0 = lf.frame();
    std::vector<float> first = lf.planes().empty() ? std::vector<float>{}
                                                   : lf.planes()[0].value;
    int advanced = 0;
    bool changed = false;
    for (int i = 0; i < 600 && advanced < 3; ++i) {
        if (lf.poll()) {
            ++advanced;
            if (!lf.planes().empty() && lf.planes()[0].value.size() == first.size())
                for (std::size_t q = 0; q < first.size(); ++q)
                    if (lf.planes()[0].value[q] != first[q]) {
                        changed = true;
                        break;
                    }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
    }
    ok(advanced >= 3, "successive frames keep arriving (the view updates)");
    ok(lf.frame() > f0, "the frame counter advances");
    ok(changed, "the field data actually changes between frames");
    ok(r.lineCount() > 0, "the solver's output is captured");

    // STOP. This is the path the button takes for a run we spawned.
    r.stop();
    ok(!r.running(), "stop() ends the run");
    // ...and the process is really gone, not merely marked stopped.
    bool reallyGone = true;
    if (attached) {
        const int pid = lf.publisherPid();
        for (int i = 0; i < 100; ++i) {
            if (::kill(pid_t(pid), 0) != 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        reallyGone = ::kill(pid_t(pid), 0) != 0;
    }
    ok(reallyGone, "the solver PROCESS is gone after stop");
    ok(!lf.running(), "the viewer sees the run as finished");

    // The segment must not outlive a stopped run, or the next RUN attaches to
    // a corpse.
    lf.detach();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ui::LiveField again;
    ok(!again.attach("dns_uicheck").empty(),
       "the segment is gone once the run is stopped");

    // A segment whose publisher was SIGKILLed is a corpse, and the viewer must
    // not attach to it: doing so shows a dead run as live and blocks RUN.
    {
        dns::ShmFrames corpse;
        std::vector<dns::ShmField> f1(1);
        std::snprintf(f1[0].name, sizeof f1[0].name, "v");
        if (corpse.create("dns_uicorpse", f1, {{2, 0.0f, 4, 4}}, 0, 0, 0)) {
            ok(!ui::LiveField::isStale("dns_uicorpse"),
               "a segment whose publisher is alive is not stale");
            // Forge a pid that cannot exist.
            ui::LiveField probe;
            if (probe.attach("dns_uicorpse").empty()) {
                // The publisher writes its own pid; overwrite it via a second
                // mapping to simulate a process that is gone.
                const int fd = shm_open("/dns_uicorpse", O_RDWR, 0);
                if (fd >= 0) {
                    auto* h = static_cast<dns::ShmHeader*>(
                        mmap(nullptr, sizeof(dns::ShmHeader), PROT_READ | PROT_WRITE,
                             MAP_SHARED, fd, 0));
                    if (h != MAP_FAILED) {
                        h->pid = 0x7ffffffe;   // no such process
                        munmap(h, sizeof(dns::ShmHeader));
                        ok(ui::LiveField::isStale("dns_uicorpse"),
                           "a segment whose publisher is gone IS stale");
                        probe.detach();
                        ui::LiveField again2;
                        (void)again2.attach("dns_uicorpse");
                        ok(!again2.running(),
                           "a stale segment never reads as a running run");
                    }
                    close(fd);
                }
            }
        }
    }

    // And a second run must be startable straight after.
    const std::string err2 = r.start(argv, ".");
    ok(err2.empty(), "a second run starts after a stop");
    r.stop();
    ok(!r.running(), "the second run stops too");
    return bad == 0;
}

int check()
{
    int failures = 0;
    failures += !checkUnits();
    failures += ui::guiCheck() != 0;
    failures += !checkDoc();
    failures += !checkRunner();
    failures += !checkLive();
    failures += !checkGizmo();
    failures += !checkRunStop();

    ui::DocState d2;   // the default: 96^3 box, cylinder, incompressible
    failures += !checkOne("3D cylinder, incompressible, 96^3 (the default)", d2, true);

    ui::DocState sphere = d2;
    sphere.bodyKind = 2;
    sphere.n = 96;
    sphere.nz = 96;
    failures += !checkOne("3D sphere, incompressible, n 96", sphere, true);

    // Compressible needs a Mach the solver can march, and Re and Mach are no
    // longer independent: one material and one freestream fix both. Air at
    // 68.6 m/s is Ma 0.2, and a 0.1 mm body then puts Re at ~450 -- under the
    // 800 past which CLAUDE.md requires --nscbc-strip. That coupling is the
    // point of physical units, not a workaround for it.
    ui::DocState comp = d2;
    comp.physics = 1;
    comp.n = 128;
    comp.nz = 1;      // 2D: keeps --check fast; the 3D path is covered above
    comp.far = 3;
    comp.phys.dBody = 1e-4;
    comp.phys.uInf = 68.6;
    failures += !checkOne("2D cylinder, compressible, NSCBC", comp, false);

    // The blend is widened past the shared default: at n = 128 a ratio of 4
    // over the default blend puts the adjacent-cell growth at 1.2518, and the
    // validator rejects anything over 1.25 (the central flux has no
    // dissipation to absorb a sharp transition). Not a UI quirk -- the CLI
    // refuses the same combination, which is exactly what the editor should
    // surface rather than paper over.
    ui::DocState focal = comp;
    focal.n = 256;
    focal.focalAuto = true;
    focal.focalBlend = 0.8;
    failures += !checkOne("2D cylinder, compressible, --focal auto", focal, false);

    ui::DocState scene = d2;
    scene.bodyKind = 8;
    scene.items.clear();
    for (int i = 0; i < 3; ++i) {
        dns::SceneObject o;
        o.kind = dns::SceneObjectKind::Cylinder;
        o.x = -1.2 + 1.2 * i;
        o.y = 0.6 * (i - 1);
        o.d = 0.6;
        o.label = dns::defaultSceneName(o.kind, size_t(i));
        scene.items.push_back({o, ui::ObjId(i + 1)});
    }
    failures += !checkOne("3D scene, three cylinders", scene, true);

    ui::DocState les = sphere;
    les.les = 1.35;
    les.sgs = 1;   // sigma
    failures += !checkOne("3D sphere, sigma LES", les, true);

    std::printf("\n%s\n", failures ? "FAILED" : "all cases ok");
    return failures ? 7 : 0;
}

} // namespace

int main(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--check") return check();
        if (a == "--help" || a == "-h") {
            std::printf(
                "usage: dns-ui\n"
                "\n"
                "Interactive obstacle editor and grid viewer for `dns flow`.\n"
                "Edit the case in the panels; the grid, the body surface and the\n"
                "cut cells (fluid fraction lambda and face apertures) update from\n"
                "the real geometry pipeline. Nothing is written unless you ask:\n"
                "\"write scene.toml\" in the Export panel is the only output.\n"
                "\n"
                "mouse   LMB orbit (pan in 2D), RMB/shift+LMB pan, wheel zoom\n"
                "        click an object to select it, drag to move it\n"
                "keys    F frame domain, B frame body, G grid, C cut cells,\n"
                "        M solid mask, A apertures, Del remove object, Esc quit\n"
                "\n"
                "env     DNS_UI_FONT / DNS_UI_FONT_MONO   font override\n"
                "        DNS_UI_VALIDATION=1              Vulkan validation\n"
                "\n"
                "--scene FILE opens an existing scene .toml (docs/scene.md); you\n"
                "can also drop one on the window.\n"
                "\n"
                "--check runs the geometry pipeline headlessly and prints the\n"
                "resolution and cut-cell numbers; no window, no GPU, no font.\n");
            return 0;
        }
        if (a == "--scene") { ++i; continue; }   // handled below
        std::fprintf(stderr, "dns-ui: unexpected argument '%s' (try --help)\n",
                     argv[i]);
        return 1;
    }

    std::string openPath;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--scene" && i + 1 < argc) openPath = argv[++i];
    }

    auto app = ui::App::create();
    if (!app) {
        std::fprintf(stderr, "dns-ui: %s\n", app.error().c_str());
        return 5;
    }
    if (!openPath.empty()) {
        const std::string err = (*app)->openSceneFile(openPath);
        if (!err.empty()) {
            std::fprintf(stderr, "dns-ui: %s\n", err.c_str());
            return 3;
        }
    }
    return (*app)->run();
}
