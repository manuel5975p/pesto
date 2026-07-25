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
#include "ui/view.hpp"

#include <chrono>
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

int check()
{
    int failures = 0;

    ui::DocState d2;   // the default: 2D cylinder, incompressible
    failures += !checkOne("2D cylinder, incompressible, Re 200", d2, false);

    ui::DocState sphere = d2;
    sphere.bodyKind = 2;
    sphere.n = 96;
    sphere.nz = 96;
    failures += !checkOne("3D sphere, incompressible, n 96", sphere, true);

    ui::DocState comp = d2;
    comp.physics = 1;
    comp.n = 128;
    comp.far = 3;   // nscbc
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
    scene.scene.objects.clear();
    for (int i = 0; i < 3; ++i) {
        dns::SceneObject o;
        o.kind = dns::SceneObjectKind::Cylinder;
        o.x = -1.2 + 1.2 * i;
        o.y = 0.6 * (i - 1);
        o.d = 0.6;
        scene.scene.objects.push_back(o);
    }
    failures += !checkOne("2D scene, three cylinders", scene, false);

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
                "--check runs the geometry pipeline headlessly and prints the\n"
                "resolution and cut-cell numbers; no window, no GPU, no font.\n");
            return 0;
        }
        std::fprintf(stderr, "dns-ui: unexpected argument '%s' (try --help)\n",
                     argv[i]);
        return 1;
    }

    auto app = ui::App::create();
    if (!app) {
        std::fprintf(stderr, "dns-ui: %s\n", app.error().c_str());
        return 5;
    }
    return (*app)->run();
}
