// Scene layer: strict TOML parse (toml++), model merge into RawConfig, validator
// plumbing for --scene, and the union-sdf mesh builder.
#include "dns/cli.hpp"
#include "dns/config.hpp"
#include "dns/scene.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include <vector>

namespace {

using dnstest::check;



const char* kTandem = R"(
[model]
type = "les"
cs = 0.16

[[objects]]
kind = "cylinder"
name = "front"
x = 0
y = 0
d = 1.0

[[objects]]
kind = "cylinder"
name = "rear"
x = 2.5
y = 0
d = 0.5
)";

void testParse()
{
    auto sc = dns::parseScene(kTandem);
    check(sc.has_value(), "tandem scene parses");
    if (!sc) return;
    check(sc->model == dns::SceneModel::Les && sc->cs == 0.16,
          "model les cs=0.16");
    check(sc->objects.size() == 2, "two objects");
    check(sc->objects[1].label == "rear" && sc->objects[1].x == 2.5
              && sc->objects[1].d == 0.5,
          "rear object fields");
    check(sc->zInvariant() && !sc->has3d(), "cylinders are z-invariant");

    // Default model is DNS; labels default to kind + index and must be unique.
    auto d = dns::parseScene("[[objects]]\nkind = \"cylinder\"\n"
                             "[[objects]]\nkind = \"lens\"\naspect = 2\n");
    check(d.has_value() && d->model == dns::SceneModel::Dns, "default model dns");
    check(d.has_value() && d->objects[0].label == "cylinder0"
              && d->objects[1].label == "lens1",
          "default labels");

    // Strictness: each of these must be a hard error, never a silent default.
    const char* bad[] = {
        "",                                                       // no objects
        "[[objects]",                                             // bad TOML
        "[[objects]]\nkind = \"torus\"",                          // unknown kind
        "[[objects]]\nkind = \"cylinder\"\nr = 1",                // unknown key
        "[[objects]]\nkind = \"cylinder\"\nd = 0",                // d <= 0
        "[[objects]]\nkind = \"cylinder\"\nd = \"1\"",            // wrong type
        "[[objects]]\nkind = \"cylinder\"\nd = inf",              // non-finite
        "[[objects]]\nkind = \"cylinder\"\nd = 1\nd = 2",         // duplicate key
        "[[objects]]\nkind = \"mesh\"",                           // mesh sans path
        "[[objects]]\nkind = \"cylinder\"\nscale = 2",            // mesh-only key
        "[[objects]]\nkind = \"lens\"\naspect = 1",               // aspect <= 1
        "blorp = 1\n[[objects]]\nkind = \"cylinder\"",            // unknown top key
        "objects = 3",                                            // not an array
        "[model]\ntype = \"rans\"\n[[objects]]\nkind = \"cylinder\"",
        "[model]\ntype = \"les\"\n[[objects]]\nkind = \"cylinder\"",   // no cs
        "[model]\ntype = \"dns\"\ncs = 0.2\n[[objects]]\nkind = \"cylinder\"",
        "[model]\ntype = \"les\"\ncs = 0.2\nsgs = \"wale\"\n"
        "[[objects]]\nkind = \"cylinder\"",                       // unknown sgs
        "[model]\ntype = \"dns\"\nsgs = \"sigma\"\n"
        "[[objects]]\nkind = \"cylinder\"",                       // sgs sans les
        "[[objects]]\nkind = \"cylinder\"\nname = \"a\"\n"
        "[[objects]]\nkind = \"sphere\"\nname = \"a\"",           // duplicate name
        "[[objects]]\nkind = \"cylinder\"\nname = \"a b\"",       // bad name chars
    };
    for (const char* s : bad)
        check(!dns::parseScene(s).has_value(),
              (std::string("rejects: ") + (s[0] ? s : "<empty>")).c_str());
}

void testModelMerge()
{
    auto sc = dns::parseScene(kTandem);
    if (!sc) return;

    dns::RawConfig r;
    check(dns::applySceneModel(*sc, r).has_value() && r.les == 0.16
              && r.sgs == dns::SgsModel::Smagorinsky,
          "scene les lands in raw.les (sgs defaults to smag)");

    {   // sgs = "sigma" parses and lands in raw.sgs; a conflicting --sgs errors
        auto s2 = dns::parseScene(
            "[model]\ntype = \"les\"\ncs = 1.35\nsgs = \"sigma\"\n"
            "[[objects]]\nkind = \"cylinder\"");
        check(s2.has_value() && s2->sgs == dns::SgsModel::Sigma,
              "scene sgs = \"sigma\" parses");
        if (s2)
        {
            dns::RawConfig rs;
            check(dns::applySceneModel(*s2, rs).has_value()
                      && rs.sgs == dns::SgsModel::Sigma && rs.les == 1.35,
                  "scene sigma lands in raw.sgs");
            dns::RawConfig rc;
            rc.given.insert("--sgs");
            rc.sgs = dns::SgsModel::Smagorinsky;
            check(!dns::applySceneModel(*s2, rc).has_value(),
                  "conflicting --sgs is a config error");
        }
    }

    dns::RawConfig r2;
    r2.given.insert("--les");
    r2.les = 0.2;
    check(!dns::applySceneModel(*sc, r2).has_value(),
          "conflicting --les is a config error");

    auto dnsScene = dns::parseScene("[[objects]]\nkind = \"cylinder\"\n");
    dns::RawConfig r3;
    r3.given.insert("--les");
    r3.les = 0.2;
    check(dnsScene.has_value()
              && !dns::applySceneModel(*dnsScene, r3).has_value(),
          "--les against a dns scene is a config error");
}

void testConfigTokens()
{
    auto ct = dns::cli::configTokens(
        "subcommand = \"channel\"\nn = 512\nre_tau = 180\nmach = 0.3\n"
        "noslip = true\nemit = [\"stats\", \"png\"]\nout = \"/tmp/x\"\n");
    check(ct.has_value(), "config tokens parse");
    if (!ct) return;
    check(ct->subcommand == "channel", "config subcommand");
    // toml++ tables iterate in sorted key order; the flags are independent,
    // so order is a stability detail, not semantics.
    const std::vector<std::string> want{
        "--emit", "stats", "--emit", "png",      "--mach", "0.3",
        "--n",    "512",   "--noslip",           "--out",  "/tmp/x",
        "--re-tau", "180"};
    check(ct->args == want, "config token expansion");

    check(!dns::cli::configTokens("noslip = false").has_value(),
          "false flag rejected");
    check(!dns::cli::configTokens("n = 512\nn = 3").has_value(),
          "duplicate key rejected");
    check(!dns::cli::configTokens("n = [").has_value(), "bad TOML rejected");
    check(!dns::cli::configTokens("subcommand = 3").has_value(),
          "non-string subcommand rejected");
}

void testCliAndValidate()
{
    // --scene parses as a flow option and lands in RawConfig.
    const char* argv1[] = {"dns", "flow", "--scene", "s.toml"};
    auto raw = dns::cli::parse(4, const_cast<char**>(argv1));
    check(raw.has_value() && raw->scene == "s.toml" && raw->given.contains("--scene"),
          "--scene parses");
    if (!raw) return;

    auto c = dns::validate(*raw);
    check(c.has_value() && c->bodyKind == dns::BodyKind::Scene
              && c->scenePath == "s.toml",
          "validate maps --scene to BodyKind::Scene");
    check(c.has_value() && c->meta.body == "scene:s.toml", "meta.body records it");

    dns::RawConfig both = *raw;
    both.body = "cylinder";
    both.given.insert("--body");
    check(!dns::validate(both).has_value(), "--scene + --body rejected");
}

void testBuildGeometry()
{
    using std::numbers::pi;
    // The incompressible flow grid convention at n = 64: L = 2 pi, ghost ring.
    const int n = 64;
    const double L = 2 * pi, h = L / n;
    const auto g = dns::Grid::make(n + 2, n + 2, 1, (n + 2) * h, (n + 2) * h, 1.0);
    const double D = L / 8.0;
    const dns::Placement ref{h + L / 3.0, h + L / 2.0, 0.5 * h, D};

    auto sc = dns::parseScene(kTandem);
    if (!sc) return;
    auto geo = dns::buildSceneGeometry(g, ref, *sc);
    check(geo.has_value(), "tandem geometry builds");
    if (!geo) return;

    // Union sdf: solid (< 0) at both centres, fluid far upstream.
    check(geo->sdf(ref.cx, ref.cy, 0) < 0, "solid at the front centre");
    check(geo->sdf(ref.cx + 2.5 * D, ref.cy, 0) < 0, "solid at the rear centre");
    check(geo->sdf(ref.cx - 2.0 * D, ref.cy, 0) > 0, "fluid upstream");

    // The cut agrees: lambda 0 in the bodies, 1 in the free stream.
    auto at = [&](double x, double y) {
        return geo->cut.lambda[g.idx(int(x / g.hx), int(y / g.hy), 0)];
    };
    check(at(ref.cx, ref.cy) == 0.0, "cut solid in front cylinder");
    check(at(ref.cx + 2.5 * D, ref.cy) == 0.0, "cut solid in rear cylinder");
    check(at(ref.cx - 2.0 * D, ref.cy) == 1.0, "cut fluid upstream");

    // Per-object tally boxes: labelled, disjoint, each around its own object.
    check(geo->objects.size() == 2 && geo->objects[0].label == "front"
              && geo->objects[1].label == "rear",
          "object boxes labelled");
    check(geo->objects[0].bbox[3] < geo->objects[1].bbox[0],
          "tandem boxes disjoint in x");
    check(geo->bbox[0] == geo->objects[0].bbox[0]
              && geo->bbox[3] == geo->objects[1].bbox[3],
          "union bbox spans both");

    // 3D object on a native-2D grid is an input error.
    auto s3 = dns::parseScene("[[objects]]\nkind = \"sphere\"\n");
    check(s3.has_value() && !dns::buildSceneGeometry(g, ref, *s3).has_value(),
          "sphere at nz == 1 rejected");

    // An object reaching the domain boundary is an input error.
    auto sBig = dns::parseScene("[[objects]]\nkind = \"cylinder\"\nd = 9\n");
    check(sBig.has_value() && !dns::buildSceneGeometry(g, ref, *sBig).has_value(),
          "boundary-crossing object rejected");
}

} // namespace

int main()
{
    testParse();
    testModelMerge();
    testConfigTokens();
    testCliAndValidate();
    testBuildGeometry();
    if (dnstest::gFailures) { std::printf("%d failure(s)\n", dnstest::gFailures); return 1; }
    std::printf("ok\n");
    return 0;
}
