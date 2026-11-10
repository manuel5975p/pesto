// G26 (docs/nozzle.md): every refusal of the nozzle route, verbatim, through the real binary;
// --dry-run passes on the route; --shock's other routes refuse as before; the label never says
// DNS under FV hll. Host: dns --dry-run touches neither GPU nor output.
#include "check.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <string>

using dnstest::check;
namespace fs = std::filesystem;

namespace {

struct Out
{
    int rc = -1;
    std::string text;
};

Out dns(const std::string& args)
{
    const std::string cmd = std::format("{} {} 2>&1", DNS_BIN, args);
    FILE* p = popen(cmd.c_str(), "r");
    Out o;
    if (!p) return o;
    std::array<char, 4096> b;
    while (fgets(b.data(), int(b.size()), p)) o.text += b.data();
    const int st = pclose(p);
    o.rc = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
    return o;
}

void refuses(const std::string& args, int rc, const std::string& text)
{
    const Out o = dns(args);
    const bool ok = o.rc == rc && o.text.find(text) != std::string::npos;
    if (!ok) std::printf("    rc %d: %s", o.rc, o.text.c_str());
    check(ok, std::format("rc {}: {}", rc, text));
}

fs::path tmp()
{
    static const fs::path d = [] {
        const fs::path p = fs::temp_directory_path() / std::format("dns_nozcli_{}", getpid());
        fs::create_directories(p);
        return p;
    }();
    return d;
}

std::string bc(const std::string& name, const std::string& text)
{
    const fs::path f = tmp() / (name + ".toml");
    std::ofstream(f) << text;
    return f.string();
}

} // namespace

int main()
{
    const std::string dataA = std::string(DNS_MESH_DATA) + "/../fv/nozzle_A", dataW = std::string(DNS_MESH_DATA) + "/../fv/nozzle_W";
    const std::string bcA = dataA + "/bc.toml", bcW = dataW + "/bc.toml";
    const std::string o = " --out " + (tmp() / "o").string() + " --dry-run";
    const std::string hll = "flow --physics compressible --polymesh " + dataA + " --bc " + bcA + " --shock hll --mach 1 --tt 1" + o;
    const std::string wedge = "flow --physics compressible --polymesh " + dataW + " --bc " + bcW + " --shock hll --mach 1 --tt 1" + o;

    // The route.
    Out a = dns(hll + " --inviscid");
    std::printf("%s", a.text.c_str());
    check(a.rc == 0, "--dry-run passes on the planar nozzle");
    check(a.text.find("Euler") != std::string::npos && a.text.find("DNS,") == std::string::npos && a.text.find(" DNS:") == std::string::npos,
          "Euler in the label, never DNS");
    check(a.text.find("hll (mesh lines, stride 2, 0.02..0.06, per step) + D4 eps 0.0625 |u.S|") != std::string::npos, "the shock line");
    check(a.text.find("NUMERICAL dissipation everywhere, not DNS") != std::string::npos, "kFvShockSuffix");
    a = dns(wedge + " --re 1000");
    check(a.rc == 0 && a.text.find("compressible laminar") != std::string::npos, "--dry-run passes on the wedge; laminar under --re");
    check(a.text.find("sgs        DNS") == std::string::npos, "the dry-run's model line never says DNS");

    // parser (exit 3)
    const std::string run = "flow --physics compressible --polymesh " + dataA + " --shock hll --mach 1 --tt 1 --inviscid" + o + " --bc ";
    const std::string in = "[nozzle]\np = 1\nt = 1\nastar = 1\n[patch.a]\ntype = \"total\"\n";
    refuses(run + bc("t1", "[patch.inlet]\ntype = \"total\"\np0 = 1\n"), 3, "patch.inlet: total needs p0 and t0");
    refuses(run + bc("t2", "[patch.o]\ntype = \"outflow\"\nexit = \"wide\"\n"), 3, "patch.o.exit: must be \"fixed\", \"extrapolate\" or \"riemann\"");
    refuses(run + bc("t3", "[patch.o]\ntype = \"outflow\"\nexit = \"extrapolate\"\np = 1\n"), 3, "patch.o: exit = \"extrapolate\" reads no p, tback: give neither");
    refuses(run + bc("t4", "[patch.o]\ntype = \"outflow\"\nexit = \"riemann\"\np = 1\n"), 3, "patch.o: exit = \"riemann\" needs p and tback");
    refuses(run + bc("t5", "[nozzle]\nq = 1\n[patch.o]\ntype = \"far\"\n"), 3, "[nozzle]: unknown key 'q' (p, t, astar, axis, cd)");
    refuses(run + bc("t6", "[nope]\n[patch.o]\ntype = \"far\"\n"), 3, "unknown top-level key 'nope' (reference, patch, nozzle, thrust)");
    refuses(run + bc("t7", "[patch.o]\ntype = \"wobble\"\n"), 3, "one of: wall, slip, inflow, outflow, far, empty, open, total, wedge");
    refuses(run + bc("t8", "[patch.o]\n"), 3, "needs type = \"wall|slip|inflow|outflow|far|empty|open|total|wedge\"");

    // the --bc against the mesh
    std::string w = "[nozzle]\np = 0.1\nt = 1\nastar = 1\n[patch.inlet]\ntype = \"total\"\np0 = 1\nt0 = 1\n[patch.outlet]\ntype = \"outflow\"\nexit = \"extrapolate\"\n[patch.wall]\ntype = \"slip\"\n";
    refuses("flow --physics compressible --polymesh " + dataW + " --bc " + bc("n1", "[patch.inlet]\ntype = \"far\"\n[patch.outlet]\ntype = \"far\"\n[patch.wall]\ntype = \"wall\"\n") +
                " --re 100 --mach 0.3 --tt 1" + o,
            3, "patch `back`: wedge patches are not supported (FaceMesh P8)");
    refuses("flow --physics compressible --polymesh " + dataW + " --shock hll --mach 1 --tt 1 --inviscid" + o + " --bc " + bc("n2", w + "[patch.back]\ntype = \"slip\"\n"), 3,
            "patch `back`: a `wedge` patch takes type = \"wedge\" only");
    refuses("flow --physics compressible --polymesh " + dataA + " --shock hll --mach 1 --tt 1 --inviscid" + o + " --bc " + bc("n3", w + "[patch.axis]\ntype = \"wedge\"\n"), 3,
            "patch `axis`: type = \"wedge\" needs a `wedge` mesh patch, it is symmetryPlane");

    // flags
    refuses("flow --physics compressible --body cylinder --shock hll --shock-d4 0.1" + o, 2,
            "--shock-d4: is the FV polyMesh route's (the structured solvers carry no background term)");
    refuses("flow --physics compressible --polymesh " + dataA + " --bc " + bc("p1", "[patch.inlet]\ntype = \"far\"\n[patch.outlet]\ntype = \"far\"\n[patch.wall]\ntype = \"slip\"\n[patch.axis]\ntype = \"slip\"\n") +
                " --shock sense --shock-d4 0.1 --re 100 --mach 0.3 --tt 1" + o,
            2, "--shock-d4: needs --shock hll");
    refuses(hll + " --inviscid --shock-d4 0.2", 2, "--shock-d4: must be in [0, 0.125]");
    refuses("flow --physics compressible --polymesh " + dataA + " --bc " + bc("p2", "[patch.inlet]\ntype = \"far\"\n[patch.outlet]\ntype = \"far\"\n[patch.wall]\ntype = \"slip\"\n[patch.axis]\ntype = \"slip\"\n") +
                " --inviscid --mach 0.3 --tt 1" + o,
            2, "--inviscid: needs --shock hll");
    refuses(hll + " --inviscid --re 100", 2, "--inviscid: --re is the viscous number: give one of them");
    refuses(hll + " --inviscid --prandtl 0.7", 2, "--inviscid: no viscous flag with it (--visc, --prandtl, --tinf)");
    refuses("flow --physics compressible --polymesh " + dataA + " --shock hll --mach 1 --tt 1 --inviscid" + o + " --bc " + bc("p3", "[nozzle]\np = 1\nt = 1\nastar = 1\n[patch.wall]\ntype = \"wall\"\n[patch.inlet]\ntype = \"far\"\n"), 2,
            "--bc: patch `wall`: --inviscid: a no-slip wall needs viscosity, use type = \"slip\"");
    refuses("flow --physics compressible --polymesh " + dataA + " --shock hll --mach 1 --tt 1 --inviscid" + o + " --bc " + bc("p4", "[patch.inlet]\ntype = \"total\"\np0 = 1\nt0 = 1\n"), 2,
            "--bc: a total patch needs [nozzle] (the rest state)");
    refuses("flow --physics compressible --polymesh " + dataA + " --bc " + bcA + " --re 100 --mach 0.3 --tt 1" + o, 2, "--bc: [nozzle] needs --shock hll");
    refuses(hll + " --inviscid --cfl 1.3", 2, "--cfl: must be <= 1.2 on the FV path: the Sigma form is already the axis sum (docs/shock.md)");
    check(dns(hll + " --inviscid --cfl 1.2").rc == 0, "--cfl 1.2 passes");
    refuses("flow --physics compressible --body cylinder --foam-write 1" + o, 2, "--foam-write: needs --polymesh");
    refuses(wedge + " --re 100 --les 0.1", 2, "a `wedge` mesh takes no --les, --rans (an axisymmetric mean flow)");
    refuses(hll + " --re 100 --les 0.1", 2, "FV --shock with LES: not built (no gate)");
    refuses(hll + " --inviscid --steady 1e-3", 2, "--steady is not built with --shock");
    refuses(hll.substr(0, hll.find(" --mach 1")) + " --mach -1 --tt 1 --inviscid" + o, 2, "--mach: must be finite and > 0");
    refuses("flow --physics compressible --polymesh tests/x --bc " + bc("p5", "[patch.a]\ntype = \"far\"\n") + " --re 100 --mach 2 --tt 1" + o, 2,
            "--mach: required, in (0, 1): the KG flux has no shock capturing");

    // the other routes refuse as before
    refuses("flow --physics compressible --solver fv --body none --shock hll" + o, 2, "--shock: is not built on --solver fv");
    // masterplan_combnozzle P3: the flame polyMesh route takes --shock hll (Shock x C0) and the nozzle's tables.
    refuses("flame --polymesh " + dataA + " --bc " + bcA + " --shock hll" + o, 2, "--bc: patch.inlet: total needs y on a flame (the fresh mass fractions)");
    refuses("flame --polymesh " + dataA + " --bc " + bcA + o, 2, "belongs to the nozzle route (dns flow --shock hll --polymesh)");
    refuses("flame --physics lowmach --polymesh " + dataA + " --bc " + bcA + o, 3, "[nozzle]: the compressible nozzle route's table");
    refuses("flame --physics lowmach --polymesh " + dataA + " --bc " + bc("lm", "[patch.inlet]\ntype = \"total\"\np0 = 1\nt0 = 1\n") + o, 3,
            "patch.inlet: type = \"total\": a low-Mach inlet has no stagnation state");
    refuses("conv --solver fv --polymesh " + dataA + " --bc " + bcA + " --ny 16 --rayleigh 2e4 --tt 1" + o, 2, "belongs to the nozzle route");

    // --gamma on the flame (masterplan_combnozzle R7): the premixed C0 polyMesh flame only.
    const std::string flBc = bc("fl", "[patch.inlet]\ntype = \"far\"\ny = [1.0]\n[patch.outlet]\ntype = \"outflow\"\nburnt = "
                                      "true\n[patch.wall]\ntype = \"slip\"\n[patch.axis]\ntype = \"slip\"\n");
    const std::string fl = "flame --polymesh " + dataA + " --bc " + flBc + " --upstream 7 --tt 1" + o;
    Out g = dns(fl + " --gamma 1.2");
    check(g.rc == 0, "--gamma 1.2 passes on the premixed polyMesh flame");
    check(g.text.find("calorically perfect, gamma 1.2 for fresh AND burnt gas, frozen, Le=1") != std::string::npos,
          "the label names gamma 1.2 for fresh and burnt gas");
    check(g.text.find("M 0.01 gamma 1.2 (fresh AND burnt gas, frozen; Tu 8333.333333, q/(cp Tu) 4)") != std::string::npos,
          "the dry-run's flame line: Tu = 1/(gamma M^2)");
    g = dns(fl);
    check(g.rc == 0 && g.text.find("gamma") == std::string::npos, "no --gamma: no gamma text (the 1.4 bytes)");
    g = dns(fl + " --gamma 1.4");
    check(g.rc == 0 && g.text.find("gamma") == std::string::npos, "--gamma 1.4 is the default, bytes and all");
    refuses("flame --gamma 1.2" + o, 2,
            "--gamma: is the compressible FV polyMesh routes' (structured and low-Mach flames are gamma 1.4)");
    refuses("flame --physics lowmach --gamma 1.2" + o, 2,
            "--gamma: is the compressible FV polyMesh routes' (structured and low-Mach flames are gamma 1.4)");
    refuses("flame --solver fv --gamma 1.2" + o, 2, "--gamma: needs --polymesh");
    refuses(fl + " --gamma 1.7", 2, "--gamma: must be in (1, 1.67]");
    refuses(fl + " --gamma 1", 2, "--gamma: must be in (1, 1.67]");
    refuses(fl + " --gamma 1.2 --les 0.1", 2, "--gamma: with --les: not built");
    refuses(fl + " --gamma 1.2 --tfles 2", 2, "--gamma: with --tfles: not built");
    refuses(fl + " --gamma 1.2 --equivalence 1.3", 2, "--gamma: with --equivalence: not built");
    refuses("flame --mode diffusion --polymesh " + dataA + " --bc " + flBc + " --gamma 1.2" + o, 2,
            "--gamma: with --mode other than premixed: not built");
    // [thrust] (masterplan_combnozzle R2): keys, texts, the dry-run line; --shock's mesh rule (R8).
    std::string bcText;
    {
        std::ifstream in(bcA);
        bcText.assign(std::istreambuf_iterator<char>(in), {});
    }
    const std::string thr = "flow --physics compressible --polymesh " + dataA + " --shock hll --mach 1 --tt 1 --inviscid" + o + " --bc ";
    Out t = dns(thr + bc("th0", bcText + "\n[thrust]\npa = 0.05\n"));
    check(t.rc == 0 && t.text.find("thrust     pa 0.05 (a ratio, as [nozzle] p): the impulse ledger, PREFIX_thrust.csv") != std::string::npos,
          "[thrust] passes the dry-run and names the ledger");
    check(dns(hll + " --inviscid").text.find("thrust ") == std::string::npos, "no [thrust]: no thrust line");
    refuses(thr + bc("th1", bcText + "\n[thrust]\npa = 0\nq = 1\n"), 3, "[thrust]: unknown key 'q' (pa, station)");
    refuses(thr + bc("th2", "[thrust]\npa = 0\n[patch.o]\ntype = \"far\"\n"), 3, "[thrust] needs [nozzle]");
    refuses(thr + bc("th3", bcText + "\n[thrust]\npa = -0.1\n"), 3, "[thrust].pa: must be >= 0");
    refuses(thr + bc("th4", bcText + "\n[thrust]\nstation = 1\n"), 3, "[thrust]: needs pa");
    refuses(thr + bc("th5", bcText + "\n[thrust]\npa = 0\nstation = 1\n"), 2, "--bc: [thrust].station is the reacting nozzle's (dns flame)");
    refuses("flame --polymesh " + dataA + " --bc " + bc("th7", bcText + "\n[thrust]\npa = 0\n") + o, 2,
            "[nozzle] belongs to the nozzle route (dns flow --shock hll --polymesh)");
    // Shock x C0 on the flame route (masterplan_combnozzle P3): acceptance, label, every text verbatim.
    {
        Out s = dns(fl + " --shock hll");
        check(s.rc == 0 && s.text.find("laminar reacting [premixed single-step C0, calorically perfect gamma 1.4, Le 1, M_flame 0.01 "
                                       "(inflated)]") != std::string::npos &&
                  s.text.find("background D4 eps 0.0625 |u.S|: NUMERICAL dissipation everywhere, not DNS") != std::string::npos &&
                  s.text.find("label      DNS") == std::string::npos,
              "flame --polymesh --shock hll: passes, labelled laminar reacting + the FV shock suffix, never DNS");
        check(dns(fl + " --shock hll --gamma 1.2").text.find("calorically perfect gamma 1.2") != std::string::npos, "the label's gamma");
        check(dns(fl + " --shock hll --shock-d4 0").text.find("D4 eps 0 |u.S|") != std::string::npos, "--shock-d4 0 on the flame");
        refuses(fl + " --shock-d4 0", 2, "--shock-d4: needs --shock hll");
        refuses(fl + " --shock hll --shock-d4 0.2", 2, "--shock-d4: must be in [0, 0.125]");
        refuses("flame --shock hll --shock-d4 0" + o, 2, "--shock-d4: is the FV polyMesh route's (the structured solvers carry no background term)");
        refuses("flame --solver fv --shock hll" + o, 2, "--shock: is not built on --solver fv");
        refuses("flame --physics lowmach --shock hll" + o, 2, "--shock: is compressible only");
        check(dns(fl + " --shock hll --cfl 0.9").rc == 0, "--cfl 0.9 passes on the FV flame (the Sigma form, not the structured 0.6)");
        refuses(fl + " --shock hll --equivalence 1.3", 2, "--shock: is not built with --equivalence (two reactants)");
        refuses("flame --mode diffusion --polymesh " + dataA + " --bc " + flBc + " --shock hll" + o, 2,
                "--shock: is not built for --mode diffusion: premixed and chamber");
        refuses(fl + " --shock hll --mach 0.06", 2, "--mach: must be in (0, 0.05]");
        const std::string inlet = "[patch.outlet]\ntype = \"outflow\"\nexit = \"riemann\"\np = 0.5\ntback = 1\n[patch.wall]\ntype = "
                                  "\"slip\"\n[patch.axis]\ntype = \"slip\"\n[patch.inlet]\ntype = \"total\"\np0 = 1.01\nt0 = 1.003\n";
        const std::string fh = "flame --polymesh " + dataA + " --upstream 7 --tt 1 --shock hll" + o + " --bc ";
        Out n = dns(fh + bc("sx0", inlet + "y = [1.0]\n[nozzle]\nastar = 0.1\n[thrust]\npa = 0.01\nstation = 1\n"));
        check(n.rc == 0 && n.text.find("thrust     pa 0.01") != std::string::npos, "total + y, [nozzle] astar, [thrust] station: the dry-run passes");
        // combscale S1: the ledger is checkpointed (gpu_restart_fv_thrust), the refusal is gone.
        n = dns(fh + bc("sx0", inlet + "y = [1.0]\n[nozzle]\nastar = 0.1\n[thrust]\npa = 0.01\nstation = 1\n") + " --checkpoint-dt 1");
        if (n.rc != 0 || n.text.find("checkpoint every 1") == std::string::npos) std::printf("    rc %d: %s", n.rc, n.text.c_str());
        check(n.rc == 0 && n.text.find("checkpoint every 1") != std::string::npos, "[thrust] with --checkpoint-dt: the dry-run passes");
        refuses(fh + bc("sx1", inlet + "y = [1.0]\n[nozzle]\np = 0.1\nastar = 0.1\n"), 2,
                "--bc: [nozzle].p, t: the flame route starts from the composition IC (docs/nozzle.md \"Reacting\")");
        refuses(fh + bc("sx2", inlet + "y = [1.0]\n[nozzle]\nastar = 0.1\n[thrust]\npa = 0.01\n"), 2,
                "--bc: [thrust]: needs station on dns flame (the burnt stagnation plane, masterplan_combnozzle R2)");
        // combscale S6a: the wedge on the compressible hll flame (W1-W5, docs/nozzle.md "Wedge flame"); every other
        // flame route, frames and --wrinkle keep a refusal by name.
        {
            const std::string wIn = "[patch.outlet]\ntype = \"outflow\"\nexit = \"riemann\"\np = 0.5\ntback = 1\n[patch.wall]\ntype = \"slip\"\n"
                                    "[patch.inlet]\ntype = \"total\"\np0 = 1.01\nt0 = 1.003\ny = [1.0]\n";
            const std::string sides = "[patch.back]\ntype = \"wedge\"\n[patch.front]\ntype = \"wedge\"\n";
            const std::string open = "[patch.inlet]\ntype = \"far\"\ny = [1.0]\n[patch.outlet]\ntype = \"outflow\"\nburnt = true\n[patch.wall]\ntype = "
                                     "\"slip\"\n";
            const std::string lm = "[patch.inlet]\ntype = \"inflow\"\ny = [1.0]\n[patch.outlet]\ntype = \"outflow\"\n[patch.wall]\ntype = \"slip\"\n";
            const std::string fw = "flame --polymesh " + dataW + " --upstream 7 --tt 1" + o + " --bc ";
            n = dns(fw + bc("sx3", wIn + sides) + " --shock hll");
            if (n.rc != 0) std::printf("    rc %d: %s", n.rc, n.text.c_str());
            check(n.rc == 0 && n.text.find("  patch      back: wedge") != std::string::npos && n.text.find("  patch      front: wedge") != std::string::npos,
                  "a wedge mesh on dns flame --shock hll: the dry-run passes");
            check(dns(fw + bc("sx3b", open) + " --shock hll").rc == 0, "the wedge sides left out of the file: the dry-run passes");
            check(dns(fw + bc("sx3", wIn + sides) + " --shock hll --emit stats").rc == 0, "--emit stats on the wedge flame passes");
            refuses(fw + bc("sx3b", open), 2, "a `wedge` mesh on dns flame needs --shock hll (compressible)");
            refuses(fw + bc("sx3c", lm) + " --physics lowmach", 2, "a `wedge` mesh on dns flame needs --shock hll (compressible)");
            refuses(fw + bc("sx3d", lm + sides) + " --physics lowmach", 3, "patch.back: type = \"wedge\": the wedge is the compressible nozzle route's");
            refuses(fw + bc("sx3e", open + sides), 2, "--bc: patch.back.type = \"wedge\" belongs to the nozzle route");
            for (const char* e : {"png", "bin", "shm", "web"})
                refuses(fw + bc("sx3", wIn + sides) + " --shock hll --emit stats --emit " + e, 2,
                        std::format("--emit {}: frames on a wedge flame mesh: not built", e));
            refuses(fw + bc("sx3f", wIn + sides + "[nozzle]\nastar = 0.1\n") + " --shock hll --wrinkle 0.005", 2,
                    "--wrinkle: not built on a wedge (m = 0 radial mode)");
        }
        // combscale S5: [nozzle] cd (the flame's design; dns flow refuses it), --wrinkle on a [nozzle] chamber.
        const std::string noz = inlet + "y = [1.0]\n[nozzle]\nastar = 0.1\n";
        n = dns(fh + bc("cd0", noz + "cd = 0.95\n"));
        check(n.rc == 0 && n.text.find("  nozzle     cd 0.95: the throat's discharge coefficient (the design on A_ref / (cd A*))") !=
                               std::string::npos,
              "[nozzle] cd: the flame dry-run passes and names it");
        check(dns(fh + bc("cd1", noz)).text.find("nozzle     cd") == std::string::npos, "no cd: no dry-run line");
        refuses(fh + bc("cd2", noz + "cd = 1.2\n"), 3, "[nozzle].cd: must be in (0, 1]");
        refuses(fh + bc("cd3", noz + "cd = 0\n"), 3, "[nozzle].cd: must be in (0, 1]");
        {
            std::string cdText = bcText;
            cdText.replace(cdText.find("[nozzle]\n"), 9, "[nozzle]\ncd = 0.9\n");
            refuses(thr + bc("cd4", cdText), 2, "--bc: [nozzle].cd is the reacting nozzle's design (dns flame)");
        }
        n = dns(fh + bc("wr0", noz) + " --wrinkle 0.005");
        check(n.rc == 0 && n.text.find("; --wrinkle 0.005 delta_th: the chamber front at x_f + A cos(pi r/H_c), its lowest transverse mode") !=
                               std::string::npos,
              "--wrinkle on a [nozzle] flame polyMesh: the dry-run passes and names the mode");
        refuses(fh + bc("wr1", noz) + " --wrinkle 0.005 --wrinkle-modes 2", 2,
                "--wrinkle-modes: a [nozzle] chamber seeds its lowest transverse mode only");
        refuses(fh + bc("wr2", noz) + " --wrinkle-modes 2", 2, "--wrinkle-modes: a [nozzle] chamber seeds its lowest transverse mode only");
        refuses(fl + " --shock hll --wrinkle 0.005", 2, "--wrinkle: --wrinkle does not apply with --polymesh: the mesh is the case");
        refuses(fl + " --wrinkle 0.005", 2, "--wrinkle: --wrinkle does not apply with --polymesh: the mesh is the case");
        refuses(fh + bc("wr3", noz) + " --wrinkle -1", 2, "--wrinkle: must be finite and >= 0 (thermal thicknesses)");
        refuses(thr + bc("sx5", "[nozzle]\np = 1\nastar = 1\n[patch.o]\ntype = \"far\"\n"), 3, "[nozzle]: needs t");
        refuses(thr + bc("sx6", "[nozzle]\np = 1\nt = 1\nastar = 1\n[patch.inlet]\ntype = \"total\"\np0 = 1\nt0 = 1\ny = [1.0]\n"), 2,
                "patch.inlet.y belongs to a dns flame file");
    }
    // A regular tetrahedron has no face with an opposite: refused by count, before any GPU or output.
    const std::string tet = std::string(DNS_MESH_DATA) + "/tet_regular";
    refuses("flow --physics compressible --polymesh " + tet + " --bc " + tet + "/bc.toml --shock hll --inviscid --mach 1 --tt 1 --out " +
                (tmp() / "tet").string(),
            3, "mesh: --shock needs a mesh line through every face: 1 of 1 cells have a face with no opposite (tets, triangles; "
               "docs/nozzle.md \"Sensor\", issues.md Z11)");

    std::error_code ec;
    fs::remove_all(tmp(), ec);
    return dnstest::finish();
}
