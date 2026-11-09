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
    refuses(run + bc("t5", "[nozzle]\nq = 1\n[patch.o]\ntype = \"far\"\n"), 3, "[nozzle]: unknown key 'q' (p, t, astar, axis)");
    refuses(run + bc("t6", "[nope]\n[patch.o]\ntype = \"far\"\n"), 3, "unknown top-level key 'nope' (reference, patch, nozzle)");
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
    refuses("flame --polymesh " + dataA + " --bc " + bcA + " --shock hll" + o, 2, "--polymesh is not built with --shock");
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
    std::error_code ec;
    fs::remove_all(tmp(), ec);
    return dnstest::finish();
}
