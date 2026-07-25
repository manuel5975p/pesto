// Drives cli::parse + validate over the whole §2.4 migration table of
// masterplan_recompose, plus the rejection cases the old parsers silently
// swallowed (unknown flags became the grid size; atof garbage became a default).
#include "dns/cli.hpp"
#include "dns/config.hpp"
#include "check.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

using dnstest::check;



std::expected<dns::Case, dns::ConfigError> run(std::vector<std::string> args)
{
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>("dns"));
    for (auto& a : args) argv.push_back(a.data());
    auto raw = dns::cli::parse(int(argv.size()), argv.data());
    if (!raw)
        return std::unexpected(dns::ConfigError{1, "cli", raw.error().message});
    return dns::validate(*raw);
}

std::vector<std::string> split(const std::string& s)
{
    std::vector<std::string> out;
    std::string cur;
    for (char ch : s)
    {
        if (ch == ' ') { if (!cur.empty()) out.push_back(cur); cur.clear(); }
        else cur += ch;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

void accepts(const std::string& cmd)
{
    auto c = run(split(cmd));
    if (!c) std::printf("      -> %s\n", dns::format(c.error()).c_str());
    check(c.has_value(), "should accept: dns " + cmd);
}

void rejects(const std::string& cmd)
{
    auto c = run(split(cmd));
    check(!c.has_value(), "should reject: dns " + cmd);
}

} // namespace

int main()
{
    // --- §2.4 migration table ----------------------------------------------
    accepts("tgv --n 64 --nz 1 --nu 0.05 --ft 1.0 --emit error --emit vtk --out o");
    accepts("tgv --n 128 --re 1600 --ft 20");
    accepts("bench --n 256 --steps 20");
    accepts("hit --n 256 --urms 60 --spin 10 --stats 10 --emit snapshot --out h");
    accepts("tgv --n 256 --ft 20 --emit slices --fdt 0.1 --out m");
    accepts("hit --n 256 --emit slices --spin 5 --stats 3 --out m");
    accepts("flow --backend cpu --body cube --far fringe --n 128 --nz 8 --re 300");
    accepts("flow --body cube --far fringe --n 256 --nz 8 --re 300");
    accepts("flow --body sphere --far fringe --n 128 --re 300");
    accepts("flow --body cube --far tunnel --n 256 --nz 8 --re 300");
    accepts("flow --body sphere --far tunnel --n 128 --re 300 --dimples 300");
    accepts("flow --body cylinder --far tunnel --n 1024 --nz 1 --re 60 --dr 8");
    accepts("flow --body cylinder --far tunnel --n 512 --nz 8 --re 60");
    accepts("flow --body cylinder --far fringe --n 512 --re 60");
    accepts("flow --body cylinder --far tunnel --n 512 --nz 1 --les 0.17");
    accepts("flow --body cylinder --far tunnel --n 512 --nz 8 --les 1.35 "
            "--sgs sigma");
    accepts("flow --body cylinder --far tunnel --n 1024 --dimples 40");
    accepts("flow --body lens --aspect 3.5 --far tunnel --n 1024 --re 60");
    accepts("flow --body m.stl --far tunnel --n 96 --re 300 --robust "
            "--emit volume --fdt 0.75");
    accepts("flow --body m.stl --far fringe --n 96");
    accepts("flow --physics compressible --body cylinder --far sponge --n 512 "
            "--re 800 --mach 0.2");
    accepts("flow --physics compressible --body cylinder --far nscbc "
            "--nscbc-beta 0.5 --n 512");
    accepts("flow --physics compressible --body cylinder --far nscbc --open --n 512");
    accepts("flow --physics compressible --body cylinder --far nscbc --open "
            "--pot --n 512");
    accepts("flow --physics compressible --body cylinder --far nscbc --dr 16 "
            "--n 1024 --ny 1024");
    accepts("flow --body cylinder --far tunnel --open --pot --n 512 --nz 1");
    rejects("flow --physics compressible --body cylinder --far sponge --pot");
    rejects("flow --body cylinder --far tunnel --pot"); // pot needs --open
    rejects("flow --physics compressible --body cylinder --dr 3"); // below dr floor 4
    accepts("flow --physics compressible --body cylinder --dr 4 --far nscbc "
            "--n 512"); // D = L/4, smallest allowed compressible box
    accepts("flow --physics compressible --body cylinder --far nscbc "
            "--nscbc-strip 12 --n 1024 --re 2000");
    accepts("flow --physics compressible --body m.stl --far nscbc --open --n 96 "
            "--nz 96 --les 0.17");
    accepts("flow --physics compressible --body m.stl --far nscbc --open --n 128 "
            "--emit slices");

    // --- rejections the old parsers swallowed ------------------------------
    rejects("flow --body cylinder --oepn");            // unknown flag was the grid size
    rejects("flow --body cylinder --n abc");           // atof garbage
    rejects("flow --body cylinder --n");               // missing value
    rejects("flow");                                   // --body required
    rejects("flow --body banana");
    rejects("flow --body cylinder 512");               // bare positional
    rejects("flow --body cylinder --far sponge");      // compressible far field
    rejects("flow --physics compressible --body cylinder --far tunnel");
    accepts("flow --body cylinder --far tunnel --open --n 512 --nz 1");
    accepts("flow --body sphere --far tunnel --open --n 96");
    accepts("flow --body cylinder --les 0.17 --nz 8 --n 512"); // 3D LES
    rejects("flow --body cylinder --far fringe --open");
    rejects("flow --body cylinder --far periodic --open");
    rejects("flow --body cylinder --mach 0.2");        // incompressible + --mach
    rejects("flow --physics compressible --body cylinder --far nscbc "
            "--nscbc-beta 1.0");                       // measured to diverge
    rejects("flow --physics compressible --body cylinder --far nscbc --re 2000");
    rejects("flow --body cylinder --far fringe --n 16"); // clamp() UB guard
    rejects("flow --body cylinder --nz 2");            // nz == 1 or >= 4
    // png at nz > 1 renders the centre z-plane (channel precedent) -- allowed.
    rejects("flow --body cylinder --out \"a\"b");      // JSON-unsafe prefix
    rejects("flow --body lens --aspect 0.5");
    rejects("flow --body cylinder --pitch 10");        // mesh-only flag
    rejects("tgv --emit error --n 64");                // needs --nz 1
    rejects("hit --n 64 --nz 2");
    rejects("flow --body cylinder --cfl 0");
    rejects("flow --body cylinder --tol 0");
    rejects("nonsense --n 4");

    // Body-free box, y-stretch (curvilinear) and no-slip walls: the scope is
    // validator-enforced -- stretch needs compressible + --body none + --far
    // nscbc (the default; its terms evaluate the PointRemap) or an EXPLICIT
    // --far sponge (never silently swapped), no --open. LES is allowed.
    accepts("flow --physics compressible --body none --far sponge --n 128 "
            "--ny 128 --nz 1");
    accepts("flow --physics compressible --body none --far sponge --noslip "
            "--stretch 2 --n 128 --ny 128 --nz 1");
    accepts("flow --physics compressible --body none --far nscbc --noslip "
            "--stretch 2 --n 128 --ny 128 --nz 1");
    accepts("flow --physics compressible --body none --stretch 2 --n 128 "
            "--ny 128 --nz 1");                        // default far IS nscbc
    accepts("flow --physics compressible --body cylinder --far nscbc --noslip "
            "--n 512");
    rejects("flow --body none --n 128");               // none is compressible-only
    rejects("flow --physics compressible --body none --far sponge --stretch 5");
    rejects("flow --physics compressible --body none --far periodic "
            "--stretch 2 --n 128");                    // periodic has no y walls
    rejects("flow --physics compressible --body cylinder --far sponge "
            "--stretch 2 --n 512");                    // immersed body + stretch
    accepts("flow --physics compressible --body none --far sponge --stretch 2 "
            "--les 0.17 --n 128 --ny 128 --nz 1");     // LES rides the curv path
    rejects("flow --physics compressible --body none --far sponge --stretch 2 "
            "--open --n 128");                         // open defeats the walls
    rejects("flow --body cylinder --noslip --n 512");  // noslip is compressible-only

    // Wall-mounted fence: incompressible tunnel only, needs the floor wall,
    // and D >= 32 cells (the blocker itself is D/8 thick).
    accepts("flow --body fence --dr 10 --re 2000 --n 1024");
    accepts("flow --body fence --dr 8 --n 512");
    rejects("flow --body fence --physics compressible --n 512");
    rejects("flow --body fence --far periodic --n 1024");
    rejects("flow --body fence --far fringe --n 1024");
    rejects("flow --body fence --open --n 1024");
    rejects("flow --body fence --dr 10 --n 256");      // D/8 under 4 cells

    // Raised floor: same floor-mounted family rules, but no thickness bound
    // (it has no thin dimension).
    accepts("flow --body floor --dr 10 --re 2000 --n 1024");
    accepts("flow --body floor --dr 10 --n 256");
    accepts("flow --body floor --physics compressible --n 512"); // ghost floor
    accepts("flow --body floor --physics compressible --bumps 16 --dr 10 "
            "--re 10000 --nscbc-strip 16 --n 1024 --ny 1024");
    rejects("flow --body floor --physics compressible --far periodic --n 512");
    rejects("flow --body fence --physics compressible --n 512"); // fence stays
    rejects("flow --body floor --far periodic --n 1024");
    rejects("flow --body floor --open --n 1024");
    accepts("flow --body floor --bumps 10 --dr 10 --re 5000 --n 2048");
    accepts("flow --body floor --bumps 64 --dr 10 --re 10000 --n 2048");
    rejects("flow --body cylinder --bumps 4 --n 512");   // spikes are floor-only
    rejects("flow --body floor --bumps 128 --dr 10 --n 1024"); // pitch under 16h
    rejects("flow --physics compressible --body none --far nscbc --pot --n 128");
    {
        auto c = run(split("flow --physics compressible --body none --far sponge "
                           "--stretch 2 --n 128 --ny 128 --nz 1"));
        check(c && c->stretchBeta == 2.0, "--stretch reaches the case");
        check(c && c->stretchH1 > 0 && c->stretchH1 < c->grid.hy,
              "resolved first cell is compressed");
        check(c && c->stretchGrowth > 1.0 && c->stretchGrowth < 1.2,
              "growth ratio is bounded (~5%/cell at beta 2, ny 128)");
    }

    // Focal clustering: same scope as --stretch (compressible, nscbc or
    // explicit sponge; LES allowed), plus its own geometry rules -- the
    // blend must clear the domain faces so the map is the exact identity
    // there, and the discrete growth ratio is capped (the central flux has
    // no dissipation to absorb a sharp coarse/fine step).
    accepts("flow --physics compressible --body none --focal 2,4,2,4,0,0 "
            "--n 128 --ny 128 --nz 1");
    accepts("flow --physics compressible --body none --far sponge "
            "--focal 2,4,0,0,0,0 --n 128 --ny 128 --nz 1");   // x only
    accepts("flow --physics compressible --body cylinder --focal auto "
            "--n 256 --nz 1");
    accepts("flow --physics compressible --body sphere --focal auto "
            "--focal-blend 0.8 --n 128 --nz 64");
    rejects("flow --body none --focal 2,4,2,4,0,0 --n 128");  // incompressible
    rejects("flow --physics compressible --body none --focal auto --n 128");
    rejects("flow --physics compressible --body floor --focal auto --n 512");
    rejects("flow --physics compressible --body cylinder --focal 0.5,1.0,2.9,3.4"
            ",0,0 --n 256 --nz 1");   // plateau misses the body + guard band
    rejects("flow --physics compressible --body none --focal 2,4,2,4,0,0 "
            "--stretch 2 --n 128");                     // two curvilinear meshes
    accepts("flow --physics compressible --body none --focal 2,4,2,4,0,0 "
            "--les 0.17 --n 128 --ny 128 --nz 1");      // LES rides the curv path
    accepts("flow --physics compressible --body cube --focal auto "
            "--focal-blend 0.8 --re 2000 --les 1.35 --sgs sigma "
            "--nscbc-strip 4 --n 128 --nz 128");
    rejects("flow --physics compressible --body none --focal 2,4,2,4,0,0 "
            "--open --n 128");
    rejects("flow --physics compressible --body none --far periodic "
            "--focal 2,4,2,4,0,0 --n 128");
    rejects("flow --physics compressible --body none --focal 0.1,6.2,2,4,0,0 "
            "--n 128");                                 // blend hits the faces
    rejects("flow --physics compressible --body none --focal 2,4,2,4,0,0 "
            "--focal-ratio 1 --n 128");                 // ratio must exceed 1
    rejects("flow --physics compressible --body none --focal 2,4,2,4,0,0 "
            "--focal-ratio 20 --n 128");
    rejects("flow --physics compressible --body none --focal 2,4,2,4,0,0 "
            "--focal-blend 0.02 --n 128");              // growth cap
    rejects("flow --physics compressible --body none --focal 0,0,0,0,0,0 "
            "--n 128");                                 // every axis degenerate
    rejects("flow --physics compressible --body none --focal 2,4,2 --n 128");
    rejects("flow --physics compressible --body none --focal auto --n 128");
    rejects("flow --physics compressible --body none --focal-ratio 4 --n 128");
    rejects("flow --physics compressible --body none --focal 2,4,2,4,2,4 "
            "--n 128 --nz 1");                          // z plateau at nz == 1
    {
        auto c = run(split("flow --physics compressible --body none "
                           "--focal 2,4,2,4,0,0 --n 128 --ny 128 --nz 1"));
        check(c && c->focalActive, "--focal reaches the case");
        check(c && c->focalHFine[0] < c->grid.hx
                && c->focalHCoarse[0] > c->grid.hx,
              "plateau cells shrink, far cells grow");
        check(c && std::abs(c->focalHCoarse[0] / c->focalHFine[0] - 4.0) < 1e-9,
              "resolved cell-size ratio is exactly --focal-ratio");
        check(c && c->focalGrowth > 1.0 && c->focalGrowth <= 1.25,
              "discrete growth ratio is bounded");
        check(c && c->focalDtFactor > 0 && c->focalDtFactor < 1,
              "dt factor reports the clustering cost");
        auto d = run(split("flow --physics compressible --body none "
                           "--focal 2,4,0,0,0,0 --n 128 --ny 128 --nz 1"));
        check(d && d->focalHFine[1] == d->grid.hy,
              "a degenerate pair leaves that axis uniform");
    }

    // --nz is INTERIOR cells everywhere: the ghost ring is added internally.
    {
        auto c = run(split("flow --body cylinder --n 64 --nz 8"));
        check(c && c->grid.nz == 10, "--nz 8 allocates the ghost ring internally");
        check(c && c->nzInterior == 8, "--nz 8 reports 8 interior cells");
        auto d = run(split("flow --body cylinder --n 64 --nz 1"));
        check(d && d->grid.nz == 1, "--nz 1 is native 2D, no z ghosts");
    }
    // The two grid conventions stay separate (masterplan §6.12).
    {
        auto i = run(split("flow --body cylinder --n 512"));
        auto k = run(split("flow --physics compressible --body cylinder --n 512"));
        // Incompressible carries a ghost ring (nx = n+2); compressible does not.
        check(i && i->grid.nx == 514, "incompressible --n 512 -> 514 with ghosts");
        check(k && k->grid.nx == 512, "compressible --n is the interior extent");
    }

    // --cut-flux scope. The compressible rejection is the invariant the
    // deleted compressible aperture path relies on: it DIVERGES near the body
    // (KG split flux loses its KE property under per-face area weighting,
    // measured twice 2026-07-23). If this rejection is ever relaxed, the
    // implementation has to come back with it.
    {
        auto ok = run(split("flow --body cylinder --cut-flux --n 64 --nz 1"));
        check(ok.has_value(), "--cut-flux is accepted incompressibly");
        auto comp = run(split("flow --physics compressible --body cylinder "
                              "--cut-flux --n 64 --nz 1"));
        check(!comp && comp.error().field == "--cut-flux",
              "--cut-flux is REJECTED under --physics compressible");
        // (--body none is compressible-only, so the "no body" branch of the
        // --cut-flux guard is unreachable from the CLI today; the compressible
        // rejection above fires first.)
    }

    // --penalize scope: the three constraints at config.cpp's penalize block.
    {
        auto ok = run(split("flow --body cylinder --penalize 0.01 --n 64 --nz 1"));
        check(ok.has_value(), "--penalize is accepted with a body");
        auto neg = run(split("flow --body cylinder --penalize -1 --n 64 --nz 1"));
        check(!neg && neg.error().field == "--penalize",
              "--penalize must be finite and >= 0");
        auto nobody = run(split("flow --physics compressible --body none "
                                "--penalize 0.01 --n 64 --nz 1"));
        check(!nobody && nobody.error().field == "--penalize",
              "--penalize without a body is rejected");
        auto str = run(split("flow --physics compressible --body cylinder "
                             "--penalize 0.01 --stretch 2 --n 64 --nz 1"));
        check(!str.has_value(), "--penalize with --stretch is rejected");
    }

    if (dnstest::gFailures == 0) std::printf("cli: ok\n");
    return dnstest::finish();
}
