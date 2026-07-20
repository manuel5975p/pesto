// Drives cli::parse + validate over the whole §2.4 migration table of
// masterplan_recompose, plus the rejection cases the old parsers silently
// swallowed (unknown flags became the grid size; atof garbage became a default).
#include "dns/cli.hpp"
#include "dns/config.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what)
{
    if (!ok) { std::printf("FAIL: %s\n", what.c_str()); ++failures; }
}

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
    rejects("flow --physics compressible --body cylinder --dr 4");
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
            "--nscbc-beta 1.0");                       // measured to blow up
    rejects("flow --physics compressible --body cylinder --far nscbc --re 2000");
    rejects("flow --body cylinder --far fringe --n 16"); // clamp() UB guard
    rejects("flow --body cylinder --nz 2");            // nz == 1 or >= 4
    rejects("flow --body cylinder --emit png --nz 8");
    rejects("flow --body cylinder --out \"a\"b");      // JSON-unsafe prefix
    rejects("flow --body lens --aspect 0.5");
    rejects("flow --body cylinder --pitch 10");        // mesh-only flag
    rejects("tgv --emit error --n 64");                // needs --nz 1
    rejects("hit --n 64 --nz 2");
    rejects("flow --body cylinder --cfl 0");
    rejects("flow --body cylinder --tol 0");
    rejects("nonsense --n 4");

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

    if (failures == 0) std::printf("cli: ok\n");
    return failures == 0 ? 0 : 1;
}
