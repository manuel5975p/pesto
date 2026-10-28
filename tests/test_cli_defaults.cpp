// The RESOLVED defaults the CLAUDE.md invariants name, pinned through the real
// parse + validate: a default that drifts silently changes every run that
// does not spell the flag.
//   - compressible flow: --far nscbc, sink 16 amp 1, etaIn 100, etaLat 0.05,
//     beta 0.5, the lambda = u family following etaIn, the Yoo-Im constant;
//   - NO subcommand / physics / body resolves to FarField::Sponge;
//   - the default sink silently stays off where it is undefined (stretch,
//     focal, ogrid, floor) -- absence, not an error;
//   - describe() carries the DNS/LES label, with Cs and the SGS model.
#include "dns/config.hpp"
#include "check.hpp"
#include "cli_test_util.hpp"

#include <cstdio>
#include <optional>
#include <string>
#include <vector>

namespace {

using dnstest::check;
using dnstest::runCli;

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

// The resolved case of `cmd`, or nullopt with the rejection printed and
// counted (every command here is meant to be valid).
std::optional<dns::Case> resolve(const std::string& cmd)
{
    auto c = runCli(split(cmd));
    if (!c)
    {
        std::printf("      -> %s\n", dns::format(c.error()).c_str());
        check(false, "accepts: dns " + cmd);
        return std::nullopt;
    }
    return std::move(*c);
}

bool contains(const std::string& hay, const std::string& needle)
{
    return hay.find(needle) != std::string::npos;
}

} // namespace

int main()
{
    // --- compressible flow: the boundary defaults ----------------------------
    if (auto c = resolve("flow --physics compressible --body cylinder --n 256"))
    {
        std::printf("comp cylinder: far %s sink %d amp %g etaIn %g etaLat %g "
                    "beta %g etaInShear %g abc %s strip %d\n",
                    dns::name(c->far), c->nscbcSink, c->nscbcSinkAmp,
                    c->nscbcEtaIn, c->nscbcEtaLat, c->nscbcBeta,
                    c->nscbcEtaInShear, dns::name(c->nscbcAbc), c->nscbcStrip);
        check(c->far == dns::FarField::Nscbc, "compressible default far = nscbc");
        check(c->nscbcSink == 16, "default sink band = 16 cells");
        check(c->nscbcSinkAmp == 1.0, "default sink amp = 1");
        check(c->nscbcEtaIn == 100.0, "default etaIn = 100");
        check(c->nscbcEtaLat == 0.05, "default etaLat = 0.05");
        check(c->nscbcBeta == 0.5, "default nscbc beta = 0.5");
        check(c->nscbcEtaInShear < 0.0,
              "default lambda = u family rate follows etaIn (< 0)");
        check(c->nscbcAbc == dns::NscbcAbc::YooIm,
              "default transverse coefficient is the Yoo-Im constant");
        check(c->nscbcStrip == 0, "no freestream strip by default");
        check(!c->open, "closed (free-slip) laterals by default");
        check(!c->lesLabel && c->cs == 0.0, "no --les: molecular DNS");
    }
    // --open alone resolves to the measured open-box configuration.
    if (auto c = resolve("flow --physics compressible --body cylinder --open "
                         "--n 256"))
    {
        check(c->open && c->nscbcSink == 16 && c->nscbcEtaLat == 0.05,
              "plain --open = sink 16 + etaLat 0.05");
    }
    if (auto c = resolve("flow --body cylinder --n 256"))
        check(c->far == dns::FarField::Tunnel,
              "incompressible default far = tunnel");

    // --- sponge is never a default -------------------------------------------
    const char* cmds[] = {
        "flow --body cylinder --n 128",
        "flow --body sphere --n 64",
        "flow --body cube --n 64 --nz 64",
        "flow --body lens --n 128",
        "flow --physics compressible --body cylinder --n 128",
        "flow --physics compressible --body sphere --n 64 --nz 64",
        "flow --physics compressible --body cube --n 64 --nz 64",
        "flow --physics compressible --body lens --n 128",
        "flow --physics compressible --body none --n 64",
        "flow --physics compressible --body floor --n 128",
        "flow --physics compressible --body cylinder --open --n 128",
        "flow --physics compressible --body cylinder --focal auto --n 256",
        "flow --physics compressible --body none --stretch 2 --n 64 --ny 64",
        "flow --physics compressible --body cylinder --ogrid --n 128 --ny 66 "
         "--ogrid-far 5",
        "flow --physics compressible --body cylinder --rotor-mach 0.3 "
        "--rotor-axis z --n 128",
        "flow --physics compressible --body cylinder --rotor-mach 0.3 "
        "--rotor-axis z --rotor-frame mrf --n 128",
        "tgv --n 32",
        "hit --n 32",
        "bench --n 32",
        "channel",
        "conv",
        "conv --physics incompressible",
        "flame",
        "flame --nscbc-sink 0",
    };
    for (const char* cmd : cmds)
    {
        auto c = resolve(cmd);
        if (!c) continue;
        std::printf("  %-75s far %s sink %d\n", cmd, dns::name(c->far),
                    c->nscbcSink);
        check(c->far != dns::FarField::Sponge,
              std::string("never resolves to sponge: dns ") + cmd);
    }

    // --- the default sink stays off where it is undefined --------------------
    struct Off { const char* cmd; const char* why; };
    const Off offs[] = {
        {"flow --physics compressible --body none --stretch 2 --n 64 --ny 64",
         "stretch"},
        {"flow --physics compressible --body cylinder --focal auto --n 256",
         "focal"},
        {"flow --physics compressible --body cylinder --ogrid --n 128 --ny 66 "
         "--ogrid-far 5",
         "ogrid"},
        {"flow --physics compressible --body floor --n 128", "floor"},
    };
    for (const Off& o : offs)
        if (auto c = resolve(o.cmd))
        {
            std::printf("  %-8s sink %d far %s\n", o.why, c->nscbcSink,
                        dns::name(c->far));
            check(c->nscbcSink == 0,
                  std::string("default sink silently off on ") + o.why);
        }

    // --- the DNS/LES label, with Cs and the model ----------------------------
    if (auto c = resolve("flow --body cylinder --n 128"))
    {
        const std::string d = c->describe();
        check(contains(d, "(DNS)") && !contains(d, "LES"),
              "no --les: describe() says DNS");
    }
    if (auto c = resolve("flow --body cylinder --n 128 --les 0.17"))
    {
        const std::string d = c->describe();
        std::printf("%s", d.c_str());
        check(c->lesLabel, "--les 0.17 sets the LES label");
        check(contains(d, "Cs=0.17") && contains(d, "(LES smag)"),
              "--les 0.17: describe() states Cs=0.17 and LES smag");
    }
    if (auto c = resolve("flow --body cylinder --n 128 --nz 8 --les 1.35 "
                         "--sgs sigma"))
    {
        const std::string d = c->describe();
        check(contains(d, "Cs=1.35") && contains(d, "(LES sigma)"),
              "--les 1.35 --sgs sigma: describe() states Cs=1.35 and LES sigma");
    }
    if (auto c = resolve("flow --physics compressible --body cylinder --n 128 "
                         "--nz 8 --les 1.35 --sgs sigma"))
    {
        const std::string d = c->describe();
        check(contains(d, "Cs=1.35") && contains(d, "(LES sigma)"),
              "compressible --les 1.35 --sgs sigma: label with Cs and model");
    }

    // --- incompressible placement centres on the INTERIOR -------------------
    // The body sat at y = h + L/2 whatever --ny was: --ny 32 at --n 64 put its
    // centre above the box. The square box keeps the L/2 form bit for bit.
    if (auto c = resolve("flow --body cylinder --n 64"))
        check(c->placement.cy == c->h + c->L / 2.0 + 0.17 * c->h,
              "ny == n: cy is the square-box h + L/2 + 0.17h, bit for bit");
    if (auto c = resolve("flow --body cylinder --n 64 --ny 32"))
    {
        const double h = c->h, r = 0.5 * c->placement.D;
        check(c->placement.cy == h + 16.0 * h + 0.17 * h,
              "--ny 32: cy is the interior centre + 0.17h");
        check(c->placement.cy - r > h && c->placement.cy + r < 33.0 * h,
              "--ny 32: the body lies inside the interior");
    }
    if (auto c = resolve("flow --body sphere --n 64 --nz 32"))
        check(c->placement.cz == c->h + 16.0 * c->h + 0.23 * c->h,
              "sphere --nz 32: cz is the interior centre + 0.23h");

    return dnstest::finish();
}
