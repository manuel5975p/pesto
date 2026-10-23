// `dns prop` through the real binary: the exit-code contract (0 ok, 1 CLI,
// 2 config, 3 input, 4 output -- and never 5, it is host-only), --dry-run
// reading no file, and the CSV the run writes. GPU-free by construction, so
// it runs (and must pass) on a machine without a device.

#include "check.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <sys/wait.h>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif
#ifndef DNS_PROP_DATA
#error "DNS_PROP_DATA must be defined (tests/data/prop)"
#endif

namespace {

namespace fs = std::filesystem;
using dnstest::check;

int run(const std::string& args)
{
    const std::string cmd = "\"" DNS_BIN "\" prop " + args + " > /dev/null 2>&1";
    const int rc = std::system(cmd.c_str());
    if (rc == -1 || !WIFEXITED(rc)) return -1;
    return WEXITSTATUS(rc);
}

void expect(const std::string& args, int want, const std::string& what)
{
    const int got = run(args);
    check(got == want, what + " (want " + std::to_string(want) + ", got "
                           + std::to_string(got) + ")");
}

std::string data(const std::string& f) { return std::string(DNS_PROP_DATA) + "/" + f; }

void write(const fs::path& p, const std::string& text) { std::ofstream(p) << text; }

std::vector<std::string> lines(const fs::path& p)
{
    std::ifstream f(p);
    std::vector<std::string> out;
    for (std::string l; std::getline(f, l);)
        if (!l.empty()) out.push_back(l);
    return out;
}

} // namespace

int main()
{
    const fs::path dir = fs::temp_directory_path() / "dns_prop_cli";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string pf = data("t4412_planform.csv");
    const std::string ok = "--blades 2 --planform " + pf + " --polar thin --thin-alpha0 -4.15";

    // --- 0 ----------------------------------------------------------------------
    expect("--help", 0, "prop --help");
    expect("--planform /nonexistent/p.csv --polar /nonexistent/q.csv --dry-run", 0,
           "--dry-run validates without reading the input files");
    expect("--method liftline --selftest", 0, "the lifting-line self-test passes");
    {
        const fs::path out = dir / "bemt.csv";
        expect(ok + " --j 0:0.9:0.1 --out " + out.string(), 0, "a BEMT curve runs");
        const auto l = lines(out);
        check(l.size() == 11 && l[0] == "J,CT,CP,CQ,eta,T_N,Q_Nm,P_W,Re75",
              "BEMT CSV: header and one row per J");
        double ct0 = 0;
        check(l.size() > 1 && std::sscanf(l[1].c_str(), "0,%lf", &ct0) == 1 && ct0 > 0.1,
              "BEMT CSV: positive static CT");
    }
    {
        const fs::path out = dir / "ll.csv";
        expect("--method liftline --panels 12 " + ok + " --j 0.3,0.6 --out " + out.string(),
               0, "a lifting-line curve runs");
        const auto l = lines(out);
        check(l.size() == 3 && l[0] == "J,CT,CP,CQ,eta,T_N,Q_Nm,P_W,Re75,converged"
                  && l[1].ends_with(",1") && l[2].ends_with(",1"),
              "lifting-line CSV: header with converged, both J converged");
    }
    expect(ok + " --j 0.5 --dump-span 0.5 --quiet", 0, "--dump-span and --quiet run");

    // --- 1: CLI -------------------------------------------------------------------
    expect(ok + " --oepn", 1, "an unknown flag");
    expect(ok + " --backend gpu", 1, "--backend is not a prop flag (never a GPU)");
    expect(ok + " --rpm 6k", 1, "a non-numeric value");

    // --- 2: config ----------------------------------------------------------------
    expect("--polar thin --dry-run", 2, "--planform is required");
    expect("--planform " + pf + " --dry-run", 2, "--polar is required");
    expect(ok + " --j -0.1", 2, "J < 0");
    expect(ok + " --panels 20", 2, "a lifting-line flag with --method bemt");

    // --- 3: input -----------------------------------------------------------------
    expect("--planform /nonexistent/p.csv --polar thin", 3, "a planform that does not exist");
    {
        const fs::path p = dir / "hdr.csv";
        write(p, "r,c,beta\n0.2,0.1,20\n1,0.1,10\n");
        expect("--planform " + p.string() + " --polar thin", 3, "a planform with a bad header");
        write(p, "r_R,c_R,beta_deg\n0.5,0.1,20\n0.4,0.1,10\n");
        expect("--planform " + p.string() + " --polar thin", 3, "a non-monotonic r_R");
        write(p, "r_R,c_R,beta_deg\n0.2,0.1,20\n1,abc,10\n");
        expect("--planform " + p.string() + " --polar thin", 3, "a non-numeric planform cell");
    }
    {
        const fs::path q = dir / "polar.csv";
        write(q, "re,alpha_deg,cl,cd\n1e5,-90,0,1\n1e5,90,0,1\n");
        expect("--planform " + pf + " --polar " + q.string(), 3,
               "a polar not spanning -180..180 deg");
        expect("--planform " + pf + " --polar /nonexistent/q.csv", 3,
               "a polar that does not exist");
    }

    // --- 4: output ----------------------------------------------------------------
    expect(ok + " --out /proc/nonexistent/x.csv", 4, "an unwritable --out");
    if (fs::exists("/dev/full"))
        expect(ok + " --out /dev/full", 4, "a write that fails (full disk)");

    fs::remove_all(dir);
    return dnstest::finish();
}
