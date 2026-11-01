// Checkpoint / restart, host side (dns/checkpoint.hpp; masterplan real_bunsen
// S2): the file format (every record type bitwise, header-only reads, a file
// cut short or of another kind rejected by name, the atomic rewrite), the case
// signature (order- and spelling-free, the excluded flags, the first
// difference named), the validator's rejections by name, and RunOutput's
// resume: a series + frame stream stopped at a resume point, junk appended
// past it, resumed and continued equals the straight output byte for byte.
// Touches the filesystem under its own directory (DNS_TEST_DIR).
#include "dns/checkpoint.hpp"
#include "dns/cli.hpp"
#include "dns/config.hpp"
#include "dns/output.hpp"
#include "check.hpp"
#include "cli_test_util.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#ifndef DNS_TEST_DIR
#error "DNS_TEST_DIR must be defined (the test's scratch directory)"
#endif

namespace {

namespace fs = std::filesystem;
using dnstest::check;

const std::string kDir = DNS_TEST_DIR;

std::string slurp(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

std::expected<dns::RawConfig, dns::cli::CliError>
parseArgs(std::vector<std::string> args)
{
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>("dns"));
    for (auto& a : args) argv.push_back(a.data());
    return dns::cli::parse(int(argv.size()), argv.data());
}

dns::CaseFlags flagsOf(std::vector<std::string> args)
{
    auto r = parseArgs(std::move(args));
    return r ? r->caseFlags : dns::CaseFlags{};
}

void rejects(std::vector<std::string> args, int code, const std::string& flag,
             const std::string& text, const std::string& what)
{
    auto c = dnstest::runCli(std::move(args));
    const bool ok = !c && c.error().code == code && c.error().field == flag &&
                    c.error().message.find(text) != std::string::npos;
    if (!ok && !c)
        std::printf("    got code %d field '%s': %s\n", c.error().code,
                    c.error().field.c_str(), c.error().message.c_str());
    check(ok, what);
}

void fileFormat()
{
    std::printf("file format\n");
    dns::Checkpoint ck;
    const double d[] = {0.1, -0.0, std::numeric_limits<double>::denorm_min(),
                        std::numeric_limits<double>::quiet_NaN(), 1e308};
    const std::int64_t n[] = {-1, 0, std::numeric_limits<std::int64_t>::max()};
    std::vector<float> f(1000);
    for (std::size_t i = 0; i < f.size(); ++i)
        f[i] = std::bit_cast<float>(std::uint32_t(0x3f800000u + 7919u * i));
    f[3] = -0.0f;
    f[4] = std::numeric_limits<float>::quiet_NaN();
    ck.putText("case", "--a 1\n--b\n");
    ck.putF64("d", d);
    ck.putI64("n", n);
    ck.putF32("state.f", f);
    ck.putF64("t", 1.5);
    ck.putText("case", "--a 2\n"); // replaces
    const std::string path = kDir + "/fmt_ckpt.bin";
    check(ck.write(path).has_value(), "write succeeds");
    check(!fs::exists(path + ".tmp"), "no .tmp left behind");

    auto rd = dns::Checkpoint::read(path);
    check(rd.has_value(), "read succeeds");
    if (!rd) return;
    check(rd->text("case").value_or("") == "--a 2\n", "text round-trips (replaced)");
    auto dv = rd->f64("d");
    bool same = dv && dv->size() == 5;
    for (std::size_t i = 0; same && i < 5; ++i)
        same = std::bit_cast<std::uint64_t>((*dv)[i]) == std::bit_cast<std::uint64_t>(d[i]);
    check(same, "f64 round-trips bitwise (-0, denormal, NaN)");
    auto nv = rd->i64("n");
    check(nv && *nv == std::vector<std::int64_t>(std::begin(n), std::end(n)),
          "i64 round-trips");
    std::vector<float> g(f.size());
    auto fe = rd->f32Into("state.f", g.data(), g.size());
    same = fe.has_value();
    for (std::size_t i = 0; same && i < f.size(); ++i)
        same = std::bit_cast<std::uint32_t>(g[i]) == std::bit_cast<std::uint32_t>(f[i]);
    check(same, "f32 round-trips bitwise");
    check(rd->f64Scalar("t").value_or(0) == 1.5, "scalar f64");
    check(!rd->f32Into("state.f", g.data(), 999), "f32 size mismatch rejected");
    auto miss = rd->f64Scalar("nope");
    check(!miss && miss.error().find("'nope'") != std::string::npos,
          "a missing record is an error by name");
    auto wrong = rd->i64("d");
    check(!wrong && wrong.error().find("not i64") != std::string::npos,
          "a type mismatch is an error by name");
    check(rd->names("state.") == std::vector<std::string>{"state.f"}, "names(prefix)");

    auto hd = dns::Checkpoint::read(path, true);
    check(hd && hd->text("case").value_or("") == "--a 2\n" &&
              !hd->f32Into("state.f", g.data(), g.size()),
          "header-only read keeps the facts, skips the f32 payload");

    // Cut short / not a checkpoint.
    const std::string bytes = slurp(path);
    {
        std::ofstream o(kDir + "/cut_ckpt.bin", std::ios::binary);
        o.write(bytes.data(), std::streamsize(bytes.size() - 5));
    }
    auto cut = dns::Checkpoint::read(kDir + "/cut_ckpt.bin");
    check(!cut && cut.error().find("not a dns checkpoint") != std::string::npos,
          "a file cut short is rejected");
    {
        std::ofstream o(kDir + "/junk_ckpt.bin", std::ios::binary);
        o << "t,dt\n1,2\n";
    }
    auto junk = dns::Checkpoint::read(kDir + "/junk_ckpt.bin");
    check(!junk && junk.error().find("bad magic") != std::string::npos,
          "another file is rejected (bad magic)");
    check(!dns::Checkpoint::read(kDir + "/absent_ckpt.bin"), "a missing file is an error");

    // Atomic rewrite: the second write replaces the first whole.
    dns::Checkpoint c2;
    c2.putF64("t", 2.5);
    check(c2.write(path).has_value(), "rewrite succeeds");
    auto r2 = dns::Checkpoint::read(path);
    check(r2 && r2->f64Scalar("t").value_or(0) == 2.5 && !r2->has("case"),
          "rewrite replaces the file");

    // Facts + output resume records.
    dns::Checkpoint c3;
    c3.putText("binary", "abc");
    c3.putText("solver", "flame-compressible");
    c3.putText("case", "--mode bunsen\n--quiescent\n");
    const std::int64_t grid[4] = {10, 20, 1, 2};
    c3.putI64("grid", grid);
    c3.putF64("t", 0.75);
    c3.putI64("steps", 123);
    c3.putI64("restarts", 2);
    dns::OutputResume at;
    at.files = {{"_stats.csv", 100}, {"_temp.bin", 4096}};
    at.frames = 3;
    at.pngIdx = 2;
    at.nextFrame = 0.75;
    at.nextPng = 1.0;
    at.nextWeb = 0.5;
    dns::putOutputResume(c3, at);
    auto fa = dns::checkpointFacts(c3);
    check(fa && fa->nx == 10 && fa->ny == 20 && fa->nY == 2 && fa->t == 0.75 &&
              fa->steps == 123 && fa->restarts == 2 &&
              fa->caseFlags == dns::CaseFlags{{"--mode", "bunsen"}, {"--quiescent", ""}},
          "checkpointFacts reads the header records");
    auto ra = dns::outputResume(c3);
    check(ra && ra->files == at.files && ra->frames == 3 && ra->pngIdx == 2 &&
              ra->nextFrame == 0.75 && ra->nextPng == 1.0 && ra->nextWeb == 0.5,
          "the output resume point round-trips");
}

void signature()
{
    std::printf("case signature\n");
    const auto a = flagsOf({"flame", "--mode", "bunsen", "--jet-speed", "4",
                            "--emit", "stats", "--emit", "bin"});
    const auto b = flagsOf({"flame", "--emit", "bin", "--jet-speed", "4.0",
                            "--emit", "stats", "--mode", "bunsen"});
    check(!a.empty() && a == b, "order and number spelling do not matter");
    check(dns::caseFlagsText(a) == "--emit stats,bin\n--jet-speed 4\n--mode bunsen\n",
          "canonical text: sorted, shortest numbers, enum names, the emit set");
    check(dns::parseCaseFlagsText(dns::caseFlagsText(a)) == a, "text round-trips");
    const auto c = flagsOf({"flame", "--mode", "bunsen", "--jet-speed", "4",
                            "--emit", "stats", "--emit", "bin", "--tt", "30",
                            "--checkpoint-dt", "5", "--wall-limit", "1700",
                            "--restart", "x", "--out", "x", "--quiet",
                            "--backend", "gpu"});
    check(c == a, "--tt, --checkpoint-dt, --wall-limit, --restart, --out, "
                  "--quiet, --backend are not case flags");
    const auto d = flagsOf({"flame", "--mode", "bunsen", "--jet-speed", "5",
                            "--emit", "stats", "--emit", "bin"});
    check(dns::caseFlagsDiff(a, d).value_or("") == "--jet-speed 4 vs 5",
          "a changed value is named: --jet-speed 4 vs 5");
    const auto e = flagsOf({"flame", "--mode", "bunsen", "--jet-speed", "4",
                            "--emit", "stats", "--emit", "bin", "--quiescent"});
    check(dns::caseFlagsDiff(a, e).value_or("") == "--quiescent (absent) vs given",
          "a flag only the run gives is named");
    check(dns::caseFlagsDiff(e, a).value_or("") == "--quiescent given vs (absent)",
          "a flag only the checkpoint gives is named");
    const auto f = flagsOf({"flame", "--equivalence", "1.3", "--sl-law",
                            "onestep", "--mix-phi", "1.2"});
    check(dns::caseFlagsText(f) == "--equivalence 1.3\n--mix-phi 1.2\n--sl-law onestep\n",
          "--equivalence, --sl-law, --mix-phi are case flags");
    check(!dns::caseFlagsDiff(a, a), "equal signatures: no difference");
}

void validation()
{
    std::printf("validation\n");
    using dnstest::runCli;
    auto ok = runCli({"flame", "--checkpoint-dt", "0.5", "--wall-limit", "0",
                      "--tt", "2", "--out", "p"});
    check(ok && ok->checkpointDt == 0.5 && ok->wallLimit == 0 &&
              ok->restartPrefix.empty(),
          "--checkpoint-dt 0.5 --wall-limit 0 resolve");
    check(ok && ok->describe().find("checkpoint every 0.5") != std::string::npos,
          "dry-run names the checkpoint");
    auto none = runCli({"flame", "--tt", "2"});
    check(none && none->checkpointDt == 0 && none->wallLimit < 0 &&
              none->describe().find("checkpoint") == std::string::npos,
          "no checkpoint flag: nothing set, dry-run unchanged");
    rejects({"flame", "--wall-limit", "100"}, 2, "--wall-limit",
            "needs --checkpoint-dt", "--wall-limit without --checkpoint-dt");
    rejects({"flame", "--checkpoint-dt", "0"}, 2, "--checkpoint-dt", "must be > 0",
            "--checkpoint-dt 0");
    rejects({"flame", "--checkpoint-dt", "-1"}, 2, "--checkpoint-dt", "must be > 0",
            "--checkpoint-dt < 0");
    rejects({"flame", "--checkpoint-dt", "1", "--wall-limit", "-5"}, 2,
            "--wall-limit", "must be >= 0", "--wall-limit < 0");
    rejects({"flame", "--restart", "a", "--out", "b"}, 2, "--restart",
            "must equal --out", "--restart != --out");
    rejects({"flame", "--restart", "a"}, 2, "--restart", "must equal --out",
            "--restart != the default --out");
    rejects({"flame", "--checkpoint-dt", "1", "--emit", "web"}, 2, "--emit",
            "web cannot resume", "--emit web with --checkpoint-dt");
    rejects({"flame", "--mode", "chamber", "--checkpoint-dt", "1"}, 2,
            "--checkpoint-dt", "not wired", "--mode chamber");
    rejects({"flame", "--mode", "candle", "--wick", "--checkpoint-dt", "1"}, 2,
            "--checkpoint-dt", "not wired", "--mode candle --wick");
    for (const char* sub : {"flow", "channel", "conv"}) {
        auto r = parseArgs({sub, "--checkpoint-dt", "1"});
        check(!r && r.error().message.find("unknown option") != std::string::npos,
              std::string(sub) + ": --checkpoint-dt is an unknown option (exit 1)");
    }
    // The phase-1.5 fact: --tt must pass the checkpoint's t.
    auto raw = parseArgs({"flame", "--restart", "p", "--out", "p", "--tt", "2"});
    check(raw.has_value(), "restart parses");
    if (raw) {
        raw->restartT = 2.0;
        auto c = dns::validate(*raw);
        check(!c && c.error().code == 2 && c.error().field == "--tt" &&
                  c.error().message.find("not past the checkpoint") != std::string::npos,
              "--tt at the checkpoint's t is exit 2");
        raw->restartT = 1.9;
        auto c2 = dns::validate(*raw);
        check(c2 && c2->restartPrefix == "p", "--tt past the checkpoint's t resolves");
    }
}

// RunOutput: rows + frames 0..9 straight vs 0..4, junk, resume, 5..9.
void outputResume()
{
    std::printf("output resume\n");
    const dns::Grid g{4, 3, 1, 1, 1, 1};
    auto make = [&](const std::string& prefix) {
        dns::RunOutput::Config oc;
        oc.prefix = prefix;
        oc.kinds = {dns::Emit::Stats, dns::Emit::Bin};
        oc.fdt = 1;
        return dns::RunOutput::make(oc, dns::RunMeta{}, g,
                                    dns::FrameGeometry::centred(g, 1));
    };
    std::vector<float> field(g.size());
    const std::string cols[] = {"t", "x"};
    auto step = [&](dns::RunOutput& o, int i) {
        const double row[] = {double(i), 0.1 * i};
        o.row("stats", row);
        for (std::size_t c = 0; c < field.size(); ++c) field[c] = float(i * 100 + c);
        const dns::FrameField ff[1]{{"temp", field.data(), 0, 1, false}};
        (void)o.frame(double(i), ff);
    };
    const std::string a = kDir + "/outA", b = kDir + "/outB";
    {
        auto o = make(a);
        check(o.has_value(), "straight output made");
        if (!o) return;
        (void)o->declareSeries("stats", cols);
        for (int i = 0; i < 10; ++i) step(*o, i);
        (void)o->finish();
    }
    dns::OutputResume at;
    {
        auto o = make(b);
        if (!o) return;
        (void)o->declareSeries("stats", cols);
        for (int i = 0; i < 5; ++i) step(*o, i);
        auto r = o->resumePoint();
        check(r.has_value() && r->files.size() == 2 && r->frames == 5 &&
                  r->nextFrame == 5,
              "resumePoint: two files, 5 frames, next frame at 5");
        if (r) at = *r;
        for (int i = 5; i < 7; ++i) step(*o, i); // rows past the checkpoint
        (void)o->finish();
    }
    {
        std::ofstream j(b + "_stats.csv", std::ios::app);
        j << "junk,row\n";
    }
    {
        auto o = make(b);
        if (!o) return;
        check(o->resumeAt(at).has_value(), "resumeAt truncates");
        (void)o->declareSeries("stats", cols);
        for (int i = 5; i < 10; ++i) step(*o, i);
        o->setCompletion(true, 1);
        (void)o->finish();
    }
    check(slurp(a + "_stats.csv") == slurp(b + "_stats.csv") &&
              !slurp(a + "_stats.csv").empty(),
          "series: resumed == straight byte for byte");
    check(slurp(a + "_temp.bin") == slurp(b + "_temp.bin") &&
              slurp(a + "_temp.bin").size() == 10 * g.size() * sizeof(float),
          "frames: resumed == straight byte for byte");
    check(slurp(b + ".json").find("\"frames\":10,") != std::string::npos &&
              slurp(b + ".json").find("\"complete\":true,\"restarts\":1") !=
                  std::string::npos,
          ".json: frame count continued, complete + restarts");
    check(slurp(a + ".json").find("complete") == std::string::npos,
          ".json without setCompletion has no complete key");
    // A file shorter than its checkpointed length: rows were lost.
    fs::resize_file(b + "_temp.bin", 8);
    auto o = make(b);
    auto bad = o ? o->resumeAt(at) : std::expected<void, std::string>{};
    check(!bad && bad.error().find("fewer than the checkpoint's") != std::string::npos,
          "a shorter output is rejected by name");
}

} // namespace

int main()
{
    std::error_code ec;
    fs::remove_all(kDir, ec);
    fs::create_directories(kDir);
    fileFormat();
    signature();
    validation();
    outputResume();
    return dnstest::finish();
}
