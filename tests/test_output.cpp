// RunOutput: the single run-artefact sink. Checks that each Emit kind produces
// exactly the files the apps used to write by hand, that the .json is rewritten
// with the final frame count, that the stats header is verbatim, and that the
// cadences (fdt/pdt/wdt) fire independently.
#include "dns/output.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok)
    {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

std::string slurp(const std::string& p)
{
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

std::size_t fileSize(const std::string& p)
{
    std::error_code ec;
    const auto s = std::filesystem::file_size(p, ec);
    return ec ? 0 : std::size_t(s);
}

dns::RunMeta meta(const dns::Grid& g)
{
    dns::RunMeta m;
    m.nx = g.nx; m.ny = g.ny; m.nz = g.nz;
    m.hx = g.hx; m.hy = g.hy; m.hz = g.hz;
    m.D = 1.0; m.re = 60; m.frameDt = 1.0;
    m.body = "cylinder"; m.farField = "tunnel"; m.physics = "incompressible";
    return m;
}

} // namespace

int main()
{
    using namespace dns;
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path() / "dns_test_output";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    std::filesystem::current_path(tmp);

    const int n = 16;
    const auto g = Grid::make(n, n, 1, 1.0, 1.0, 1.0);
    std::vector<float> vort(g.size(), 0.5f), speed(g.size(), 1.0f);
    const FrameField ff[2]{{"vort", vort.data(), 0, 8, true},
                           {"speed", speed.data(), 0, 1.6f, false}};
    const FrameViews fv{ff};

    // --- 2D: stats + bin + png ---------------------------------------------
    {
        FrameGeometry geo;
        geo.mask.assign(g.size(), 1.0f);
        geo.mask[g.idx(3, 3, 0)] = 0.0f;
        geo.ic = n / 2; geo.jc = n / 2; geo.kc = 0; geo.D = 1.0;

        RunOutput::Config cfg;
        cfg.prefix = "a";
        cfg.kinds = {Emit::Stats, Emit::Bin, Emit::Png};
        cfg.fdt = 1.0;
        cfg.pdt = 2.0;

        auto o = RunOutput::make(cfg, meta(g), g, geo);
        check(o.has_value(), "RunOutput::make succeeds");
        if (!o) return 1;

        check(std::filesystem::exists("a.json"), "json exists at construction");
        check(fileSize("a_mask.bin") == std::size_t(n) * n * 4, "mask.bin size");
        check(std::filesystem::is_directory("png_a"), "png dir created");

        // A prefix with directories keeps the png dir NEXT TO the other
        // outputs -- "outdir/png_b", never a literal "png_outdir/b" under the
        // CWD (measured: a channel run filed its frames inside build/).
        {
            std::filesystem::create_directories("outdir");
            RunOutput::Config c2 = cfg;
            c2.prefix = "outdir/b";
            auto o2 = RunOutput::make(c2, meta(g), g, geo);
            check(o2.has_value(), "RunOutput::make with dir prefix");
            check(std::filesystem::is_directory("outdir/png_b"),
                  "png dir lands beside the prefix");
            check(!std::filesystem::exists("png_outdir"),
                  "no mangled png_<dir> tree");
        }

        const std::string cols[] = {"t", "dt", "K", "cd", "cl",
                                    "cdf", "cfl", "maxDiv", "pcgIters"};
        o->declareSeries("stats", cols);
        const double r[] = {0.0, 1e-3, 2.0, 1.5, 0.0, 0.5, 0.4, 1e-9, 7};
        check(o->row("stats", r).has_value(), "row writes");

        // t = 0, 1, 2, 3: bin every 1.0 => 4 frames; png every 2.0 => 2.
        for (double t = 0; t < 3.5; t += 1.0)
            check(o->frame(t, fv).has_value(), "frame writes");
        check(o->frames() == 4, "four bin frames");
        check(o->flush().has_value(), "flush succeeds");
        check(fileSize("a_vort.bin") == 4 * std::size_t(n) * n * 4,
              "vort.bin has four planes appended");
        check(fileSize("a_speed.bin") == 4 * std::size_t(n) * n * 4,
              "speed.bin has four planes appended");
        check(std::filesystem::exists("png_a/vort_00000.png"), "png 0 written");
        check(std::filesystem::exists("png_a/speed_00001.png"), "png 1 written");
        check(!std::filesystem::exists("png_a/vort_00002.png"),
              "png cadence is independent of the bin cadence");

        const std::string csv = slurp("a_stats.csv");
        check(csv.starts_with("t,dt,K,cd,cl,cdf,cfl,maxDiv,pcgIters\n"),
              "stats header is verbatim");

        o->setMsPerStep(1.25);
    }
    {   // destructor rewrote the json with the final frame count
        const std::string j = slurp("a.json");
        check(j.find("\"frames\":4") != std::string::npos,
              "json rewritten with the final frame count");
        check(j.find("\"msPerStep\":1.25") != std::string::npos,
              "json carries ms/step");
    }

    // --- 3D: volume vs slices ----------------------------------------------
    {
        const auto g3 = Grid::make(8, 8, 8, 1.0, 1.0, 1.0);
        std::vector<float> curl(g3.size(), 1.0f), sp(g3.size(), 2.0f);
        const FrameField f3[2]{{"curl", curl.data(), 0, 15, false},
                               {"speed", sp.data(), 0, 1.6f, false}};
        FrameGeometry geo;
        geo.ic = 4; geo.jc = 4; geo.kc = 4; geo.D = 1.0;

        RunOutput::Config cfg;
        cfg.prefix = "v";
        cfg.kinds = {Emit::Volume};
        cfg.fdt = 1.0;
        auto o = RunOutput::make(cfg, meta(g3), g3, geo);
        check(o.has_value(), "volume RunOutput::make");
        if (o) {
            check(o->frame(0.0, FrameViews{f3}).has_value(), "volume frame");
            check(o->flush().has_value(), "volume flush");
            check(fileSize("v_curl.bin") == g3.size() * 4, "volume is the full grid");
        }

        cfg.prefix = "s";
        cfg.kinds = {Emit::Slices};
        auto o2 = RunOutput::make(cfg, meta(g3), g3, geo);
        if (o2) {
            check(o2->frame(0.0, FrameViews{f3}).has_value(), "slice frame");
            check(o2->flush().has_value(), "slice flush");
            check(fileSize("s_xy_curl.bin") == 8 * 8 * 4, "xy slice size");
            check(fileSize("s_xz_curl.bin") == 8 * 8 * 4, "xz slice size");
            check(fileSize("s_yz_speed.bin") == 8 * 8 * 4, "yz slice size");
        }
    }

    // --- rejections ---------------------------------------------------------
    {
        const auto g3 = Grid::make(8, 8, 8, 1.0, 1.0, 1.0);
        RunOutput::Config cfg;
        cfg.prefix = "p";
        cfg.kinds = {Emit::Png};
        // nz > 1 PNG renders the centre z-plane (subcommands gate whether
        // that plane is meaningful); the sink itself accepts it.
        check(RunOutput::make(cfg, meta(g3), g3, {}).has_value(),
              "PNG output at nz > 1 slices the centre plane");
        cfg.prefix.clear();
        check(!RunOutput::make(cfg, meta(g3), g3, {}).has_value(),
              "an empty prefix is rejected");
    }

    if (failures == 0) std::printf("output: OK\n");
    return failures == 0 ? 0 : 1;
}
