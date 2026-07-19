// Forced homogeneous isotropic turbulence driver (GPU). Configured in SI
// (air nu, target u'), sized so h = 1.5 eta; runs internally in box units
// (L = 2pi, u' = 1) and reports both. Usage:
//   hit [n] [--urms m/s] [--nu m2/s] [--spin Te] [--stats Te] [--seed s]
//       [--out prefix] [--snap]
#include "dns/diagnostics.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/kernels.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/gpu/timestepper.hpp"
#include "dns/hit_ic.hpp"
#include "dns/units.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numbers>
#include <string>
#include <vector>

namespace {

double parseOr(const char* s, double fallback)
{
    double v{};
    const std::string_view sv(s);
    auto [p, ec] = std::from_chars(sv.begin(), sv.end(), v);
    return ec == std::errc{} && p == sv.end() ? v : fallback;
}

struct Options
{
    int n = 256;
    double urmsSI = 60.0;
    double nuSI = 1.5e-5;
    double spinTe = 10.0;
    double statsTe = 10.0;
    unsigned seed = 1;
    std::string out = "hit";
    bool snapshots = false;
};

Options parseArgs(int argc, char** argv)
{
    Options o;
    if (argc > 1 && argv[1][0] != '-') o.n = int(parseOr(argv[1], o.n));
    for (int a = 1; a < argc; ++a)
    {
        const std::string_view s(argv[a]);
        auto next = [&](double d) { return a + 1 < argc ? parseOr(argv[++a], d) : d; };
        if (s == "--urms") o.urmsSI = next(o.urmsSI);
        else if (s == "--nu") o.nuSI = next(o.nuSI);
        else if (s == "--spin") o.spinTe = next(o.spinTe);
        else if (s == "--stats") o.statsTe = next(o.statsTe);
        else if (s == "--seed") o.seed = unsigned(next(o.seed));
        else if (s == "--snap") o.snapshots = true;
        else if (s == "--out" && a + 1 < argc) o.out = argv[++a];
    }
    return o;
}

void writeSpectrum(const std::string& path, const std::vector<double>& shellE)
{
    std::ofstream f(path);
    f << "k,E\n";
    for (std::size_t s = 1; s < shellE.size(); ++s)
        f << s << ',' << shellE[s] << '\n';
}

void writeSnapshot(const std::string& prefix, const dns::VectorField& U, int n,
                   double tBox, const dns::HitDesign& d)
{
    std::ofstream raw(prefix + ".raw", std::ios::binary);
    auto dump = [&](const dns::Scalar& q) {
        std::vector<float> tmp(q.begin(), q.end());
        raw.write(reinterpret_cast<const char*>(tmp.data()),
                  std::streamsize(tmp.size() * sizeof(float)));
    };
    dump(U.u);
    dump(U.v);
    dump(U.w);
    std::ofstream meta(prefix + ".json");
    meta << "{\"n\":" << n << ",\"tBox\":" << tBox << ",\"tSI\":"
         << d.units().timeToSI(tBox) << ",\"lBoxSI\":" << d.lBoxSI
         << ",\"layout\":\"u,v,w float32 i-fastest MAC\"}\n";
}

} // namespace

int main(int argc, char** argv)
{
    using namespace dns;
    using std::numbers::pi;

    const Options o = parseArgs(argc, argv);
    if (!gpu::deviceAvailable())
    {
        std::fprintf(stderr, "no HIP device\n");
        return 1;
    }

    const auto d = HitDesign::make(o.n, o.nuSI, o.urmsSI);
    const double nu = d.nuBox(), epsF = d.epsBox(), tE = d.tEddyBox();
    std::printf("# HIT n=%d  SI: u'=%g m/s nu=%g  L=%.3e m  eps=%.3e W/kg  "
                "eta=%.3e m  Re_lambda=%.1f  T_e=%.3e s\n",
                o.n, o.urmsSI, o.nuSI, d.lBoxSI, d.epsSI, d.etaSI, d.reLambda,
                d.tEddySI);
    std::printf("# box units: nu=%.6e eps_f=%.6e T_e=%.4f  spin=%g Te stats=%g Te\n",
                nu, epsF, tE, o.spinTe, o.statsTe);

    const auto g = Grid::make(o.n, o.n, o.n, 2 * pi, 2 * pi, 2 * pi);
    auto U = VectorField::zeros(g);
    hitInitialCondition(g, 1.0, o.seed, U);

    auto ts = gpu::TimeStepper::make(g, {nu, 0.9});
    auto D = gpu::DeviceField::make(g.size());
    if (!ts || !D)
    {
        std::fprintf(stderr, "GPU init failed\n");
        return 1;
    }
    (void)gpu::uploadField(U, *D);
    const auto gg = ts->grid();
    auto& red = ts->reduceScratch();

    const LinearForcing forcing{epsF, 1.5 / 5.0}; // kFloor = k_stationary/5
    const double tEnd = (o.spinTe + o.statsTe) * tE, tSpin = o.spinTe * tE;
    const std::string statsCsv = o.out + "_stats.csv";

    double t = 0.0, dt = 0.0;
    long steps = 0;
    int spectrumIdx = 0, snapIdx = 0;
    double nextSpectrum = tSpin, nextSnap = tSpin;
    double epsSum = 0.0, kSum = 0.0, reSum = 0.0, wSum = 0.0;

    while (t < tEnd)
    {
        const auto mean = gpu::meanVelocity(gg, *D, red);
        const double kPrime = gpu::fluctuationEnergy(gg, *D, red);
        const double A = epsF / (2.0 * std::max(kPrime, forcing.kFloor));
        ts->setForcing({A, {mean.x, mean.y, mean.z}});
        if (steps % 100 == 0)
            gpu::subtractMean(gg, gpu::Real(mean.x), gpu::Real(mean.y),
                              gpu::Real(mean.z), *D);

        dt = ts->suggestDt(*D, tEnd - t);
        ts->step(*D, dt);
        t += dt;
        ++steps;

        if (steps % 10 == 0)
        {
            FlowStats fs{};
            fs.kPrime = gpu::fluctuationEnergy(gg, *D, red);
            fs.uPrime = std::sqrt(2.0 * fs.kPrime / 3.0);
            fs.epsVisc = nu * gpu::velocityGradientSq(gg, *D, red);
            if (fs.epsVisc > 0)
            {
                fs.eta = std::pow(nu * nu * nu / fs.epsVisc, 0.25);
                fs.reLambda = fs.uPrime * fs.uPrime
                            * std::sqrt(15.0 / (nu * fs.epsVisc));
                fs.hOverEta = double(g.hx) / fs.eta;
            }
            fs.courantPerDt = gpu::maxCourantPerDt(gg, *D, red);
            fs.maxDiv = gpu::maxAbsDivergence(gg, *D, red);
            (void)appendStatsCsv(statsCsv, t, dt, A, fs);
            if (fs.maxDiv * g.hx > 1e-3)
                std::fprintf(stderr, "WARN t=%.3f: maxDiv*h = %.3e\n", t,
                             fs.maxDiv * g.hx);
            if (t > tSpin)
            {
                epsSum += dt * 10 * fs.epsVisc;
                kSum += dt * 10 * fs.kPrime;
                reSum += dt * 10 * fs.reLambda;
                wSum += dt * 10;
            }
        }

#if DNS_WITH_FFTW
        if (t >= nextSpectrum)
        {
            (void)gpu::downloadField(*D, U);
            writeSpectrum(o.out + "_spec_" + std::to_string(spectrumIdx++) + ".csv",
                          energySpectrum(g, U));
            nextSpectrum += 0.25 * tE;
        }
#endif
        if (o.snapshots && t >= nextSnap)
        {
            (void)gpu::downloadField(*D, U);
            writeSnapshot(o.out + "_snap_" + std::to_string(snapIdx++), U, o.n, t, d);
            nextSnap += tE;
        }
    }

    const double epsMean = epsSum / wSum, kMean = kSum / wSum;
    const double uPrime = std::sqrt(2.0 * kMean / 3.0);
    std::printf("# %ld steps.  Stationary averages over %.1f Te:\n", steps, o.statsTe);
    std::printf("#   eps_visc/eps_f = %.4f   u'/target = %.4f   Re_lambda = %.1f "
                "(design %.1f)\n",
                epsMean / epsF, uPrime, reSum / wSum, d.reLambda);
    std::printf("#   SI: u' = %.2f m/s  eps = %.4e W/kg\n",
                d.units().velocityToSI(uPrime), d.units().epsToSI(epsMean));
    return 0;
}
