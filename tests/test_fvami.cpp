// dns/fvami against OpenFOAM v2606's AMIInterpolation (stage 3c). The
// references are tools/fv/foamAmiDump dumps (tools/fv/run_ami_foam.sh) of the
// tools/fv/make_ami_cases.py meshes, at rest and with the cellZone rotor turned:
//   A1  addressing: srcAddress / tgtAddress, the same lists in the same order
//   A2  srcMagSf / tgtMagSf, srcWeightsSum / tgtWeightsSum, srcWeights /
//       tgtWeights: BIT FOR BIT
//   A3  interpolateToSource / interpolateToTarget of f(j) = 1 + j/4 + j^2/512:
//       bit for bit
//   A4  (dumps with -fv) the cyclicAMIFvPatch geometry from the dumped Cf, Cn,
//       nf of both patches and this file's restatement of
//       cyclicAMIFvPatch::makeWeights / delta() and basicFvGeometryScheme:
//       weights, delta, deltaCoeffs, nonOrthDeltaCoeffs,
//       nonOrthCorrectionVectors, the interpolated x + 2 y + 3 z: bit for bit
//   E1  errors: a non-owner call, a non-AMI patch
// usage:
//   test_fvami                         the checked-in set (DNS_FV_DATA/ami)
//   test_fvami CASEDIR DUMP [DUMP...]  one case (CASEDIR/constant/polyMesh)
//   test_fvami --mesh POLYMESHDIR [DUMP] [--precision N] [--repeat N]
//                                      every AMI pair of a mesh: time, the
//                                      log's sum(weights) lines, uncovered
//                                      faces; compared when a dump is given
#include "dns/fvami.hpp"
#include "dns/polymesh.hpp"

#include "check.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace dns;
using dnstest::check;
namespace fs = std::filesystem;

namespace {

struct DumpPatch // the -fv section of one patch
{
    std::string name;
    std::vector<double> Cf, Cn, nf, weights, delta, deltaCoeffs, nonOrthDeltaCoeffs, corr, lin;
};

struct DumpPair
{
    std::string src, tgt;
    FoamAmiSide s, t;
    std::vector<double> toSrc, toTgt;
    std::vector<DumpPatch> fv;
};

struct Dump
{
    int nPoints = 0;
    std::uint64_t hash = 0;
    std::string zone = "-";
    double angle = 0, axis[3] = {0, 0, 1}, origin[3] = {0, 0, 0};
    std::vector<DumpPair> pairs;
};

// Reads a foamAmiDump file; false + err on a malformed one.
bool readDump(const fs::path& file, Dump& d, std::string& err)
{
    std::ifstream in(file);
    if (!in) {
        err = "cannot open " + file.string();
        return false;
    }
    std::string tok;
    const auto num = [&](double& v) {
        std::string s;
        if (!(in >> s)) return false;
        char* end = nullptr;
        v = std::strtod(s.c_str(), &end);
        return end && *end == 0;
    };
    const auto scalars = [&](const char* name, std::vector<double>& v, int nComp) {
        int n = 0;
        if (!(in >> tok >> n) || tok != name) return false;
        v.resize(std::size_t(n) * std::size_t(nComp));
        for (double& x : v)
            if (!num(x)) return false;
        return true;
    };
    const auto lists = [&](const char* name, FoamAmiSide& S, bool weights) {
        int n = 0;
        if (!(in >> tok >> n) || tok != name) return false;
        if (!weights) S.start.assign(std::size_t(n) + 1, 0);
        for (int f = 0; f < n; ++f) {
            int k = 0;
            if (!(in >> k)) return false;
            if (!weights) S.start[f + 1] = S.start[f] + k;
            for (int i = 0; i < k; ++i) {
                if (weights) {
                    double w = 0;
                    if (!num(w)) return false;
                    S.w.push_back(w);
                } else {
                    int a = 0;
                    if (!(in >> a)) return false;
                    S.addr.push_back(a);
                }
            }
        }
        return true;
    };
    int version = 0;
    std::string hash;
    if (!(in >> tok >> version) || tok != "foamAmiDump" || version != 1) return err = "not a foamAmiDump 1 file", false;
    if (!(in >> tok >> d.nPoints >> tok >> hash)) return err = "points line", false;
    d.hash = std::strtoull(hash.c_str(), nullptr, 16);
    if (!(in >> tok >> d.zone) || tok != "rotate" || !num(d.angle) || !num(d.axis[0]) || !num(d.axis[1])
        || !num(d.axis[2]) || !num(d.origin[0]) || !num(d.origin[1]) || !num(d.origin[2]))
        return err = "rotate line", false;
    while (in >> tok) {
        if (tok == "end") return true;
        if (tok == "pair") {
            DumpPair p;
            int ns = 0, nt = 0;
            if (!(in >> p.src >> p.tgt >> ns >> nt)) return err = "pair line", false;
            if (!scalars("srcMagSf", p.s.magSf, 1) || !scalars("srcWeightsSum", p.s.wSum, 1)
                || !lists("srcAddress", p.s, false) || !lists("srcWeights", p.s, true)
                || !scalars("tgtMagSf", p.t.magSf, 1) || !scalars("tgtWeightsSum", p.t.wSum, 1)
                || !lists("tgtAddress", p.t, false) || !lists("tgtWeights", p.t, true)
                || !scalars("interpolateToSource", p.toSrc, 1) || !scalars("interpolateToTarget", p.toTgt, 1))
                return err = "pair " + p.src + ": malformed at " + tok, false;
            d.pairs.push_back(std::move(p));
            continue;
        }
        if (tok == "fvPatch" && !d.pairs.empty()) {
            DumpPatch q;
            int n = 0;
            if (!(in >> q.name >> n)) return err = "fvPatch line", false;
            if (!scalars("Cf", q.Cf, 3) || !scalars("Cn", q.Cn, 3) || !scalars("nf", q.nf, 3)
                || !scalars("weights", q.weights, 1) || !scalars("delta", q.delta, 3)
                || !scalars("deltaCoeffs", q.deltaCoeffs, 1) || !scalars("nonOrthDeltaCoeffs", q.nonOrthDeltaCoeffs, 1)
                || !scalars("nonOrthCorrectionVectors", q.corr, 3) || !scalars("interpolateLinear", q.lin, 1))
                return err = "fvPatch " + q.name + ": malformed at " + tok, false;
            d.pairs.back().fv.push_back(std::move(q));
            continue;
        }
        return err = "unexpected token " + tok, false;
    }
    return err = "no end line", false;
}

// foamAmiDump's rotation of the zone's points, scalar by scalar.
bool rotateZone(const PolyMesh& pm, const Dump& d, std::vector<double>& x, std::vector<double>& y,
                std::vector<double>& z)
{
    if (d.zone == "-") return true;
    const auto it = std::ranges::find(pm.man.cellZones, d.zone);
    if (it == pm.man.cellZones.end()) return false;
    const auto& cells = pm.zoneCells[std::size_t(it - pm.man.cellZones.begin())];
    std::vector<char> inZone(std::size_t(pm.man.nCells), 0), move(x.size(), 0);
    for (const std::int32_t c : cells) inZone[c] = 1;
    for (std::int64_t f = 0; f < pm.man.nFaces; ++f) {
        const bool in = inZone[pm.owner[f]] || (f < pm.man.nInternalFaces && inZone[pm.neighbour[f]]);
        if (!in) continue;
        for (std::int32_t k = pm.faceStart[f]; k < pm.faceStart[f + 1]; ++k) move[pm.faceNodes[k]] = 1;
    }
    const double ax = d.axis[0], ay = d.axis[1], az = d.axis[2];
    const double m = std::sqrt(ax * ax + ay * ay + az * az);
    const double kx = ax / m, ky = ay / m, kz = az / m;
    const double c = std::cos(d.angle), s = std::sin(d.angle);
    const double ox = d.origin[0], oy = d.origin[1], oz = d.origin[2];
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (!move[i]) continue;
        const double dx = x[i] - ox, dy = y[i] - oy, dz = z[i] - oz;
        const double kd = kx * dx + ky * dy + kz * dz;
        const double cx = ky * dz - kz * dy, cy = kz * dx - kx * dz, cz = kx * dy - ky * dx;
        const double rx = dx * c + cx * s + kx * kd * (1.0 - c);
        const double ry = dy * c + cy * s + ky * kd * (1.0 - c);
        const double rz = dz * c + cz * s + kz * kd * (1.0 - c);
        x[i] = ox + rx;
        y[i] = oy + ry;
        z[i] = oz + rz;
    }
    return true;
}

std::uint64_t pointHash(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z)
{
    std::uint64_t h = 1469598103934665603ULL;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double xyz[3] = {x[i], y[i], z[i]};
        unsigned char b[sizeof(xyz)];
        std::memcpy(b, xyz, sizeof(xyz));
        for (const unsigned char ch : b) {
            h ^= ch;
            h *= 1099511628211ULL;
        }
    }
    return h;
}

struct Diff
{
    double abs = 0, rel = 0;
    std::size_t nBits = 0, n = 0;
    bool sized = true;
};
Diff diff(const std::vector<double>& a, const std::vector<double>& b)
{
    Diff d;
    d.n = a.size();
    if (a.size() != b.size()) {
        d.sized = false;
        return d;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(double)) == 0) continue;
        ++d.nBits;
        const double e = std::fabs(a[i] - b[i]);
        d.abs = std::max(d.abs, e);
        d.rel = std::max(d.rel, e / std::max(std::fabs(b[i]), 1e-300));
    }
    return d;
}
// One compared quantity: PASS = bit for bit.
void same(const std::string& what, const std::vector<double>& mine, const std::vector<double>& foam, Diff* worst = nullptr)
{
    const Diff d = diff(mine, foam);
    if (worst && (d.rel > worst->rel || !d.sized)) *worst = d;
    check(d.sized && d.nBits == 0, std::format("{}: {} values, {} differ in bits, max |d| {:.3g}, max rel {:.3g}{}", what,
                                               d.n, d.nBits, d.abs, d.rel, d.sized ? "" : " SIZE MISMATCH"));
}

int patchIndex(const PolyMesh& pm, const std::string& name)
{
    for (std::size_t i = 0; i < pm.man.patches.size(); ++i)
        if (pm.man.patches[i].name == name) return int(i);
    return -1;
}

// cyclicAMIFvPatch::makeWeights, delta() and basicFvGeometryScheme's boundary
// expressions for patch `me` (side S of the AMI) against `nb`.
void checkFvPatch(const std::string& tag, const DumpPatch& me, const DumpPatch& nb, const FoamAmiSide& S)
{
    const std::size_t n = me.weights.size(), nn = nb.weights.size();
    std::vector<double> dMe(3 * n), dNb(3 * nn), nd(nn), lin(nn);
    for (std::size_t i = 0; i < 3 * n; ++i) dMe[i] = me.Cf[i] - me.Cn[i]; // coupledFvPatch::delta()
    for (std::size_t i = 0; i < 3 * nn; ++i) dNb[i] = nb.Cf[i] - nb.Cn[i];
    for (std::size_t j = 0; j < nn; ++j) {
        nd[j] = nb.nf[3 * j] * dNb[3 * j] + nb.nf[3 * j + 1] * dNb[3 * j + 1] + nb.nf[3 * j + 2] * dNb[3 * j + 2];
        lin[j] = nb.Cf[3 * j] + 2.0 * nb.Cf[3 * j + 1] + 3.0 * nb.Cf[3 * j + 2];
    }
    const std::vector<double> nbrDeltas = foamAmiInterpolate(S, nd), nbrPatchD = foamAmiInterpolate(S, dNb, 3);
    std::vector<double> w(n), delta(3 * n), dc(n), dcNO(n), corr(3 * n);
    for (std::size_t f = 0; f < n; ++f) {
        const double* nf = &me.nf[3 * f];
        const double di = std::fabs(nf[0] * dMe[3 * f] + nf[1] * dMe[3 * f + 1] + nf[2] * dMe[3 * f + 2]);
        const double dni = std::fabs(nbrDeltas[f]);
        w[f] = dni / (di + dni);
        for (int c = 0; c < 3; ++c) delta[3 * f + c] = dMe[3 * f + c] - nbrPatchD[3 * f + c];
        const double* dl = &delta[3 * f];
        const double magD = std::sqrt(dl[0] * dl[0] + dl[1] * dl[1] + dl[2] * dl[2]);
        dc[f] = 1.0 / magD;
        dcNO[f] = 1.0 / std::max(nf[0] * dl[0] + nf[1] * dl[1] + nf[2] * dl[2], 0.05 * magD);
        for (int c = 0; c < 3; ++c) corr[3 * f + c] = nf[c] - dl[c] * dcNO[f];
    }
    same(tag + " fvPatch weights()", w, me.weights);
    same(tag + " fvPatch delta()", delta, me.delta);
    same(tag + " deltaCoeffs", dc, me.deltaCoeffs);
    same(tag + " nonOrthDeltaCoeffs", dcNO, me.nonOrthDeltaCoeffs);
    same(tag + " nonOrthCorrectionVectors", corr, me.corr);
    same(tag + " interpolate(x + 2 y + 3 z)", foamAmiInterpolate(S, lin), me.lin);
}

std::vector<double> indexField(std::size_t n)
{
    std::vector<double> f(n);
    for (std::size_t j = 0; j < n; ++j) {
        const double x = double(j);
        f[j] = 1.0 + 0.25 * x + 0.001953125 * x * x;
    }
    return f;
}

void compareDump(const PolyMesh& pm, const fs::path& dumpFile, const std::string& label)
{
    Dump d;
    std::string err;
    if (!readDump(dumpFile, d, err)) {
        check(false, label + ": " + err);
        return;
    }
    std::vector<double> x = pm.px, y = pm.py, z = pm.pz;
    if (!rotateZone(pm, d, x, y, z)) {
        check(false, label + ": no cellZone " + d.zone);
        return;
    }
    const std::string tag0 = d.zone == "-" ? label + " rest" : std::format("{} {} {:.17g} rad", label, d.zone, d.angle);
    check(int(x.size()) == d.nPoints && pointHash(x, y, z) == d.hash, tag0 + ": the points are OpenFOAM's bit for bit");
    for (const DumpPair& p : d.pairs) {
        const std::string tag = tag0 + " " + p.src + "/" + p.tgt;
        const int is = patchIndex(pm, p.src), it = patchIndex(pm, p.tgt);
        if (is < 0 || it < 0) {
            check(false, tag + ": patches not in the mesh");
            continue;
        }
        const auto A = foamCyclicAmi(pm, x, y, z, is, it);
        if (!A) {
            check(false, tag + ": " + A.error());
            continue;
        }
        for (const std::string& wn : A->warnings) std::printf("  warning: %s\n", wn.c_str());
        check(A->src.start == p.s.start && A->src.addr == p.s.addr,
              std::format("{} srcAddress identical ({} faces, {} entries)", tag, p.s.wSum.size(), p.s.addr.size()));
        check(A->tgt.start == p.t.start && A->tgt.addr == p.t.addr,
              std::format("{} tgtAddress identical ({} faces, {} entries)", tag, p.t.wSum.size(), p.t.addr.size()));
        same(tag + " srcMagSf", A->src.magSf, p.s.magSf);
        same(tag + " tgtMagSf", A->tgt.magSf, p.t.magSf);
        same(tag + " srcWeightsSum", A->src.wSum, p.s.wSum);
        same(tag + " tgtWeightsSum", A->tgt.wSum, p.t.wSum);
        same(tag + " srcWeights", A->src.w, p.s.w);
        same(tag + " tgtWeights", A->tgt.w, p.t.w);
        same(tag + " interpolateToSource", foamAmiInterpolate(A->src, indexField(p.t.wSum.size())), p.toSrc);
        same(tag + " interpolateToTarget", foamAmiInterpolate(A->tgt, indexField(p.s.wSum.size())), p.toTgt);
        const FoamAmiSum ss = foamAmiSum(A->src), st = foamAmiSum(A->tgt);
        const auto low = [](const FoamAmiSide& S) {
            return std::ranges::count_if(S.wSum, [](double w) { return w < 0.2; });
        };
        std::printf("  %s\n  %s\n  uncovered: source %zu, target %zu; sum(weights) < 0.2: source %td, target %td\n",
                    foamAmiSumLine(A->src, "source", 16).c_str(), foamAmiSumLine(A->tgt, "target", 16).c_str(),
                    ss.nZero, st.nZero, low(A->src), low(A->tgt));
        if (p.fv.size() == 2) {
            checkFvPatch(tag + " " + p.fv[0].name, p.fv[0], p.fv[1], A->src);
            checkFvPatch(tag + " " + p.fv[1].name, p.fv[1], p.fv[0], A->tgt);
        }
    }
}

void testErrors(const PolyMesh& pm)
{
    const int a1 = patchIndex(pm, "AMI1"), a2 = patchIndex(pm, "AMI2"), wl = patchIndex(pm, "walls");
    const auto r1 = foamCyclicAmi(pm, pm.px, pm.py, pm.pz, a2, a1);
    check(!r1 && r1.error() == "AMI interpolator only available to owner patch", "E1 the neighbour patch is refused in OpenFOAM's words");
    const auto r2 = foamCyclicAmi(pm, pm.px, pm.py, pm.pz, wl, a2);
    check(!r2 && r2.error().find("not both cyclicAMI") != std::string::npos, "E1 a wall patch is refused by name");
    const auto r3 = foamCyclicAmi(pm, pm.px, pm.py, pm.pz, a1, a2);
    const auto r4 = foamCyclicAmi(pm, pm.px, pm.py, pm.pz, a1, a2);
    check(r3 && r4 && *r3 == *r4, "E1 a second call returns the same bytes");
}

int runCase(const fs::path& caseDir, const std::vector<fs::path>& dumps, bool errors)
{
    const std::string label = caseDir.filename().string();
    const auto pm = readPolyMesh(caseDir / "constant" / "polyMesh");
    if (!pm) {
        check(false, label + ": " + pm.error());
        return 1;
    }
    std::printf("%s: %lld cells\n", label.c_str(), static_cast<long long>(pm->man.nCells));
    for (const fs::path& d : dumps) compareDump(*pm, d, label);
    if (errors) testErrors(*pm);
    return 0;
}

// Every AMI pair of a mesh: time, the log lines, uncovered / low faces.
void runMesh(const fs::path& dir, const fs::path& dump, int precision, int repeat)
{
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    const auto pm = readPolyMesh(dir);
    if (!pm) {
        check(false, dir.string() + ": " + pm.error());
        return;
    }
    std::printf("%s: %lld cells, %lld points, read in %.2f s\n", dir.string().c_str(),
                static_cast<long long>(pm->man.nCells), static_cast<long long>(pm->man.nPoints),
                std::chrono::duration<double>(clock::now() - t0).count());
    for (std::size_t i = 0; i < pm->man.patches.size(); ++i) {
        const FoamPatch& p = pm->man.patches[i];
        if (p.type != FoamPatchType::CyclicAMI) continue;
        const int j = patchIndex(*pm, p.neighbourPatch);
        if (j < int(i)) continue;
        double best = 1e300;
        std::expected<FoamAmi, std::string> A = std::unexpected(std::string("not run"));
        for (int r = 0; r < repeat; ++r) {
            const auto t1 = clock::now();
            A = foamCyclicAmi(*pm, pm->px, pm->py, pm->pz, int(i), j);
            best = std::min(best, std::chrono::duration<double>(clock::now() - t1).count());
        }
        if (!A) {
            check(false, p.name + ": " + A.error());
            continue;
        }
        std::printf("AMI: Creating AMI for source:%s and target:%s  [%.1f ms, best of %d]\n", p.name.c_str(),
                    p.neighbourPatch.c_str(), 1e3 * best, repeat);
        std::printf("AMI: Patch source faces: %zu\nAMI: Patch target faces: %zu\n", A->src.wSum.size(),
                    A->tgt.wSum.size());
        for (const std::string& wn : A->warnings) std::printf("  warning: %s\n", wn.c_str());
        const auto low = [](const FoamAmiSide& S, double t) {
            return std::ranges::count_if(S.wSum, [t](double w) { return w < t; });
        };
        std::printf("%s\n%s\n", foamAmiSumLine(A->src, "source", precision).c_str(),
                    foamAmiSumLine(A->tgt, "target", precision).c_str());
        std::printf("  source: wSum == 0: %zu, < 0.2: %td, < 0.5: %td, < 0.95: %td; entries %zu\n",
                    foamAmiSum(A->src).nZero, low(A->src, 0.2), low(A->src, 0.5), low(A->src, 0.95), A->src.addr.size());
        std::printf("  target: wSum == 0: %zu, < 0.2: %td, < 0.5: %td, < 0.95: %td; entries %zu\n",
                    foamAmiSum(A->tgt).nZero, low(A->tgt, 0.2), low(A->tgt, 0.5), low(A->tgt, 0.95), A->tgt.addr.size());
    }
    if (!dump.empty()) compareDump(*pm, dump, dir.parent_path().parent_path().filename().string());
}

} // namespace

int main(int argc, char** argv)
{
    if (argc >= 3 && std::string(argv[1]) == "--mesh") {
        fs::path dump;
        int precision = 6, repeat = 1;
        for (int i = 3; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--precision" && i + 1 < argc) precision = std::atoi(argv[++i]);
            else if (a == "--repeat" && i + 1 < argc) repeat = std::max(1, std::atoi(argv[++i]));
            else dump = a;
        }
        runMesh(argv[2], dump, precision, repeat);
        return dnstest::finish();
    }
    if (argc >= 3) {
        std::vector<fs::path> dumps;
        for (int i = 2; i < argc; ++i) dumps.emplace_back(argv[i]);
        runCase(argv[1], dumps, false);
        return dnstest::finish();
    }
#ifdef DNS_FV_DATA
    const fs::path root = fs::path(DNS_FV_DATA) / "ami";
    if (!fs::exists(root)) dnstest::skip("no " + root.string());
    std::vector<fs::path> cases;
    for (const auto& e : fs::directory_iterator(root))
        if (e.is_directory()) cases.push_back(e.path());
    std::ranges::sort(cases);
    for (const fs::path& c : cases) {
        std::vector<fs::path> dumps;
        for (const auto& e : fs::directory_iterator(c))
            if (e.path().filename().string().starts_with("amiDump_")) dumps.push_back(e.path());
        std::ranges::sort(dumps);
        runCase(c, dumps, c.filename() == "ami_planar");
    }
    check(!cases.empty(), "the checked-in AMI cases were found");
    return dnstest::finish();
#else
    std::printf("usage: test_fvami CASEDIR DUMP [DUMP...] | --mesh POLYMESHDIR [DUMP]\n");
    return 1;
#endif
}
