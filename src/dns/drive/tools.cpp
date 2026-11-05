// dns prop, dns mesh (host-only).
#include "dns/drive/internal.hpp"

#include "dns/config.hpp"
#include "dns/facemesh.hpp"
#include "dns/prop/bemt.hpp"
#include "dns/prop/liftline.hpp"

#if DNS_WITH_GPU
#endif

#include <cmath>
#include <cstdio>
#include <format>
#include <fstream>
#include <limits>
#include <numbers>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace dns::drive {

using std::numbers::pi;
namespace {

// --- prop --------------------------------------------------------------------

std::expected<std::string, std::string> readText(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::unexpected("cannot read '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    if (f.bad()) return std::unexpected("error reading '" + path + "'");
    return ss.str();
}

// Actuator-disk (Froude) efficiency bound at this CT and J.
double etaIdeal(double J, double CT)
{
    if (J <= 0 || CT <= 0) return std::numeric_limits<double>::quiet_NaN();
    return 2.0 / (1.0 + std::sqrt(1.0 + 8.0 * CT / (pi * J * J)));
}

int runPropSelfTest()
{
    std::printf("liftline self-test\n");
    bool good = true;
    for (const auto& ck : prop::liftlineSelfTest())
    {
        std::printf("  %s  %s: %s\n", ck.pass ? "PASS" : "FAIL", ck.name.c_str(),
                    ck.message.c_str());
        good = good && ck.pass;
    }
    return good ? 0 : exitNumerical;
}
} // namespace


int runProp(const Case& cs)
{
    const prop::Case& pc = cs.prop;
    if (pc.selftest) return runPropSelfTest();

    // Phase 3: inputs, then every output, before any work.
    auto pfText = readText(pc.planformPath);
    if (!pfText) return die(exitInput, "--planform: " + pfText.error());
    auto pf = prop::Planform::fromCsv(*pfText);
    if (!pf) return die(exitInput, "--planform " + pc.planformPath + ": " + pf.error());
    const double hubRR = pc.hubRR > 0 ? pc.hubRR : pf->rootRR();
    if (!(hubRR > 0 && hubRR < 1))
        return die(exitInput, "--planform " + pc.planformPath
                                  + ": its first r_R must be < 1 to serve as the hub "
                                    "radius (or pass --hub-radius)");
    std::optional<prop::Polar> polar;
    if (pc.polarPath.empty())
        polar = prop::Polar::thinAirfoil(pc.thinAlpha0Deg, pc.thinCd0,
                                         prop::viternaCdMax(pf->aspectRatio(hubRR)));
    else
    {
        auto pt = readText(pc.polarPath);
        if (!pt) return die(exitInput, "--polar: " + pt.error());
        auto p = prop::Polar::fromCsv(*pt);
        if (!p) return die(exitInput, "--polar " + pc.polarPath + ": " + p.error());
        polar = std::move(*p);
    }
    std::ofstream csv;
    if (!pc.outCsv.empty())
    {
        csv.open(pc.outCsv, std::ios::binary | std::ios::trunc);
        if (!csv) return die(exitOutput, "cannot create '" + pc.outCsv + "'");
    }
    prop::Case run = pc;
    run.hubRR = hubRR;

    // Phase 4.
    const bool ll = pc.method == prop::Method::Liftline;
    const double n = pc.op.revPerSec();
    std::vector<prop::CurvePoint> rows;
    std::string report, span;
    if (ll)
    {
        const auto geo = prop::liftGeometry(*pf, hubRR, pc.ll.panels, pc.ll.hub,
                                            pc.op.diameter);
        auto curve = prop::liftlineCurve(run, geo, *polar);
        rows = std::move(curve.rows);
        report = std::format("liftline: B={} D={:g} m {:.0f} rpm, {} panels, wake {:g} D, "
                             "tip Mach {:.3f}\n"
                             "     J       CT        CP       eta      T[N]     Q[Nm]   conv\n",
                             pc.blades, pc.op.diameter, pc.op.rpm, pc.ll.panels,
                             pc.ll.wakeLength, pi * n * pc.op.diameter / pc.op.sound);
        for (const auto& r : rows)
            report += std::format("  {:5.3f} {:9.5f} {:9.5f} {:8.4f} {:9.4f} {:9.5f}   {}\n",
                                  r.J, r.CT, r.CP, r.eta, r.T, r.Q, int(r.converged));
        report += prop::curveSummary(rows);
        if (curve.span) span = "\n" + prop::liftlineSpanTable(*pc.dumpSpan, geo, *curve.span);
    }
    else
    {
        const auto bl = prop::bemtBlade(*pf, pc.blades, hubRR, pc.stations, pc.op.diameter);
        auto curve = prop::bemtCurve(run, bl, *polar);
        rows = std::move(curve.rows);
        if (curve.machHi > 0.6)
            std::fprintf(stderr, "dns: warning: station Mach up to %.2f with "
                                 "incompressible polars\n", curve.machHi);
        report = std::format("bemt: B={} D={:g} m {:g} rpm, planform {}, polar {}, {} "
                             "elements\n"
                             "     J       CT       CP      eta  eta_ideal       T[N]     "
                             "Q[Nm]      P[W]     Re75\n",
                             pc.blades, pc.op.diameter, pc.op.rpm, pc.planformPath,
                             pc.polarPath.empty() ? std::string("thin") : pc.polarPath,
                             pc.stations);
        std::vector<double> over;
        for (const auto& r : rows)
        {
            const double ei = etaIdeal(r.J, r.CT);
            report += std::format("  {:5.3f} {:8.5f} {:8.5f} {:8.4f} {:9.4f} {:10.4f} "
                                  "{:9.5f} {:9.3f} {:8.3g}\n",
                                  r.J, r.CT, r.CP, r.eta, ei, r.T, r.Q, r.P, r.re75);
            if (std::isfinite(ei) && r.eta > ei + 1e-9) over.push_back(r.J);
        }
        report += prop::curveSummary(rows);
        report += std::format("  momentum check: max |dT_blade - dT_momentum| = {:.1e} of "
                              "peak dT/dr (momentum-region annuli)\n",
                              curve.worstMomErr);
        if (!over.empty())
        {
            std::string js;
            for (double j : over) js += std::format("{}{:g}", js.empty() ? "" : ", ", j);
            std::fprintf(stderr, "dns: warning: eta above the actuator-disk bound at J = %s\n",
                         js.c_str());
        }
        if (curve.unsolved)
            std::fprintf(stderr, "dns: warning: %d annulus solves found no bracket (nan, "
                                 "dropped from the integrals)\n", curve.unsolved);
        if (pc.dumpSpan)
            span = "\n" + prop::bemtSpanTable(*pc.dumpSpan, bl,
                                              prop::bemtSolve(*pc.dumpSpan, bl, pc.op, *polar));
    }
    if (csv.is_open())
    {
        csv << prop::curveCsv(rows, ll);
        csv.close();
        if (!csv) return die(exitOutput, "write failed on '" + pc.outCsv + "'");
        report += "wrote " + pc.outCsv + "\n";
    }
    if (!pc.quiet) std::fputs((report + span).c_str(), stdout);
    std::string bad;
    for (const auto& r : rows)
        if (!r.converged) bad += std::format("{}{:g}", bad.empty() ? "" : ", ", r.J);
    if (!bad.empty())
        std::fprintf(stderr, "dns: warning: lifting line NOT converged at J = %s "
                             "(flagged converged = 0)\n", bad.c_str());
    return 0;
}

int runMeshManifest(const Case& cs)
{
    auto m = readPolyManifest(cs.polyMeshDir);
    if (!m) return die(exitInput, m.error());
    std::string s = std::format("manifest:\n"
                                "  format     {}, label {}, scalar 64\n"
                                "  counts     {} points, {} cells, {} faces ({} internal)\n",
                                m->binary ? "binary" : "ascii", m->labelBits, m->nPoints, m->nCells,
                                m->nFaces, m->nInternalFaces);
    for (const auto& p : m->patches)
    {
        s += std::format("  patch      {:<24} {:<14} {:>9} faces", p.name, foamPatchTypeName(p.type),
                         p.nFaces);
        if (p.type == FoamPatchType::Cyclic)
            s += " <-> " + p.neighbourPatch
                 + (p.separation ? std::format(" (translational {} {} {})", (*p.separation)[0],
                                               (*p.separation)[1], (*p.separation)[2])
                                 : std::string(" (transform unknown)"));
        s += "\n";
    }
    for (const auto& z : m->cellZones) s += "  cellZone   " + z + "\n";
    std::fputs(s.c_str(), stdout);
    return 0;
}

int runMesh(const Case& cs)
{
    auto pm = readPolyMesh(cs.polyMeshDir);
    if (!pm) return die(exitInput, pm.error());
    auto fm = FaceMesh::fromPolyMesh(*pm, CellOrder(cs.meshOrder));
    if (!fm) return die(exitInput, cs.polyMeshDir + ": " + fm.error());
    std::fputs(meshReport(*fm, meshQuality(*fm, *pm)).c_str(), stdout);
    return 0;
}


} // namespace dns::drive
