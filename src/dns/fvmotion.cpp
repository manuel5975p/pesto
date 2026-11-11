#include "dns/fvmotion.hpp"

#include "dns/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <format>

// OpenFOAM's operand order everywhere in this file; its build (gcc -O3, x86-64)
// contracts nothing, neither may this one: CMakeLists.txt compiles this file
// with -ffp-contract=off.

namespace dns {
namespace {

using V3 = std::array<double, 3>;

V3 add(const V3& a, const V3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
V3 sub(const V3& a, const V3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
V3 mul(double s, const V3& a) { return {s * a[0], s * a[1], s * a[2]}; }
V3 div(const V3& a, double s) { return {a[0] / s, a[1] / s, a[2] / s}; }
V3 neg(const V3& a) { return {-a[0], -a[1], -a[2]}; }
double dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
V3 cross(const V3& a, const V3& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double magSqr(const V3& a) { return a[0] * a[0] + a[1] * a[1] + a[2] * a[2]; }
double mag(const V3& a) { return std::sqrt(magSqr(a)); }

V3 at(FoamPoints p, std::int32_t i) { return {p.x[i], p.y[i], p.z[i]}; }

constexpr double kVSmall = 1e-300, kSmall = 1e-15, kRootVSmall = 1e-150;

// OpenFOAM quaternion (w, v) and the members rotatingMotion::transformation() uses.
struct Quat
{
    double w = 1;
    V3 v{};
};
Quat qmul(const Quat& a, const Quat& b) // operator*(q1, q2)
{
    return {a.w * b.w - dot(a.v, b.v), add(add(mul(a.w, b.v), mul(b.w, a.v)), cross(a.v, b.v))};
}
Quat conj(const Quat& q) { return {q.w, neg(q.v)}; }
Quat mulq0v(const Quat& q, const V3& u) { return {-dot(q.v, u), add(mul(q.w, u), cross(q.v, u))}; }
V3 invTransform(const Quat& q, const V3& u) { return qmul(mulq0v(conj(q), u), q).v; }
Quat normalised(const Quat& q)
{
    const double s = std::sqrt(q.w * q.w + magSqr(q.v));
    if (s < kRootVSmall) return {0, {0, 0, 0}};
    return {q.w / s, div(q.v, s)};
}
std::array<double, 9> rotTensor(const Quat& q) // quaternion::R()
{
    const double w2 = q.w * q.w, x2 = q.v[0] * q.v[0], y2 = q.v[1] * q.v[1], z2 = q.v[2] * q.v[2];
    const double txy = 2 * q.v[0] * q.v[1], twz = 2 * q.w * q.v[2], txz = 2 * q.v[0] * q.v[2];
    const double twy = 2 * q.w * q.v[1], tyz = 2 * q.v[1] * q.v[2], twx = 2 * q.w * q.v[0];
    return {w2 + x2 - y2 - z2, txy - twz,         txz + twy,
            txy + twz,         w2 - x2 + y2 - z2, tyz - twx,
            txz - twy,         tyz + twx,         w2 - x2 - y2 + z2};
}

// triangle(a, b, c).sweptVol(triangle(ta, tb, tc))
double triSweptVol(const V3& a, const V3& b, const V3& c, const V3& ta, const V3& tb, const V3& tc)
{
    return (1.0 / 12.0) *
           (dot(sub(ta, a), cross(sub(b, a), sub(c, a))) + dot(sub(tb, b), cross(sub(c, b), sub(ta, b))) +
            dot(sub(c, tc), cross(sub(tb, tc), sub(ta, tc)))

            + dot(sub(ta, a), cross(sub(b, a), sub(c, a))) + dot(sub(b, tb), cross(sub(ta, tb), sub(tc, tb))) +
            dot(sub(c, tc), cross(sub(b, tc), sub(ta, tc))));
}

} // namespace

FvVec3 foamFaceCentre(std::span<const std::int32_t> n, FoamPoints p)
{
    const std::size_t nP = n.size();
    if (nP == 3) return mul(1.0 / 3.0, add(add(at(p, n[0]), at(p, n[1])), at(p, n[2])));
    V3 avg{0, 0, 0};
    for (const std::int32_t i : n) avg = add(avg, at(p, i));
    const V3 centrePoint = div(avg, double(nP));
    double sumA = 0;
    V3 sumAc{0, 0, 0};
    for (std::size_t pI = 0; pI < nP; ++pI) {
        const V3 thisPoint = at(p, n[pI]), nextPoint = at(p, n[pI == nP - 1 ? 0 : pI + 1]);
        const V3 ttc = add(add(thisPoint, nextPoint), centrePoint);              // 3 x the triangle centre
        const double ta = mag(cross(sub(thisPoint, centrePoint), sub(nextPoint, centrePoint))); // 2 x its area
        sumA += ta;
        sumAc = add(sumAc, mul(ta, ttc));
    }
    if (sumA > kVSmall) return div(sumAc, 3.0 * sumA);
    return centrePoint;
}

double foamFaceSweptVol(std::span<const std::int32_t> n, FoamPoints oldP, FoamPoints newP)
{
    double sv = 0;
    const V3 centreOld = foamFaceCentre(n, oldP), centreNew = foamFaceCentre(n, newP);
    const std::size_t nP = n.size();
    for (std::size_t pi = 0; pi + 1 < nP; ++pi)
        sv += triSweptVol(centreOld, at(oldP, n[pi]), at(oldP, n[pi + 1]), centreNew, at(newP, n[pi]),
                          at(newP, n[pi + 1]));
    sv += triSweptVol(centreOld, at(oldP, n[nP - 1]), at(oldP, n[0]), centreNew, at(newP, n[nP - 1]), at(newP, n[0]));
    return sv;
}

FoamRigidTransform foamRotatingMotion(const FvSolidBody& b, double t)
{
    // scalar angle = omega_->integrate(0, t) (Function1 Constant: (x2 - x1) * value);
    // quaternion R(axis_, angle); septernion TR(septernion(-origin_) * R * septernion(origin_));
    const double angle = (t - 0.0) * b.omega;
    const double am = mag(b.axis);
    const V3 nAxis = am < kRootVSmall ? V3{0, 0, 0} : div(b.axis, am); // Foam::normalised(d)
    const Quat R{std::cos(0.5 * angle), mul(std::sin(0.5 * angle), nAxis)};
    const Quat I{1, {0, 0, 0}};
    // septernion(-origin) * R  (operator*(septernion, quaternion)): (R.invTransform(t), r * R)
    const V3 t1 = invTransform(R, neg(b.origin));
    const Quat r1 = qmul(I, R);
    // ... * septernion(origin)  (operator*(tr1, tr2)): (tr2.r.invTransform(tr1.t) + tr2.t, tr1.r.transform(tr2.r))
    FoamRigidTransform T;
    T.trans = add(invTransform(I, t1), b.origin);
    const Quat r = normalised(qmul(r1, I)); // quaternion::transform(quaternion)
    T.R = rotTensor(r);
    T.translate = mag(T.trans) > kVSmall;
    double d2 = 0; // mag(tr.r().R() - I) > SMALL
    for (int i = 0; i < 9; ++i) {
        const double x = T.R[std::size_t(i)] - (i % 4 == 0 ? 1.0 : 0.0);
        d2 += x * x;
    }
    T.rotate = std::sqrt(d2) > kSmall;
    return T;
}

FvVec3 FoamRigidTransform::apply(const FvVec3& p) const
{
    const V3 q = translate ? sub(p, trans) : p;
    if (!rotate) return q;
    return {R[0] * q[0] + R[1] * q[1] + R[2] * q[2], R[3] * q[0] + R[4] * q[1] + R[5] * q[2],
            R[6] * q[0] + R[7] * q[1] + R[8] * q[2]};
}

std::expected<FvMeshMotion, std::string> FvMeshMotion::make(const FvLoaded& L)
{
    const PolyMesh& pm = L.pm;
    const FvDynamicMesh& dm = L.c.motion;
    if (!dm.on || dm.bodies.empty()) return std::unexpected("constant/dynamicMeshDict: no mesh motion");
    FvMeshMotion M;
    M.x0_ = pm.px;
    M.y0_ = pm.py;
    M.z0_ = pm.pz;
    const std::size_t nP = pm.px.size(), nCells = std::size_t(pm.man.nCells), nF = pm.owner.size();
    // The motion group of every cell: -1 static, else its body (cellZones are disjoint).
    std::vector<std::int32_t> group(nCells, -1);
    for (std::size_t k = 0; k < dm.bodies.size(); ++k) {
        const FvSolidBody& b = dm.bodies[k];
        if (b.cellZone.empty()) {
            std::ranges::fill(group, std::int32_t(k));
            continue;
        }
        const auto zi = std::ranges::find(pm.man.cellZones, b.cellZone);
        if (zi == pm.man.cellZones.end())
            return std::unexpected("constant/dynamicMeshDict: No matching cellZones: " + b.cellZone);
        const auto& cells = pm.zoneCells[std::size_t(zi - pm.man.cellZones.begin())];
        if (cells.empty()) { // zoneMotion: "No cell points selected => move all points"
            std::ranges::fill(group, std::int32_t(k));
            continue;
        }
        for (const std::int32_t c : cells) group[std::size_t(c)] = std::int32_t(k);
    }
    // A face between cells of different motion is sheared: OpenFOAM runs it (the
    // cells deform), this path does not -- such an interface is a cyclicAMI pair.
    for (std::size_t f = 0; f < pm.neighbour.size(); ++f) {
        const std::int32_t a = group[std::size_t(pm.owner[f])], b = group[std::size_t(pm.neighbour[f])];
        if (a == b) continue;
        std::size_t n = 0;
        for (std::size_t q = 0; q < pm.neighbour.size(); ++q)
            n += group[std::size_t(pm.owner[q])] != group[std::size_t(pm.neighbour[q])];
        const std::int32_t mv = std::max(a, b);
        const std::string& z = dm.bodies[std::size_t(mv)].cellZone;
        return std::unexpected(std::format(
            "constant/dynamicMeshDict: the moving {} borders {} cells through {} ordinary internal faces (the cells "
            "there would shear): a moving zone needs a cyclicAMI interface or must be the whole mesh",
            z.empty() ? std::string("mesh") : "cellZone '" + z + "'", std::min(a, b) < 0 ? "non-moving" : "differently moving",
            n));
    }
    // zoneMotion::selectZonePoints: the points of the faces of the body's cells;
    // dynamicMultiMotionSolverFvMesh::update applies the bodies in order.
    M.body_.assign(nP, -1);
    for (std::size_t k = 0; k < dm.bodies.size(); ++k)
        for (std::size_t f = 0; f < nF; ++f) {
            const bool mine = group[std::size_t(pm.owner[f])] == std::int32_t(k) ||
                              (f < pm.neighbour.size() && group[std::size_t(pm.neighbour[f])] == std::int32_t(k));
            if (!mine) continue;
            for (std::int32_t q = pm.faceStart[f]; q < pm.faceStart[f + 1]; ++q) {
                std::int32_t& pb = M.body_[std::size_t(pm.faceNodes[std::size_t(q)])];
                if (pb >= 0 && pb != std::int32_t(k))
                    return std::unexpected("constant/dynamicMeshDict: a point is shared by the cells of '" +
                                           dm.bodies[std::size_t(pb)].cellZone + "' and '" + dm.bodies[k].cellZone +
                                           "' (two motions on one point)");
                pb = std::int32_t(k);
            }
        }
    // Stage 3b's gated configuration: ONE rigid motion of the whole mesh. Stage
    // 3c: regions of different motion (a static remainder included) joined by
    // cyclicAMI pairs only -- no ordinary face (above) and no point between them.
    const bool whole = std::ranges::all_of(group, [&](std::int32_t g) { return g == group[0]; }) && group[0] >= 0;
    const bool ami = std::ranges::any_of(pm.man.patches,
                                         [](const FoamPatch& p) { return p.type == FoamPatchType::CyclicAMI; });
    if (!whole && !ami)
        return std::unexpected("constant/dynamicMeshDict: the motion does not cover every cell with one rigid body "
                               "(disconnected regions moving differently, or a static remainder, need a cyclicAMI "
                               "interface between them: docs/fvpimple.md \"Moving mesh\")");
    if (!whole)
        for (std::size_t f = 0; f < nF; ++f) {
            const std::int32_t gf = group[std::size_t(pm.owner[f])];
            for (std::int32_t q = pm.faceStart[f]; q < pm.faceStart[f + 1]; ++q)
                if (M.body_[std::size_t(pm.faceNodes[std::size_t(q)])] != gf)
                    return std::unexpected(std::format(
                        "constant/dynamicMeshDict: point {} is shared by cells of different motion (a cyclicAMI "
                        "interface made with createBaffles shares its points: split them, e.g. mergeOrSplitBaffles "
                        "-split; OpenFOAM would move the point and shear the cells)",
                        pm.faceNodes[std::size_t(q)]));
        }
    if (L.fm.emptyAxis >= 0)
        for (const FvSolidBody& b : dm.bodies) {
        for (int q = 0; q < 3; ++q)
            if (q != L.fm.emptyAxis && b.axis[std::size_t(q)] != 0)
                return std::unexpected(std::format("constant/dynamicMeshDict: a 2D (empty-patch) mesh must rotate about "
                                                   "its empty axis {} (axis ({:g} {:g} {:g}))",
                                                   "xyz"[L.fm.emptyAxis], b.axis[0], b.axis[1], b.axis[2]));
        }
    return M;
}

std::int64_t FvMeshMotion::movedPoints() const
{
    return std::ranges::count_if(body_, [](std::int32_t b) { return b >= 0; });
}

std::expected<FvMeshMove, std::string> FvMeshMotion::update(double t, double deltaT, FvLoaded& L, FvGraph& G,
                                                            FvSimpleSetup& s, const Beside& beside) const
{
    PolyMesh& pm = L.pm;
    const FoamCase& c = L.c;
    const std::size_t nP = pm.px.size();
    // polyMesh::movePoints: oldPoints = the current points.
    const std::vector<double> ox = pm.px, oy = pm.py, oz = pm.pz;
    std::vector<FoamRigidTransform> tr;
    for (const FvSolidBody& b : c.motion.bodies) tr.push_back(foamRotatingMotion(b, t));
    for (std::size_t i = 0; i < nP; ++i) {
        if (body_[i] < 0) continue;
        const FvVec3 x = tr[std::size_t(body_[i])].apply({x0_[i], y0_[i], z0_[i]});
        pm.px[i] = x[0];
        pm.py[i] = x[1];
        pm.pz[i] = x[2];
    }
    // The geometry: the one definition, on the moved points.
    auto fm = FaceMesh::fromPolyMesh(pm, CellOrder::Source, 0, {.ami = true});
    if (!fm) return std::unexpected(std::format("moved mesh at t = {:g}: {}", t, fm.error()));
    L.fm = std::move(*fm);
    // cyclicAMIPolyPatch::initMovePoints: the AMI is recomputed on the moved
    // points (once per mesh motion), and with it the couplings' pattern.
    if (!G.ami.empty()) {
        auto ng = buildFvGraph(pm, L.fm);
        if (!ng) return std::unexpected(std::format("moved mesh at t = {:g}: {}", t, ng.error()));
        G = std::move(*ng);
    }
    auto ns = buildSimpleSetup(L.fm, c, G.g, G.pat, &G.ami);
    if (!ns) return std::unexpected(std::format("moved mesh at t = {:g}: {}", t, ns.error()));
    // From here the setup's geometry, AMI and graph are final; `beside` reads them on its own thread while this
    // one computes the wall distance (written to ns->ras only) and meshPhi. Joined at the end of the block.
    FvMeshMove mv;
    {
        std::jthread side;
        if (beside) side = std::jthread([&] { beside(*ns, G); });
        if (c.ras.model != FvTurbModel::Laminar)
            if (auto r = buildRasSetup(pm, L.fm, c, *ns, &G.ami); !r)
                return std::unexpected(std::format("moved mesh at t = {:g}: {}", t, r.error()));

        // fvGeometryScheme::setMeshPhi: sweptVol / deltaT per OpenFOAM face. Each face its serial expression,
        // written to its own index: identical at any thread count (dns/parallel.hpp).
        const FvSimpleSetup& q = *ns;
        const FoamPoints oldP{ox.data(), oy.data(), oz.data()}, newP{pm.px.data(), pm.py.data(), pm.pz.data()};
        const double rdt = 1.0 / deltaT;
        auto nodes = [&](std::int64_t f) {
            return std::span<const std::int32_t>(
                pm.faceNodes.data() + pm.faceStart[std::size_t(f)],
                std::size_t(pm.faceStart[std::size_t(f) + 1] - pm.faceStart[std::size_t(f)]));
        };
        const std::size_t nI = std::size_t(q.nI), nB = std::size_t(q.nB);
        const std::size_t kBlock = 16384; // faces per worker at least: a small mesh runs inline
        mv.meshPhi.resize(nI);
        forBlocks(workerCount(0, nI / kBlock + 1), nI, [&](int, std::size_t lo, std::size_t hi) {
            for (std::size_t f = lo; f < hi; ++f) {
                // FaceMesh orientation own -> nei; OpenFOAM's is lower source cell -> higher.
                const double sg = q.srcCell[std::size_t(q.own[f])] < q.srcCell[std::size_t(q.nei[f])] ? 1.0 : -1.0;
                mv.meshPhi[f] = sg * (foamFaceSweptVol(nodes(q.internalSrc[f]), oldP, newP) * rdt);
            }
        });
        mv.meshPhiB.resize(nB);
        mv.uWall.assign(3 * nB, 0.0);
        forBlocks(workerCount(0, nB / kBlock + 1), nB, [&](int, std::size_t lo, std::size_t hi) {
            for (std::size_t b = lo; b < hi; ++b) {
                const std::int64_t src = L.fm.srcFace[std::size_t(q.bFace[b])];
                const auto fn = nodes(src);
                mv.meshPhiB[b] = foamFaceSweptVol(fn, oldP, newP) * rdt;
                if (!c.patches[std::size_t(q.bPatch[b])].movingWall) continue;
                // movingWallVelocityFvPatchVectorField::Uwall()
                const V3 oldFc = foamFaceCentre(fn, oldP), newFc = foamFaceCentre(fn, newP);
                const V3 Up = div(sub(newFc, oldFc), deltaT);
                const V3 n{q.bN[b], q.bN[nB + b], q.bN[2 * nB + b]};
                const double Un = mv.meshPhiB[b] / (q.bMagS[b] + kVSmall);
                const V3 u = add(Up, mul(Un - dot(n, Up), n));
                for (std::size_t k = 0; k < 3; ++k) mv.uWall[k * nB + b] = u[k];
            }
        });
    }
    s = std::move(*ns);
    return mv;
}

} // namespace dns
