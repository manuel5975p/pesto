// Device-inline compressible face fluxes (3D; nz==1 degenerates exactly to
// 2D) shared by the solver rhs and the ghost-wall surface-force tally.
// GPU-internal header (HIP TUs only).
#pragma once

#include "dns/gpu/index.hpp"
#include "dns/transport.hpp"

#include <cmath>

namespace dns::gpu {

struct Flux
{
    Real rho, mx, my, mz, E;
};

// FP32 kernel view of dns::Transport: the constant-law values plus what
// Sutherland needs, in Real so the face evaluation costs one sqrt and one
// divide instead of a double-rate transcendental. `sutherland == 0` returns
// the stored mu/kappa untouched, so the constant-viscosity path -- every
// measured anchor in the repo -- is bit-for-bit unchanged.
struct ViscView
{
    Real mu, kappa;  // values at Tref
    Real invTref, s; // s = S / Tref (Sutherland, scaled)
    int sutherland;

    // mu(T)/muRef; see dns::Transport::ratio. Pre: T > 0.
    __device__ Real ratio(Real T) const
    {
        const Real t = T * invTref;
        return t * sqrt(t) * (Real(1) + s) / (t + s);
    }
};

inline ViscView viscView(const Transport& t)
{
    return {.mu = Real(t.muRef),
            .kappa = Real(t.muRef * t.cp() / t.Pr),
            .invTref = Real(1.0 / t.Tref),
            .s = Real(t.S / t.Tref),
            .sutherland = t.sutherland ? 1 : 0};
}

struct PrimView
{
    const Real *rho, *u, *v, *w, *e, *p, *T;
};

// Rotating-reference-frame zone (--rotor-frame mrf): the frozen-rotor
// approximation solved on the STATIC azimuth-0 geometry. Inside the zone the
// equations are the rotating-frame ones in absolute-velocity form: convective
// transport by u - w with w = f * omega x (p - c), a Coriolis source
// -f omega x m on the momentum, pressure work in the energy flux left on the
// ABSOLUTE velocity, viscous terms untouched (a rigid rotation carries no
// strain). f is a C1 blend that is 1 on a shaft-aligned cylinder holding the
// blade and 0 outside a slightly larger one, so the far field, the NSCBC rows
// and the sink never see the frame.
//
// The blend is built from the cylindrical radius r about the shaft and the
// axial offset s alone, and w is purely azimuthal, so grad f is perpendicular
// to w and div(f w) == 0 continuously. Continuity is NOT enough: sampled
// pointwise at faces, the blend's DISCRETE divergence is O((h/W)^2 omega r)
// -- measured 3e-2 on a 6-cell blend at tip speed 2, a coherent ring source.
// The face transport correction is therefore built from a corner STREAM
// potential instead (mrfFaceVel): psi = -omega fa(s) G(r) with G' = r fr(r),
// so w_n at a face is a difference of two corner psi values and the discrete
// divergence TELESCOPES to zero by construction, on the uniform and the
// separable focal mesh alike. A uniform axial freestream then passes the zone
// to FP roundoff (pinned in gpu_mrf). The pointwise form (mrfFrameVel) stays
// for the Coriolis source and the dt bound, where no telescoping is at stake.
// active == 0 keeps every caller bit-for-bit the static solver.
struct MrfView
{
    int active = 0;
    int axis = 0;                       // shaft axis 0/1/2
    Real omega = Real(0);               // signed rate, rad per unit time
    Real cx = Real(0), cy = Real(0), cz = Real(0); // shaft point (world)
    Real r1 = Real(0), r2 = Real(0);    // radial blend: f=1 inside r1, 0 past r2
    Real a1 = Real(0), a2 = Real(0);    // axial |s| blend, same convention
    // Physical coordinates per axis (device, length nx/ny/nz): cell centres and
    // +faces. Carried as tables because on a focal mesh the position is the
    // REMAP of the reference coordinate, which no kernel view holds.
    const Real* cc[3] = {nullptr, nullptr, nullptr};
    const Real* fc[3] = {nullptr, nullptr, nullptr};
    // General-mesh path (DeviceComp::setMrfCurv, z shaft on a z-extruded
    // curvilinear mesh -- the O-grid): per-NODE stream potential
    // psi = -omega * hz * G(r_node), (ny+1) rows of nx (row j*nx + i; row ny
    // is the Extend continuation), and per-cell physical centres in Grid::idx
    // layout. psi non-null switches the face transport correction from the
    // separable tables to node differences (mrfFaceQNode); pc feeds the
    // Coriolis source and the dt bound. Exactly one of {cc/fc, psi/pc} is set.
    const Real* psi = nullptr;
    const Real* pc[3] = {nullptr, nullptr, nullptr};
};

// 1 -> 0 smoothstep between q1 and q2 (C1; q2 > q1).
__host__ __device__ inline Real mrfRamp(Real q, Real q1, Real q2)
{
    if (q <= q1) return Real(1);
    if (q >= q2) return Real(0);
    const Real s = (q - q1) / (q2 - q1);
    return Real(1) - s * s * (Real(3) - Real(2) * s);
}

// Frame factor f at a world point, and w = f * omega x (p - c) into `w`.
// Returns f; f == 0 leaves w zeroed.
__host__ __device__ inline Real mrfFrameVel(const MrfView& z, Real px, Real py,
                                            Real pz, Real w[3])
{
    w[0] = w[1] = w[2] = Real(0);
    const Real d[3] = {px - z.cx, py - z.cy, pz - z.cz};
    const int b = (z.axis + 1) % 3, c = (z.axis + 2) % 3;
    const Real r = sqrt(d[b] * d[b] + d[c] * d[c]);
    const Real f = mrfRamp(r, z.r1, z.r2) *
                   mrfRamp(fabs(d[z.axis]), z.a1, z.a2);
    if (f == Real(0)) return f;
    // omega x d with omega = omega * unit(axis): (axis, b, c) cyclic.
    const Real fo = f * z.omega;
    w[b] = -fo * d[c];
    w[c] = fo * d[b];
    return f;
}

// G(r) = int_0^r s fr(s) ds for the radial blend fr (1 below r1, smoothstep
// down to 0 at r2): the stream potential's radial profile. Closed form of the
// cubic-smoothstep integral; G is constant past r2, which is what makes w
// vanish identically outside the zone.
__host__ __device__ inline Real mrfPsiRad(const MrfView& z, Real r)
{
    if (r <= z.r1) return Real(0.5) * r * r;
    const Real d = z.r2 - z.r1;
    const Real u = r >= z.r2 ? Real(1) : (r - z.r1) / d;
    const Real u2 = u * u, u3 = u2 * u, u4 = u2 * u2;
    return Real(0.5) * z.r1 * z.r1 +
           d * (z.r1 * (u - u3 + Real(0.5) * u4) +
                d * (Real(0.5) * u2 - Real(0.75) * u4 + Real(0.4) * u4 * u));
}

// Transport correction at the +`axis` face of cell (i, j, k), written into
// w[3] with ONLY the face-normal component nonzero. w_n comes from the corner
// potential psi = -omega fa(s_axis) G(r): the in-plane curl of psi x_shaft,
// discretized as the difference of psi at the face's two corner nodes over
// their physical distance. Every cell's face sum of these then telescopes
// over the same four corner values -- discrete solenoidality by construction.
// Both cells sharing a face see identical corner values, so conservation is
// exact. Faces normal to the shaft carry no correction (w has no axial
// component).
__device__ inline void mrfFaceVel(const MrfView& z, int i, int j, int k,
                                  int axis, Real w[3])
{
    w[0] = w[1] = w[2] = Real(0);
    if (axis == z.axis) return;
    const int b = (z.axis + 1) % 3, c2 = (z.axis + 2) % 3;
    const int e = axis == b ? c2 : b; // the in-plane axis ALONG the face
    const int idx[3] = {i, j, k};
    const Real cen[3] = {z.cx, z.cy, z.cz};
    // Axial blend factor at the shared cell-centre axial coordinate; 1e30
    // bounds (the degenerate 2D zone) pass through mrfRamp as f = 1.
    const Real s = fabs(z.cc[z.axis][idx[z.axis]] - cen[z.axis]);
    const Real fa = mrfRamp(s, z.a1, z.a2);
    if (fa == Real(0)) return;
    // Face-normal coordinate (the face's own), and the two corner nodes along
    // e: the +e face coordinates of cells idx[e]-1 and idx[e]. At idx[e] == 0
    // the true lo corner is the periodic wrap node; the zone's boundary
    // clearance (runner-enforced) puts every corner of such a face outside
    // r2, where psi is CONSTANT, so the difference is zero whatever
    // coordinate stands in -- the unit synthetic offset below only keeps the
    // division defined and the stand-in point outside the zone.
    const Real dn = z.fc[axis][idx[axis]] - cen[axis];
    const Real nHiCoord = z.fc[e][idx[e]];
    const Real nLoCoord =
        idx[e] > 0 ? z.fc[e][idx[e] - 1] : nHiCoord - Real(1);
    const Real nHi = nHiCoord - cen[e];
    const Real nLo = nLoCoord - cen[e];
    const Real he = nHiCoord - nLoCoord;
    const Real gHi = mrfPsiRad(z, sqrt(dn * dn + nHi * nHi));
    const Real gLo = mrfPsiRad(z, sqrt(dn * dn + nLo * nLo));
    // psi = -omega fa G; w_b = +d psi/d c2, w_c2 = -d psi/d b.
    const Real dpsi = -z.omega * fa * (gHi - gLo);
    w[axis] = (axis == b ? dpsi : -dpsi) / he;
}

// Node-psi form of the transport correction (z.psi non-null; z shaft on a
// z-extruded mesh): the advected AREA flux w . S through the +`axis` face of
// cell (i,j,k), exact by Stokes -- the flux of curl(psi zhat) through a face
// is the loop integral of psi zhat . dl, and only the face's two z edges
// contribute, each psi * (+-hz) at its in-plane corner node (hz is baked into
// the table). The +i face of cell (i,j) has corner nodes (i+1, j) and
// (i+1, j+1); the +j face (i, j+1) and (i+1, j+1); the orientation follows
// e_j x e_k ~ +e_i of a positive-volume cell (CurvMetrics rejects the rest).
// Every cell's face sum telescopes over shared node values EXACTLY, and both
// cells of a face read identical values, so conservation and discrete
// solenoidality hold on ANY mesh -- that is the whole point of the potential.
// Faces normal to the shaft carry no correction (w has no axial component).
// k never enters: the mesh is z-extruded, so psi is z-invariant.
__device__ inline Real mrfFaceQNode(const MrfView& z, const GpuGrid& g, int i,
                                    int j, int axis)
{
    if (axis == 2) return Real(0);
    const int i1 = i + 1 == g.nx ? 0 : i + 1;
    const std::size_t lo = std::size_t(j) * g.nx;
    const std::size_t hi = std::size_t(j + 1) * g.nx;
    if (axis == 0) return z.psi[hi + i1] - z.psi[lo + i1];
    return z.psi[hi + i] - z.psi[hi + i1];
}

// Total Kennedy-Gruber convective minus viscous flux through the face between
// cells a and b (b = a + one step along `axis`); identical expression from
// both sides of a face keeps the scheme exactly conservative.
//
// `o` are the wrap-corrected neighbour offsets (dns/gpu/index.hpp) of ANY cell
// on the a-b line: the tangential stencil only ever steps along d != axis, and
// those offsets are shared by every cell of the line. One cell's offsets
// therefore serve both of its faces on every axis, which is what removes the
// per-access gidx from the hot loop. Pre: b == a + o.p[axis].
//
// nut (optional, per-cell dynamic eddy viscosity for LES): when non-null the
// face-averaged value is added to mu and, scaled by kapScale = cp/Pr_t, to
// kappa. nut==nullptr reproduces the molecular DNS flux bit-for-bit, so the
// solver rhs (LES off) and the ghost-wall force tally (molecular stress only)
// are unchanged.
// wf (optional): frame velocity at this face (--rotor-frame mrf, MrfView).
// Non-null switches the ADVECTING normal velocity to un - wf[axis] in every
// transport term while the pressure work in the energy flux stays on the
// absolute un -- the rotating-frame flux in absolute-velocity form. nullptr is
// the inertial flux bit-for-bit.
__device__ inline Flux faceFlux(const GpuGrid& g, ViscView vv, PrimView q,
                                const Offsets& o, std::size_t a, std::size_t b,
                                int axis, const Real* nut = nullptr,
                                Real kapScale = Real(0),
                                const Real* wf = nullptr)
{
    const Real h = axis == 0 ? g.hx : axis == 1 ? g.hy : g.hz;

    const Real rm = Real(0.5) * (q.rho[a] + q.rho[b]);
    const Real um[3] = {Real(0.5) * (q.u[a] + q.u[b]),
                        Real(0.5) * (q.v[a] + q.v[b]),
                        Real(0.5) * (q.w[a] + q.w[b])};
    const Real em = Real(0.5) * (q.e[a] + q.e[b]);
    const Real pm = Real(0.5) * (q.p[a] + q.p[b]);
    const Real un = um[axis];
    const Real ua = wf ? un - wf[axis] : un; // advecting normal velocity
    Flux f{rm * ua, rm * ua * um[0], rm * ua * um[1], rm * ua * um[2],
           rm * ua * em + pm * un};
    (axis == 0 ? f.mx : axis == 1 ? f.my : f.mz) += pm;
    // Sutherland is evaluated ONCE per face, on the face-average temperature
    // the Fourier flux already uses -- not per cell -- so mu and kappa stay a
    // single consistent pair on both sides of the face and the scheme stays
    // exactly conservative.
    Real muf = vv.mu, kapf = vv.kappa;
    if (vv.sutherland) {
        const Real r = vv.ratio(Real(0.5) * (q.T[a] + q.T[b]));
        muf *= r;
        kapf *= r;
    }
    if (nut) {
        const Real nf = Real(0.5) * (nut[a] + nut[b]);
        muf += nf;
        kapf += nf * kapScale;
    }
    if (muf == Real(0)) return f;

    // du[c][d] = d u_c / d x_d at the face: compact along the normal,
    // averaged central along the tangents.
    const Real* uc[3] = {q.u, q.v, q.w};
    Real du[3][3];
    for (int c = 0; c < 3; ++c) {
        du[c][axis] = (uc[c][b] - uc[c][a]) / h;
        for (int d = 0; d < 3; ++d) {
            if (d == axis) continue;
            const Real hd = d == 0 ? g.hx : d == 1 ? g.hy : g.hz;
            du[c][d] = Real(0.25) *
                (uc[c][a + o.p[d]] - uc[c][a + o.m[d]] +
                 uc[c][b + o.p[d]] - uc[c][b + o.m[d]]) / hd;
        }
    }
    const Real divu = du[0][0] + du[1][1] + du[2][2];
    const Real dTn = (q.T[b] - q.T[a]) / h;
    Real eflux = kapf * dTn;
    Real* fm[3] = {&f.mx, &f.my, &f.mz};
    for (int c = 0; c < 3; ++c) {
        const Real tau = muf * (du[c][axis] + du[axis][c]) -
                         (c == axis ? Real(2.0 / 3.0) * muf * divu : Real(0));
        *fm[c] -= tau;
        eflux += um[c] * tau;
    }
    f.E -= eflux;
    return f;
}

// (i,j,k) convenience for the SPARSE callers -- the surface-force tally and
// the NSCBC boundary rows -- where one extra offset computation per face is
// free. The hot rhs uses the offset form above.
__device__ inline Flux faceFlux(const GpuGrid& g, ViscView vv, PrimView q,
                                int i, int j, int k, int axis,
                                const Real* nut = nullptr,
                                Real kapScale = Real(0),
                                const MrfView* mz = nullptr)
{
    const Offsets o = offsetsAt(g, i, j, k);
    const std::size_t a = gidx(g, i, j, k);
    Real w[3];
    const Real* wf = nullptr;
    if (mz && mz->active) {
        mrfFaceVel(*mz, i, j, k, axis, w);
        wf = w;
    }
    return faceFlux(g, vv, q, o, a, a + o.p[axis], axis, nut, kapScale, wf);
}

// Device pointer table of the curvilinear metrics (dns/curvmesh.hpp), one
// entry per cell: face area vectors, face inverse Jacobians, 1/volume.
struct CurvView
{
    const Real* S[3][3];  // S[a][c]: component c of the +a-face area vector
    const Real* Ji[3][9]; // Ji[a][d*3+c]: d xi_d / d x_c at the +a face
    const Real* invVol;
};

// d u_c / d x_e at the CENTRE of cell (i,j,k) on a curvilinear mesh, for the
// SGS operators. Index-space central differences (dxi = 1) contracted with the
// cell-centre inverse Jacobian; row d of dXi/dx is the average of the two
// faces normal to axis d, which is where CurvMetrics stores it. On the
// identity map this is exactly the uniform (u[+1] - u[-1]) / (2 h), and at
// nz == 1 the k stencil wraps onto itself so every z column is exactly zero.
// `vel(cell, comp)` supplies the velocity (primitives or m/rho).
template <class T, class Vel>
__host__ __device__ inline void curvVelGradCentre(const CurvView& m,
                                                  const Offsets& o,
                                                  std::size_t here, Vel vel,
                                                  T gd[3][3])
{
    T dxi[3][3]; // dxi[c][d] = d vel_c / d xi_d
    for (int c = 0; c < 3; ++c)
        for (int d = 0; d < 3; ++d)
            dxi[c][d] = T(0.5) * (T(vel(here + o.p[d], c)) -
                                  T(vel(here + o.m[d], c)));
    T Jc[3][3]; // Jc[d][e] = d xi_d / d x_e at the cell centre
    for (int d = 0; d < 3; ++d) {
        const std::size_t back = here + o.m[d];
        for (int e = 0; e < 3; ++e)
            Jc[d][e] = T(0.5) * (T(m.Ji[d][d * 3 + e][here]) +
                                 T(m.Ji[d][d * 3 + e][back]));
    }
    for (int c = 0; c < 3; ++c)
        for (int e = 0; e < 3; ++e)
            gd[c][e] = dxi[c][0] * Jc[0][e] + dxi[c][1] * Jc[1][e] +
                       dxi[c][2] * Jc[2][e];
}

// SGS filter width squared of a curvilinear cell: the isotropic (dx dy dz)^(2/3)
// written through the exact cell volume, and vol/hz at nz == 1 (the unit
// z-depth carries no filter width). Both reduce to the uniform-mesh Delta^2 of
// DeviceComp's constructor when the map is the identity.
__host__ __device__ inline Real curvDelta2(const GpuGrid& g, Real invVol)
{
    const double vol = 1.0 / double(invVol);
    if (g.nz > 1) {
        const double c = cbrt(vol);
        return Real(c * c);
    }
    return Real(vol / double(g.hz));
}

// Curvilinear counterpart of faceFlux: Un = u_avg . S, viscous gradients in
// index space mapped through the face inverse Jacobian, flux carries the
// face area (the rhs divides by the cell volume, not the spacing).
// nut (optional, per-cell dynamic eddy viscosity for LES) enters exactly as in
// faceFlux: face-averaged into mu, and scaled by kapScale = cp/Pr_t into
// kappa. nut == nullptr reproduces the molecular curvilinear flux bit-for-bit.
// wf: as in faceFlux -- advecting normal area flux Un - wf.S, pressure work on
// the absolute Un. nullptr is the inertial flux bit-for-bit.
// wS: the same correction already dotted with the face area (the node-psi MRF
// path, mrfFaceQNode); read only when wf is null. 0 is the inertial flux --
// Un - 0 is Un exactly, so the default changes nothing.
__device__ inline Flux faceFluxCurv(ViscView vv, PrimView q, CurvView m,
                                    const Offsets& o, std::size_t a,
                                    std::size_t b, int axis,
                                    const Real* nut = nullptr,
                                    Real kapScale = Real(0),
                                    const Real* wf = nullptr,
                                    Real wS = Real(0))
{
    const Real Sv[3] = {m.S[axis][0][a], m.S[axis][1][a], m.S[axis][2][a]};

    const Real rm = Real(0.5) * (q.rho[a] + q.rho[b]);
    const Real um[3] = {Real(0.5) * (q.u[a] + q.u[b]),
                        Real(0.5) * (q.v[a] + q.v[b]),
                        Real(0.5) * (q.w[a] + q.w[b])};
    const Real em = Real(0.5) * (q.e[a] + q.e[b]);
    const Real pm = Real(0.5) * (q.p[a] + q.p[b]);
    const Real Un = um[0] * Sv[0] + um[1] * Sv[1] + um[2] * Sv[2];
    const Real Ua =
        wf ? Un - (wf[0] * Sv[0] + wf[1] * Sv[1] + wf[2] * Sv[2]) : Un - wS;
    Flux f{rm * Ua, rm * Ua * um[0] + pm * Sv[0], rm * Ua * um[1] + pm * Sv[1],
           rm * Ua * um[2] + pm * Sv[2], rm * Ua * em + pm * Un};
    Real muf = vv.mu, kapf = vv.kappa;
    if (vv.sutherland) {
        const Real r = vv.ratio(Real(0.5) * (q.T[a] + q.T[b]));
        muf *= r;
        kapf *= r;
    }
    if (nut) {
        const Real nf = Real(0.5) * (nut[a] + nut[b]);
        muf += nf;
        kapf += nf * kapScale;
    }
    if (muf == Real(0)) return f;

    // d{u,v,w,T}/dxi at the face (dxi = 1): normal compact, tangential the
    // same two-cell average of central differences as the Cartesian flux.
    const Real* fld[4] = {q.u, q.v, q.w, q.T};
    Real dxi[4][3];
    for (int c = 0; c < 4; ++c) {
        dxi[c][axis] = fld[c][b] - fld[c][a];
        for (int d = 0; d < 3; ++d) {
            if (d == axis) continue;
            dxi[c][d] = Real(0.25) *
                (fld[c][a + o.p[d]] - fld[c][a + o.m[d]] +
                 fld[c][b + o.p[d]] - fld[c][b + o.m[d]]);
        }
    }
    Real gr[4][3]; // d{u,v,w,T}/dx_c
    for (int c = 0; c < 4; ++c)
        for (int cx = 0; cx < 3; ++cx)
            gr[c][cx] = dxi[c][0] * m.Ji[axis][0 + cx][a] +
                        dxi[c][1] * m.Ji[axis][3 + cx][a] +
                        dxi[c][2] * m.Ji[axis][6 + cx][a];
    const Real divu = gr[0][0] + gr[1][1] + gr[2][2];
    Real eflux = kapf * (gr[3][0] * Sv[0] + gr[3][1] * Sv[1] + gr[3][2] * Sv[2]);
    Real* fm[3] = {&f.mx, &f.my, &f.mz};
    for (int c = 0; c < 3; ++c) {
        Real ts = Real(0); // (tau . S)_c
        for (int d = 0; d < 3; ++d)
            ts += (muf * (gr[c][d] + gr[d][c]) -
                   (c == d ? Real(2.0 / 3.0) * muf * divu : Real(0))) * Sv[d];
        *fm[c] -= ts;
        eflux += um[c] * ts;
    }
    f.E -= eflux;
    return f;
}

// (i,j,k) convenience for the sparse callers; see the Cartesian twin above.
__device__ inline Flux faceFluxCurv(const GpuGrid& g, ViscView vv, PrimView q,
                                    CurvView m, int i, int j, int k, int axis,
                                    const Real* nut = nullptr,
                                    Real kapScale = Real(0),
                                    const MrfView* mz = nullptr)
{
    const Offsets o = offsetsAt(g, i, j, k);
    const std::size_t a = gidx(g, i, j, k);
    Real w[3];
    const Real* wf = nullptr;
    Real wS = Real(0);
    if (mz && mz->active) {
        if (mz->psi) {
            wS = mrfFaceQNode(*mz, g, i, j, axis);
        } else {
            mrfFaceVel(*mz, i, j, k, axis, w);
            wf = w;
        }
    }
    return faceFluxCurv(vv, q, m, o, a, a + o.p[axis], axis, nut, kapScale, wf,
                        wS);
}

} // namespace dns::gpu
