// RANS on the FV path: the face and cell algebra of the turbulence scalars
// (docs/fvcomp.md "RANS"), shared by the FP64 oracle dns/fvref and the device
// gpu/fv_comp. Host and device (DNS_HD), templated on the scalar T. The model
// algebra is dns/rans.hpp, the ONE home of every constant: SA-neg with ft2,
// SST-2003m, SST-Vm, the structured solver's closures.
//
// Restates gpu/rans.hip on an arbitrary face (a face is (S, d, corr), its mass
// flux the mean flow's F[0]): first-order upwind of the SPECIFIC scalar, face
// coefficients formed on the face, cell terms off the FV cell gradients. A
// boundary is a FACE FLUX: Wall = phi_w Dirichlet diffusion on the molecular mu
// (mu_t 0 on the face, no convection), Slip = no flux, open faces = the mean
// mass flux carrying the upwind value (no diffusion).
#pragma once

#include "dns/fvflux.hpp"
#include "dns/rans.hpp"
#include "dns/sgs.hpp"

namespace dns {

// Gradient fields per cell and boundary face. SA: nu~, rho. SST: k, omega.
inline constexpr int kFvRansFields = 2;

// Density on a wall face: the EOS at the wall temperature, else the owner's.
template <class T> DNS_HD T fvWallRho(const FvBc& bc, const FvPrim<T>& B, double Rgas)
{
    return bc.isothermal ? B.p / (T(Rgas) * B.temp) : B.rho;
}

// The temperature a Wall face's mu is read at (FvRef::faces, bndViscous).
template <class T> DNS_HD T fvWallTemp(const FvBc& bc, const FvPrim<T>& A, const FvPrim<T>& B)
{
    return bc.isothermal ? T(0.5) * (A.temp + B.temp) : A.temp;
}

// TMR omega_w = 60 nu_w / (beta1 d1^2). d1: the owner centre's wall distance.
template <class T> DNS_HD T fvOmegaWall(T mu, T rhoW, T d1)
{
    return T(sst::omegaWallFactor) * mu / (rhoW * d1 * d1);
}

// The gradient's face values on one non-empty boundary face. pA: the owner's
// fields; inf: the freestream specific scalars; muW, d1: read on Wall only.
// Open faces: upwind on the sign of the boundary state's normal velocity.
template <class T, class G>
DNS_HD void fvRansBndFields(bool sst, const FvBc& bc, const FvPrim<T>& B, const G S[3], const T pA[2],
                            const T inf[2], T muW, T d1, double Rgas, T out[2])
{
    if (bc.kind == FvBcKind::Wall) {
        const T rw = fvWallRho(bc, B, Rgas);
        out[0] = T(0.0);
        out[1] = sst ? fvOmegaWall(muW, rw, d1) : rw;
        return;
    }
    if (bc.kind == FvBcKind::Slip) {
        out[0] = pA[0];
        out[1] = sst ? pA[1] : B.rho;
        return;
    }
    const bool leaving = B.u[0] * T(S[0]) + B.u[1] * T(S[1]) + B.u[2] * T(S[2]) >= T(0.0);
    out[0] = leaving ? pA[0] : inf[0];
    out[1] = sst ? (leaving ? pA[1] : inf[1]) : B.rho;
}

// First-order upwind on the mean mass flux mdot through the face a -> b.
template <class T> DNS_HD T fvUpwind(T mdot, T pa, T pb)
{
    return (mdot > T(0.0) ? mdot : T(0.0)) * pa + (mdot < T(0.0) ? mdot : T(0.0)) * pb;
}

// Vorticity magnitude of gd[i][j] = d u_i / d x_j.
template <class T> DNS_HD T fvVorticity(const T gd[3][3])
{
    using std::sqrt;
    const T w0 = gd[2][1] - gd[1][2], w1 = gd[0][2] - gd[2][0], w2 = gd[1][0] - gd[0][1];
    return sqrt(w0 * w0 + w1 * w1 + w2 * w2);
}

// One cell's closure: mu_t, F1 (SST), the pointwise rhs per unit volume and the
// frozen implicit rate of each scalar (R[1], J[1] SST only).
template <class T> struct FvRansCell
{
    T mut, f1, R[2], J[2];
};

// SA-neg at a cell: mu_t, source, the cb2 and grad rho terms (gpu/rans.hip
// kSaStageCurv). phi = nu~; gd: velocity gradient; gn, gr: grad nu~, grad rho.
// Pre: rho, mu, d > 0.
template <class T>
DNS_HD FvRansCell<T> fvSaCell(T rho, T phi, T mu, const T gd[3][3], T d, const T gn[3], const T gr[3])
{
    const T nu = mu / rho;
    const RansPoint<T> P = saSource(phi, nu, fvVorticity(gd), d);
    const T gg = gn[0] * gn[0] + gn[1] * gn[1] + gn[2] * gn[2];
    const T rg = gr[0] * gn[0] + gr[1] * gn[1] + gr[2] * gn[2];
    FvRansCell<T> c;
    c.mut = saMut(rho, phi, nu);
    c.f1 = T(0.0);
    c.R[0] = rho * P.R + T(sa::cb2 / sa::sigma) * rho * gg - T(1.0 / sa::sigma) * saDiffCoef(nu, phi) * rg;
    c.R[1] = T(0.0);
    c.J[0] = P.J;
    c.J[1] = T(0.0);
    return c;
}

// SST at a cell (kSstMutCurv + the sstSource call of kSstStageCurv). gk, gw:
// grad k, grad omega. Pre: rho, mu, d, omega > 0.
template <class T>
DNS_HD FvRansCell<T> fvSstCell(RansModel m, T rho, T k, T omega, T mu, const T gd[3][3], T d, const T gk[3],
                               const T gw[3])
{
    const T rate = m == RansModel::SstVm ? fvVorticity(gd) : sgsRate(0, gd);
    const T gkw = gk[0] * gw[0] + gk[1] * gw[1] + gk[2] * gw[2];
    const T cd = sstCd(rho, gkw, omega, T(sstCdFloor(m)));
    const SstBlend<T> b = sstBlend(k, omega, d, mu / rho, rho, cd);
    FvRansCell<T> c;
    c.mut = sstMut(rho, k, omega, rate, b.F2);
    c.f1 = b.F1;
    const SstPoint<T> P = sstSource(m, rho, k, omega, rate, c.mut, b.F1, gkw);
    c.R[0] = P.Rk;
    c.R[1] = P.Rw;
    c.J[0] = P.Jk;
    c.J[1] = P.Jw;
    return c;
}

// SA's internal face flux (convective minus diffusive). mdot: the mean flow's
// F[0]; muf: mu at the face T; rf: the face density; gS: corrected grad nu~ . S.
template <class T> DNS_HD T fvSaFaceFlux(T mdot, T pa, T pb, T muf, T rf, T gS)
{
    const T nuf = muf / rf, ntf = T(0.5) * (pa + pb);
    return fvUpwind(mdot, pa, pb) - T(1.0 / sa::sigma) * rf * saDiffCoef(nuf, ntf) * gS;
}

// One SST scalar's. sig: its blended sigma(F1_f); mutf: the face mean of mu_t.
template <class T> DNS_HD T fvSstFaceFlux(T mdot, T pa, T pb, T muf, T sig, T mutf, T gS)
{
    return fvUpwind(mdot, pa, pb) - (muf + sig * mutf) * gS;
}

} // namespace dns
