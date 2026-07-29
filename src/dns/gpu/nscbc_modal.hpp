// Angle-adaptive incoming-acoustic closure for the NSCBC x-hi outflow face.
//
// The Yoo-Im term L1 = -beta (Tp - rho c Tu) with a FIXED beta is the
// Engquist-Majda / Mur-2 operator in characteristic form: for an outgoing
// plane wave at incidence theta the exact incoming amplitude is
//   L1 = -i omega p cos(t)(1 - cos t),   Tp = i omega p sin^2(t),
// so the exact coefficient is  beta(t) = cos(t) / (1 + cos t),  which is 1/2
// at normal incidence (the shipped default) and falls to 0 at grazing. A fixed
// 1/2 is that function's value at t = 0, hence the O(theta^2) truncation the
// oblique reflection measures.
//
// beta(t) is recovered WITHOUT a frequency transform, because the outgoing
// acoustic amplitude and the transverse term carry the angle as their ratio:
//   L5 = i c kx p (1 + cos t),   Tp = i c^2 kt^2 p / omega
//   => Tp / L5 = (1 - cos t) / cos t  =>  cos t = L5 / (L5 + Tp).
// Both are in phase (each in quadrature with p), so the ratio is a real
// number, free of the time history the exact symbol sqrt(w^2/c^2 - kt^2) would
// otherwise demand: storage stays O(face cells) with no auxiliary stack.
//
// This module forms the ratio per transverse DCT-II mode of the whole face
// plane. With the shipped `--far nscbc` laterals (free-slip mirror walls) the
// cosine basis is the exact eigenbasis of the transverse operator, so each mode
// is one kt, the ratio is formed inside that mode, and a superposition of
// angles separates instead of averaging. NOT valid when the laterals are
// `--open` (all-characteristic; the problem is not separable).
//
// The POINTWISE form of the same ratio -- one face cell at a time, correct
// where a single wave crosses the cell and 0/0 at its zero crossings -- is not
// here: it needs no plane and no transform, so it lives in the NSCBC kernel
// itself (`comp_nscbc.hpp`, Config::betaAuto) and works on every open face
// rather than this one. Same coefficient, different locus; do not add a second
// copy of either.
//
// Only the x-hi face carries a modeled incoming ACOUSTIC amplitude on the
// shipped configuration (x-lo is the stiff inflow relax), so that one plane is
// all this owns. GPU-only.
#pragma once

#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/index.hpp"

#include <expected>
#include <memory>

namespace dns::gpu {

class NscbcModalAbc
{
public:
    struct Config
    {
        double gamma = 1.4;
        // Ceiling and degenerate-case fallback for the derived coefficient
        // (`--nscbc-beta`). beta(t) <= 1/2 everywhere, so the shipped 0.5
        // never binds; a smaller value caps the whole face as it does today.
        double beta = 0.5;
    };

    // nx >= 2, ny >= 2; needs an FFT plan of length ny.
    static std::expected<NscbcModalAbc, hipError_t> make(const GpuGrid& g,
                                                         Config c);

    NscbcModalAbc(NscbcModalAbc&&) noexcept;
    NscbcModalAbc& operator=(NscbcModalAbc&&) noexcept;
    ~NscbcModalAbc();

    // Recompute the face plane from the live stage primitives; call once per RK
    // stage immediately before the NSCBC kernel that reads plane().
    // Pre: every pointer is a full g.size() field of the same grid.
    void compute(const Real* rho, const Real* u, const Real* v, const Real* w,
                 const Real* p);

    // beta_eff * (Tp - rho c Tu) at x-hi, indexed j + ny*k: the term the NSCBC
    // kernel subtracts from the incoming amplitude. Valid after compute().
    const Real* plane() const;

    // Device bytes this closure owns (buffers + FFT work area).
    std::size_t bytes() const;

private:
    struct Impl;
    explicit NscbcModalAbc(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace dns::gpu
