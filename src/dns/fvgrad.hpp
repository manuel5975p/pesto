// FV face geometry and cell gradients on a FaceMesh (docs/mesh.md): the
// frame-consistent face offsets, the face-gradient correction vectors and the
// cell-gradient weights every FV pass reads. Host FP64, threaded and
// byte-identical at any thread count; the GPU path uploads these tables.
//
// Cell gradients are written on DIFFERENCES,
//   grad_c = sum_f w_{c,f} (phi_other - phi_c),
// phi_other the neighbour's value (internal face) or the boundary face value,
// so a uniform field has exactly zero gradient on any mesh.
//   LeastSquares: w = G_c^-1 d / |d|^2, G_c = sum d d^T / |d|^2 (OpenFOAM
//                 `leastSquares`); exact for linear fields on any mesh.
//   GreenGauss:   w = (1 - lambda) S / V with OpenFOAM's `linear` weights
//                 lambda (`Gauss linear`); exact for linear fields only where
//                 the face centre lies on the line of centres.
// `empty` faces never enter; in 2D G is inverted in the 2D subspace, so the
// emptyAxis component of every weight is EXACTLY 0.
#pragma once

#include "dns/facemesh.hpp"
#include "dns/fvflux.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace dns {

enum class FvGradScheme : std::uint8_t { LeastSquares, GreenGauss };

// Per-face geometry of the FV kernels, nI + nB entries each.
struct FvFaceGeom
{
    FvFaceGradRule rule{};
    double alpha = 1;
    // Cf - C_own in the OWNER's frame: on a cyclic face whose Cf lies in the
    // neighbour's frame (FaceMesh contract) it is translated by
    // C_own + d - C_nei. The neighbour's offset is rOwn - d.
    std::array<std::vector<double>, 3> rOwn;
    // fvFaceGrad's correction vector: alpha d/|d|^2 (AlongD) or alpha S/(S.d)
    // (AlongNormal); 0 on `empty` faces.
    std::array<std::vector<double>, 3> corr;
    std::vector<std::uint8_t> active; // 0 on `empty` faces, else 1
};

// Pre: fm built by FaceMesh. Errors (user mesh): S.d <= 0 on an active face,
// alpha <= 0.
std::expected<FvFaceGeom, std::string> fvFaceGeom(const FaceMesh& fm, FvFaceGradRule rule,
                                                  double alpha, int threads = 0);

// Cell-gradient weights on differences (see the file comment).
struct FvGradWeights
{
    FvGradScheme scheme{};
    std::array<std::vector<double>, 3> own; // nI + nB: owner side (boundary: toward the face value)
    std::array<std::vector<double>, 3> nei; // nI: neighbour side
};

// Errors (user mesh): a singular least-squares matrix (a cell whose active
// faces do not span the solution space).
std::expected<FvGradWeights, std::string> fvGradWeights(const FaceMesh& fm, const FvFaceGeom& geom,
                                                        FvGradScheme scheme, int threads = 0);

// Gradient of one field over cells [lo, hi): phi per cell, phiB per boundary
// face (read on active faces only), g[3] per cell. CSR face order, so the
// result is independent of how [0, nC) is split.
void fvCellGradient(const FaceMesh& fm, const FvFaceGeom& geom, const FvGradWeights& w,
                    std::span<const double> phi, std::span<const double> phiB,
                    const std::array<double*, 3>& g, std::size_t lo, std::size_t hi);

} // namespace dns
