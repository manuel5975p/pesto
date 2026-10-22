// A validated Case -> the body's sdf, cut geometry and world boxes.
//
// This is the geometry layer's entry point for the CLI: everything below it
// (bodies, geometry, meshsdf, scene) works in shapes and grids, everything
// above it works in runs. It lived in apps/dns_main.cpp, which put the bodies'
// MEASURED constants -- dimple footprint and depth, fence thickness, the trip
// spike geometry and its margins -- in the app layer, where CLAUDE.md says
// they must not be ("validated physics constants live at their definitions
// with the measured bracket recorded"). Their definitions are here.
//
// The per-body sample budgets are carried through VERBATIM and are not
// unified: changing one shifts the cut, and a shifted cut moves every measured
// coefficient (masterplan_recompose §6.11).
#pragma once

#include "dns/config.hpp"
#include "dns/cutcell.hpp"
#include "dns/sdf.hpp"

#include <array>
#include <expected>
#include <string>
#include <utility>
#include <vector>

namespace dns {

struct Scene;

struct CaseGeometry
{
    CutGeometry cut;
    Sdf sdf;
    double D = 1.0;             // characteristic length actually realised
    std::array<double, 6> bbox{};
    bool hasBbox = false;

    // Scenes only: per-object world boxes for the per-object force tallies.
    std::vector<std::pair<std::string, std::array<double, 6>>> objectBoxes;
    // The body spans the full z extent: the z-invariant kinds (cylinder,
    // lens, fence, floor) and a scene whose every object does.
    bool zSpans = false;
};

// Builds the body's sdf and cut from a validated Case. `scene` is required iff
// c.bodyKind == BodyKind::Scene (main parses the file in phase 1.5, before
// validate, because the filesystem is not the validator's business).
//
// Touches the filesystem only for BodyKind::Mesh and Scene, so a failure here
// is an INPUT error (exit 3).
std::expected<CaseGeometry, std::string>
buildCaseGeometry(const Case& c, const Scene* scene = nullptr);

} // namespace dns
