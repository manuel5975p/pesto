// Picking a scene object with the ray, analytically.
//
// The obvious implementation -- ray vs the object's AABB -- is wrong here in a
// way that is easy to miss: a cylinder, cube or lens spans the FULL box depth
// by construction (dns/scene.cpp), so its AABB is a slab across the entire
// domain. Clicking eight diameters away from a cylinder selects it, and an
// object genuinely behind another can never be picked because every candidate's
// near hit is the same z = 0 face.
//
// So each kind is tested against its actual shape. Pure functions of the object
// and the grid -- no Build, no camera, no GPU -- so `dns-ui --check` drives them.
#pragma once

#include "dns/config.hpp"
#include "dns/scene.hpp"
#include "ui/math.hpp"

#include <array>
#include <span>

namespace ui {

// Nearest surface hit along the ray, or false. `tHit` is 0 when the ray origin
// is already inside the shape, so an object you are zoomed inside always wins.
//
// `meshBox` is the fallback for mesh objects: their true shape needs the loaded
// file, which the UI does not have, so their (tight) published box is used.
bool rayObject(const dns::SceneObject&, const dns::Placement& ref,
               const dns::Grid&, const std::array<double, 6>* meshBox,
               const Ray&, float& tHit);

// Repeated clicks at the same pixel step THROUGH overlapping candidates, which
// is the only way to reach an object exactly behind another. `hits` is sorted
// near-to-far; `current` is what is selected now; `sameSpot` says the click
// landed within a few pixels of the last one. Returns the index to select.
int cycleSelection(std::span<const int> hits, int current, bool sameSpot);

} // namespace ui
