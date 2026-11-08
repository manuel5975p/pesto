// --shock on the device (dns/shockflux.hpp, docs/shock.md): the sensor bytes and
// the additive face pass after the species stage. No existing kernel knows about
// it: absent = no launch, a silent sensor = no write, the solver bit for bit.
// Structured uniform Cartesian mesh only.
#pragma once

#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/index.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/shockflux.hpp"

#include <cmath>
#include <cstdint>
#include <expected>
#include <vector>

namespace dns::gpu {

struct PrimView;        // gpu/internal/comp_flux.hpp (HIP TUs only)
struct StageCombine;    // gpu/comp.hpp
struct DeviceCompState; // gpu/comp.hpp

// What DeviceComp::enableShock installs.
struct ShockSetup
{
    // DEVICE mask, 1 on every solid (ghost or deep) cell, owned by the caller
    // for the solver's lifetime. The sensor never reads a solid cell and a face
    // with a solid side carries no correction. Null = no wall.
    const std::uint8_t* solid = nullptr;
    // Axes not periodic: NSCBC or mirror rows at both ends. The sensor does not
    // wrap there and no face touching row 0 or n - 1 carries a correction.
    bool open[3] = {false, false, false};
    // Of `open`, the axes whose end rows are NSCBC rows (ShockRow::openMach).
    bool nscbc[3] = {false, false, false};
    bool apply = true;      // false: --shock sense, the raw v is kept
    bool testForce = false; // test only: sigma 1 on every face (teeth)
};

// Per-cell region bytes of a stats row, with the two ranges it counts: the
// cavity [cavLo, cavHi] and the lip column [lipLo, lipHi]; farCol: the first column
// of the lip's window, 3 columns upstream of it. Null region = none.
struct ShockRegions
{
    const std::uint8_t* region = nullptr;
    std::uint8_t cavLo = 0, cavHi = 0, lipLo = 0, lipHi = 0;
    int farCol = 0;
    int throatCol = -1; // the narrowest cavity column; < 0 = none
    std::uint8_t jetByte = 0; // the region the jet leaves into: ShockRow::jetMach
};

// One stats row's shock numbers on one state. top: max sigma, or max raw v
// under --shock sense. cavCells, cavTop: the cavity range. lipCells: the lip
// range. cavRest: cavity cells below lipLo, the cavity without its lip. cavFar:
// cavity cells upstream of farCol, the cavity without the lip's neighbours.
// exitMach: max local |u|/c on region byte lipLo (the chamber's exit column).
// throat*: the cavity cells of throatCol: max local Mach, mean static p, and the
// isentropic total pressure, mass-flux mean.
struct ShockRow
{
    double cells = 0, top = 0, rhoMin = HUGE_VAL, pMin = HUGE_VAL, openMach = 0;
    double cavCells = 0, cavTop = 0, lipCells = 0, cavRest = 0, exitMach = 0, cavFar = 0;
    double throatMach = 0, throatP = 0, throatP0 = 0;
    double jetMach = 0; // max local Mach on jetByte
};

// One cell of the sensor's report: raw v and the flattened index.
struct ShockPeak
{
    double v = 0;
    std::size_t cell = 0;
};

// The two byte buffers (2 B/cell: s, active; +4 B/cell raw under --shock sense).
class DeviceShock
{
public:
    // Pre: g outlives the object, n >= 3 on every non-degenerate axis.
    static std::expected<DeviceShock, hipError_t> make(const GpuGrid& g, const ShockSetup& s);

    bool applies() const { return set_.apply; }

    // Sensor bytes and the active mask of pressure field p. Pure function of p.
    void sense(const Real* p);
    // Cells with a nonzero byte. FP64 reduction; sense() ran.
    double count(ReduceScratch& r) const;
    // Adds D on every sensed face of `out` through `cmb`: F_KG + sigma (F_HLL -
    // F_KG). Y: nY Y_k caches of q's state, with their donor stream into out.y.
    // Pre: sense() ran on this step's input; q primitives of the stage input.
    void faces(const PrimView& q, const StageCombine& cmb, Real gamma, const Real* const* Y, int nY,
               DeviceCompState& out) const;
    // The stats row of the last sense(). p, u, v, w: primitives of `rho`'s state.
    // Solid cells are not in rhoMin, pMin.
    ShockRow row(const Real* rho, const Real* u, const Real* v, const Real* w, const Real* p, Real gamma,
                 const ShockRegions& reg, ReduceScratch& r) const;
    // The --shock hll guard, on a conserved state: max |u_n|/c over the NSCBC rows
    // the Super row does not own (A row may be supersonic on x-hi while it flows
    // out, nowhere else). +inf on a non-finite cell. No side effect on the state.
    double rowGuard(const Real* rho, const Real* mx, const Real* my, const Real* mz, const Real* E, Real gamma,
                    ReduceScratch& r) const;
    // --shock sense: the largest raw v of the last sense() in burning cells, omega
    // above 1% of its running peak: the flame's reading, wherever it is swept.
    // omega: the rate of the sensed state; omegaPeak: the run's, updated.
    double burnMax(const Real* omega, double& omegaPeak, ReduceScratch& r) const;
    // Cells of the last sense() burning above 1% of the peak, either mode: gate 29's
    // "no cell sensed in burning cells" under hll, which keeps no raw v.
    double burnCells(const Real* omega, double& omegaPeak, ReduceScratch& r) const;
    // The bytes of the last sense(), g.size() entries (a test and report read).
    [[nodiscard]] hipError_t bytes(std::vector<std::uint8_t>& out) const;
    // --shock sense: the k largest raw v of the last sense(), largest first, ties
    // by cell index (a host download, for the end-of-run report). Pre: !applies().
    std::expected<std::vector<ShockPeak>, hipError_t> peaks(std::size_t k) const;

private:
    DeviceShock() = default;

    GpuGrid g_{};
    ShockSetup set_{};
    DeviceBuffer<std::uint8_t> s_, active_;
    DeviceScalar raw_; // --shock sense only
};

} // namespace dns::gpu
