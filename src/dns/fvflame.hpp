// A planar premixed flame on any FaceMesh (docs/fvcomp.md "Species"): the IC along
// a direction and the structured runner's column stats, restated on slabs normal
// to that direction. Host, geometry layer. A slab holds the cells whose centroid
// falls in it, so a Cartesian mesh's slabs ARE the structured columns. The slot-jet
// diffusion flame the same way: per-face inflow rows, IC, measureJet's stats.
#pragma once

#include "dns/diffusion.hpp"
#include "dns/facemesh.hpp"
#include "dns/fvcomp_config.hpp"
#include "dns/fvflux.hpp"
#include "dns/fvref.hpp"

#include <array>
#include <expected>
#include <functional>
#include <string>
#include <vector>

namespace dns {

// One slab's FP64 means, Sum(V f)/Sum V, cells ascending.
struct FvFlameSlab
{
    double s = 0, T = 0, rho = 0, u = 0, p = 0, ry = 0, omega = 0, vol = 0;
};

// One stats row: SL = burn / (A rho Y_ref), xf the mean-T mid crossing from the
// mesh's low end, dth = (Tb - Tu)/max slope, as runner_flame's measure(), slabs
// for columns. energy = Sum V (E + q rho Y), hits = cells with Y outside [-1e-6,
// 1 + 1e-6]. uRef, pRef, TRef: the slab 2 delta_th inside the low end.
struct FvFlameMeasure
{
    double SL = 0, xf = NAN, dth = 0, Tmax = 0, mass = 0, energy = 0, fuel = 0, burn = 0;
    double omegaEdge = 0, hits = 0, yMin = 0, yMax = 0, uRef = 0, pRef = 0, TRef = 0;
    std::vector<FvFlameSlab> slabs;
};

// What a probe reads off the model. dir: unit normal of the flame, burnt side +.
struct FvFlameSpec
{
    std::array<double, 3> dir{1, 0, 0};
    double Tu = 1, sigma = 5, dth = 1, gamma = 1.4, Rgas = 1, q = 0;
    int fuel = 0;
    // The burnt side's Y of the fuel row (model C2: Y_F,rem): SL's reference is
    // rho (Y - yBurnt), the structured FlameRun::yFBurnt; 0 = the C0 expression.
    double yBurnt = 0;
    // A [nozzle] run (masterplan_combnozzle R9): S_L on the reference slab's own area A_ref =
    // V_ref/h, x_f, delta_th, Tmax, omegaEdge and the drift guard on the chamber bins only.
    bool chamber = false;
};

// The low-Mach state's cell fields (gpu/fv_incomp DeviceFvLowMach::cells), [nC] but
// ru [3 nC]; p is the dynamic pressure. All non-null, FaceMesh numbering.
struct FvFlameLowMachCells
{
    const std::vector<double>*rho, *ry, *ru, *T, *p, *omega;
};

class FvFlameProbe
{
public:
    // Slab width: the mean |d . dir| over internal faces with |d . dir| >= |d|/2,
    // fitted to a whole number of slabs. Pre: fm outlives the probe. Errors: no
    // such face, a mesh shorter than two slabs; a chamber probe whose reference slab
    // lies past the constant-area bins.
    static std::expected<FvFlameProbe, std::string> make(const FaceMesh& fm, const FvFlameSpec& spec);

    // Pre: s has the mesh's cells and the mean + species rows; omega per cell.
    FvFlameMeasure measure(const FvState& s, const std::vector<double>& omega) const;

    // The same measure of a low-Mach state: energy = Sum V (cp rho T + q rho Y), pRef
    // and the slabs' p the dynamic pressure. Pre: spec.gamma, Rgas give that cp.
    FvFlameMeasure measureLowMach(const FvFlameLowMachCells& q) const;

    // S_L's area: vol/L, or A_ref on a chamber probe. meanArea: vol/L always.
    double area() const { return area_; }
    double meanArea() const { return meanArea_; }
    // The guarded length: the mesh's, or the chamber bins' on a chamber probe.
    double length() const { return nCh_ < nBin_ ? double(nCh_) * h_ : hi_ - lo_; }
    double slab() const { return h_; }
    double lo() const { return lo_; } // the low end along spec().dir
    const FvFlameSpec& spec() const { return spec_; }
    // Slab of cell c, the slab count, the chamber bins [0, chamberBins) (all bins off a chamber probe).
    int binOf(int c) const { return bin_[std::size_t(c)]; }
    int bins() const { return nBin_; }
    int chamberBins() const { return nCh_; }

private:
    // The slab means and the flame's numbers from the slab sums.
    void finish(FvFlameMeasure& o, std::vector<FvFlameSlab> b) const;

    const FaceMesh* fm_ = nullptr;
    FvFlameSpec spec_;
    std::vector<int> bin_; // per cell
    double lo_ = 0, hi_ = 0, h_ = 0, area_ = 0, meanArea_ = 0;
    int nBin_ = 0, nCh_ = 0;
};

// What the jet's rows, IC and probe read. dir: streamwise, lift: lateral, origin:
// the slot's centre on the inflow plane; a point sits at x = (C - origin).dir, y =
// (C - origin).lift + geom.yc in the structured box's coordinates, y - yc wrapped to
// the mesh's lateral period (a sheared cyclic mesh leaves the slot's y). p0 = T0 (R 1).
// skip: cells at each end TmaxIn ignores, in slabs.
// --mode candle (candle): the stats append uMax, uOutMin; g = -gravity dir (zero: the
// jet bit for bit), the hydrostatic ambient fvJetAmbient; quiescent: every inflow face
// and the IC at rest (quiescentInflow), no extinction guard.
struct FvJetSpec
{
    std::array<double, 3> dir{1, 0, 0}, lift{0, 1, 0}, origin{};
    DiffusionModel model{};
    BunsenGeom geom{};
    double T0 = 1, gamma = 1.4, Rgas = 1, q = 0;
    int skip = 4;
    bool candle = false, quiescent = false;
    std::array<double, 3> g{};
};

// The candle's ambient column: isothermal T0, p0 = R T0 (rho0 1) at the slot's centre on
// the inflow plane, so h = (x - origin).dir: the structured candle's candleAmbientP numbers.
FvAmbient fvJetAmbient(const FvJetSpec& js);

// measureJet's numbers and the runner's. energy = Sum V (E + q rho Y_F), pMean the
// volume mean p/p0 (under gravity Sum V p / Sum V p_a: the structured candle's ambient
// reference); hits: cells with Z, Y_F or Y_O outside [-1e-6, 1 + 1e-6]; uMax max |u|.
struct FvJetMeasure
{
    double xTip = NAN, xBase = NAN, half = NAN, Tmax = 0, TmaxIn = 0, burn = 0, mass = 0, energy = 0, pMean = 0;
    double zMin = 0, zMax = 0, yfMin = 0, yoMin = 0, hits = 0, uMax = 0;
};

class FvJetProbe
{
public:
    // Slabs of the mean |d.dir|, as FvFlameProbe's; the axis: cells within the mean
    // |d.lift| of y = yc, the two columns either side on a Cartesian mesh. Pre: fm
    // outlives the probe. Errors: no face along dir or lift, under four slabs.
    static std::expected<FvJetProbe, std::string> make(const FaceMesh& fm, const FvJetSpec& spec);

    // Pre: s has the mesh's cells and rows Y_F, Y_O at 5, 6; omega per cell.
    FvJetMeasure measure(const FvState& s, const std::vector<double>& omega) const;

    double length() const { return double(nBin_) * h_; }
    const FvJetSpec& spec() const { return spec_; }

private:
    const FaceMesh* fm_ = nullptr;
    FvJetSpec spec_;
    std::vector<int> bin_;                 // per cell
    std::vector<double> x_, y_;            // per cell, from the slot's inflow centre, y wrapped
    std::vector<std::vector<int>> slabs_;  // cells per slab, ascending y
    double h_ = 0, hLat_ = 0, lo_ = 0, pAmbient_ = 0; // pAmbient_: Sum V p_a (gravity only)
    int nBin_ = 0;
};

// The table of a jet run: a copy of bc (a row per patch) plus one row per face of
// every `jet` patch, diffusionInflow(y) at the face centre (quiescentInflow if
// js.quiescent): its u, T, y[0], y[1] = Y_F, Y_O, `diffusive`. Every Outflow row is
// `characteristic` and carries its owner's Y both ways; under gravity (the candle) a
// backflow is the ambient's air instead (y = {0, Y_O2}, T0: air drawn down through the
// top, the structured candle's ambientHi), and every Far / Outflow face gets the ambient
// row of fvAmbientRows. faceRow: per boundary face, its row.
using FvJetRows = FvBcTable;
FvJetRows fvJetRows(const FaceMesh& fm, const FvBcSpec& spec, std::vector<FvBc> bc, const FvJetSpec& js);

// y of every cell off the slot's axis, wrapped: a 2D mesh with cyclic faces has the
// lateral period Sum V / (length along dir x depth).
std::vector<double> fvJetLateral(const FaceMesh& fm, const FvJetSpec& js);

// diffusionIC on any mesh: the inflow row at the cell's y, extruded along dir;
// ignition > 0: the short burning column of --ignition. Rows rho, m, E, rho Y_F,
// rho Y_O. Under gravity p = p_a(C) (fvJetAmbient), rho = p_a/(R T): the structured
// candle's IC (horizontally uniform p).
FvState fvJetState(const FaceMesh& fm, const FvJetSpec& js, double ignition);

// The FV state of t(s, C) = T/Tu at every cell centroid C, s = (C - origin) . dir,
// through premixedPoint: rows rho, m, E, rho Y(fuel). Flow along dir, nY species,
// the others 0.
FvState fvFlameState(const FaceMesh& fm, const FvFlameSpec& spec, std::array<double, 3> origin, double m, int nY,
                     const std::function<double(double)>& t);

} // namespace dns
