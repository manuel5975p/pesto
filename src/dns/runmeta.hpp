// The small value types that cross layer boundaries: what a run EMITS, where
// the body SITS, and the metadata carried into the run manifest.
//
// These live apart from config.hpp on purpose. output.hpp, scene.hpp and
// cli.hpp each needed two or three of these symbols and were including the
// whole of config.hpp -- RawConfig, Case, the validator's entire vocabulary --
// to get them. That made config.hpp a hub header and put the one backwards
// arrow in the layering diagram: `output` (the last layer) depending on the
// full surface of `config` (the first).
#pragma once

#include "dns/grid.hpp"

#include <string>

namespace dns {

// What a run writes. Which kinds a subcommand accepts is a validator question,
// not a type-system one.
enum class Emit { Stats, Bin, Png, Web, Shm, Volume, Slices, Vtk, Error,
                  Spectrum, Snapshot, None };

// Body centre and diameter in grid (world) coordinates, ghost offset included.
struct Placement
{
    double cx = 0, cy = 0, cz = 0, D = 0;
};

// Metadata carried into the run's .json / manifest.
struct RunMeta
{
    int nx = 0, ny = 0, nz = 0;
    double hx = 0, hy = 0, hz = 0;
    double D = 0, re = 0, mach = 0, frameDt = 0, cs = 0;
    bool les = false;
    std::string body, farField, physics;
    double kick = 0;
    std::string sgs;   // "smag" | "sigma" when les, "" for DNS
    std::string rans;  // "SA-neg" | "SST-2003m" | "SST-Vm" when RANS, "" otherwise

    // The six grid fields from the grid the artefacts are written on.
    void setGrid(const Grid& g)
    {
        nx = g.nx;
        ny = g.ny;
        nz = g.nz;
        hx = g.hx;
        hy = g.hy;
        hz = g.hz;
    }
};

// code: the process exit code this error maps to (1 CLI, 2 config, 3 input...).
struct ConfigError
{
    int code = 2;
    std::string field, message;
};

std::string format(const ConfigError&);

} // namespace dns
