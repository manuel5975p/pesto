#include "dns/vtk_writer.hpp"

#include <format>
#include <fstream>

namespace dns {

std::expected<void, std::string>
writeVtk(const Grid& g, const VectorField& U, const std::string& path)
{
    std::ofstream f(path);
    if (!f)
        return std::unexpected(std::format("cannot open '{}'", path));

    f << "# vtk DataFile Version 3.0\ndns snapshot\nASCII\n"
      << "DATASET STRUCTURED_POINTS\n"
      << std::format("DIMENSIONS {} {} {}\n", g.nx, g.ny, g.nz)
      << std::format("ORIGIN {} {} {}\n", 0.5 * g.hx, 0.5 * g.hy, 0.5 * g.hz)
      << std::format("SPACING {} {} {}\n", g.hx, g.hy, g.hz)
      << std::format("POINT_DATA {}\n", g.size())
      << "VECTORS U double\n";

    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const double uc = 0.5 * (U.u[g.idx(i, j, k)] + U.u[g.idx(i + 1, j, k)]);
                const double vc = 0.5 * (U.v[g.idx(i, j, k)] + U.v[g.idx(i, j + 1, k)]);
                const double wc = 0.5 * (U.w[g.idx(i, j, k)] + U.w[g.idx(i, j, k + 1)]);
                f << std::format("{} {} {}\n", uc, vc, wc);
            }

    // Close first: a buffered ENOSPC only surfaces at close, so testing the
    // stream while it is still open reports success on a truncated file.
    f.close();
    if (!f)
        return std::unexpected(std::format("write failed on '{}'", path));
    return {};
}

} // namespace dns
