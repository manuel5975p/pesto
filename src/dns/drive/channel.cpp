// dns channel.
#include "dns/drive/internal.hpp"

#include "dns/config.hpp"
#include "dns/flow_ic.hpp"
#include "dns/output.hpp"
#include "dns/remap.hpp"
#include "dns/transport.hpp"
#include "dns/walldist.hpp"

#if DNS_WITH_GPU
#include "dns/gpu/runner_channel.hpp"
#include "dns/gpu/comp.hpp"
#endif

#include <format>
#include <numbers>
#include <string>
#include <vector>

namespace dns::drive {

using std::numbers::pi;

int runChannel(const Case& c)
{
    if (!haveGpu(c))
        return die(exitNoGpu, "no GPU device (dns channel is GPU-only)");
#if DNS_WITH_GPU
    const auto& g = c.grid;
    const double gamma = kGammaAir, delta = 1.0, rho0 = 1.0;
    const ChannelStretchY remap{double(g.ny) * g.hy, g.ny, c.stretchBeta,
                                delta};

    // IC: Poiseuille + solenoidal curl(psi) trip (dns/flow_ic.hpp), or under
    // --rans the untripped mean start on the model's own wall distance.
    auto host = CompState::zeros(g);
    std::vector<float> mask(g.size(), 1.0f);
    std::vector<double> wallDist;
    std::array<std::vector<double>, 2> icTurb;
    if (c.rans == RansModel::None)
        channelIC(c, remap, delta, rho0, gamma, host, mask);
    else
    {
        wallDist = channelWallDistance(
            CurvMesh::channelY(g, c.stretchBeta, delta), delta);
        channelRansIC(c, wallDist, rho0, gamma, host, icTurb, mask);
    }

    FrameGeometry fg = FrameGeometry::centred(g, 2 * delta);
    fg.mask = std::move(mask);

    const auto oc = outputConfig(
        c, std::format("{{\"app\":\"dns\",\"retau\":{},\"mach\":{}",
                       c.reTau, c.mach));
    auto out = RunOutput::make(oc, c.meta, g, std::move(fg));
    if (!out) return die(exitOutput, out.error());

    gpu::ChannelRun rr;
    rr.mu = rho0 * c.nu;
    rr.reTau = c.reTau;
    rr.ubEst = c.ubEst;
    rr.delta = delta;
    rr.pRef = c.pRef;
    rr.Twall = c.pRef; // rho = R = 1
    rr.isothermal = c.wallIso;
    rr.stretchBeta = c.stretchBeta;
    rr.cfl = c.cfl;
    rr.tEnd = c.tEnd;
    rr.dtRecompute = c.dtRecompute;
    rr.profileDt = c.profileDt;
    rr.rans = c.rans;
    rr.steadyTol = c.steadyTol;
    rr.steadySteps = c.steadySteps;
    rr.implicitSteady = c.steadyScheme == SteadyScheme::Implicit;
    rr.compensated = c.compensated;
    rr.les = c.cs;
    rr.sgs = c.sgsModel;
    rr.wallDist = std::move(wallDist);
    rr.icTurb = std::move(icTurb);

    auto res = gpu::runChannel(g, host, rr, *out);
    if (!res) return die(res.error().code, res.error().message);
    return reportDone(*res, *out);
#else
    return exitNoGpu;
#endif
}



} // namespace dns::drive
