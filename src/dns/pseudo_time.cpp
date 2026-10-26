#include "dns/pseudo_time.hpp"

#include "dns/blockband.hpp"
#include "dns/channel_line.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace dns {
namespace {

// alpha keeping x + alpha d >= (1 - frac) x for a decrease (x > 0).
double boundDecrease(double x, double d, double frac)
{
    return d < 0.0 ? frac * x / -d : 1.0;
}

// alpha keeping |alpha d| <= frac |x|, either sign.
double boundChange(double x, double d, double frac)
{
    return d != 0.0 ? frac * std::abs(x) / std::abs(d) : 1.0;
}

} // namespace

void PseudoTime::accepted(double alpha)
{
    assert(alpha > 0.0 && alpha <= 1.0);
    const bool undamped = alpha == 1.0;
    armed_ = undamped && cfl_ == p_.cflMax;
    if (undamped) cfl_ = std::min(p_.cflMax, cfl_ * p_.growth);
}

bool PseudoTime::rejected()
{
    armed_ = false;
    cfl_ *= p_.rejectFactor;
    return cfl_ >= p_.cflMin;
}

double stepLength(std::span<const double> q, std::span<const double> dq,
                  const ChannelLine& line)
{
    const int F = line.fields(), B = line.unknowns();
    assert(q.size() == std::size_t(line.rows()) * F &&
           dq.size() == std::size_t(line.rows() - 2) * B);
    const bool sst = line.sst();
    double alpha = 1.0;
    for (int j = 1; j <= line.rows() - 2; ++j) {
        const double* x = q.data() + std::size_t(j) * F;
        const double* d = dq.data() + std::size_t(j - 1) * B;
        alpha = std::min(alpha, boundChange(x[0], d[0], kImplicitStepRho));
        alpha = std::min(alpha, boundChange(x[4], d[3], kImplicitStepE));
        if (!sst) continue;
        if (x[5] > 0.0)
            alpha = std::min(alpha, boundDecrease(x[5], d[4], kImplicitStepK));
        alpha = std::min(alpha, boundDecrease(x[6], d[5], kImplicitStepOmega));
    }
    return alpha;
}

std::expected<std::vector<double>, std::string>
pseudoUpdate(const ChannelLine& line, std::span<const double> q,
             std::span<const double> R, double cfl)
{
    const int F = line.fields(), B = line.unknowns(), nf = line.rows() - 2;
    assert(R.size() == q.size() && q.size() == std::size_t(line.rows()) * F);
    BlockBand A = line.jacobian(q);
    std::vector<double> dtau(line.rows());
    line.pseudoStep(q, cfl, dtau);
    // -J, then 1/dtau_j on every diagonal entry of block row j.
    for (double& v : A.a) v = -v;
    std::vector<double> b(std::size_t(nf) * B);
    for (int j = 0; j < nf; ++j)
        for (int u = 0; u < B; ++u) {
            A.at(j, 0, u, u) += 1.0 / dtau[j + 1];
            b[std::size_t(j) * B + u] =
                R[std::size_t(j + 1) * F + ChannelLine::field(u)];
        }
    if (auto e = solveBlockBand(std::move(A), b); !e)
        return std::unexpected(e.error());
    return b;
}

} // namespace dns
