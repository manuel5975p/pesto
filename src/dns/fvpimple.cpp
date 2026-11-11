#include "dns/fvpimple.hpp"

#include <algorithm>
#include <cmath>
#include <format>

namespace dns {

namespace {
constexpr double kSmall = 1e-15;      // OpenFOAM SMALL
constexpr double kLabelMax = 2147483647.0; // labelMax (label = int32)
} // namespace

FoamClock::FoamClock(const FvTimeCtl& c) : c_(c), dt_(c.deltaT) {}

// Time::adjustDeltaT verbatim (startTime 0).
void FoamClock::adjustDeltaT()
{
    if (c_.writeControl != FvWriteControl::AdjustableRunTime) return;
    const double timeToNextWrite = std::max(0.0, double(writeIndex_ + 1) * c_.writeInterval - (t_ - 0.0));
    const double nSteps = timeToNextWrite / dt_;
    if (!(nSteps < kLabelMax)) return;
    const long nStepsToNextWrite = std::max(1L, std::lround(nSteps));
    const double newDeltaT = timeToNextWrite / double(nStepsToNextWrite);
    if (newDeltaT >= dt_) dt_ = std::min(newDeltaT, 2.0 * dt_);
    else dt_ = std::max(newDeltaT, 0.2 * dt_);
}

void FoamClock::setInitialDeltaT(double coNum)
{
    if (!c_.adjust) return;
    if (index_ != 0 || !(coNum > kSmall)) return;
    dt_ = std::min(c_.maxCo * dt_ / coNum, std::min(dt_, c_.maxDeltaT));
    adjustDeltaT();
}

void FoamClock::setDeltaT(double coNum)
{
    if (!c_.adjust) return;
    const double maxDeltaTFact = c_.maxCo / (coNum + kSmall);
    const double deltaTFact = std::min(std::min(maxDeltaTFact, 1.0 + 0.1 * maxDeltaTFact), 1.2);
    dt_ = std::min(deltaTFact * dt_, c_.maxDeltaT);
    adjustDeltaT();
}

bool FoamClock::advance()
{
    t_ = t_ + dt_;
    ++index_;
    switch (c_.writeControl) {
    case FvWriteControl::TimeStep: return index_ % long(c_.writeInterval) == 0;
    case FvWriteControl::RunTime:
    case FvWriteControl::AdjustableRunTime: {
        const long writeIndex = long(((t_ - 0.0) + 0.5 * dt_) / c_.writeInterval);
        if (writeIndex > writeIndex_) {
            writeIndex_ = writeIndex;
            return true;
        }
        return false;
    }
    }
    return false;
}

std::string FoamClock::timeName() const { return std::format("{:.{}g}", t_, c_.timePrecision); }

} // namespace dns
