// Host side of the pimpleFoam path (docs/fvpimple.md): OpenFOAM v2606's Time
// as pimpleFoam.C drives it -- setInitialDeltaT.H, setDeltaT.H, Time::run(),
// Time::operator++ (write times), Time::adjustDeltaT (adjustableRunTime) --
// operand for operand, FP64, so the deltaT sequence of a run is OpenFOAM's
// given the same Courant numbers. Pure, no GPU, no filesystem.
#pragma once

#include "dns/fvcase.hpp"

#include <string>

namespace dns {

class FoamClock
{
public:
    // Pre: c from a pimpleFoam FoamCase (endTime > 0, deltaT > 0), start time 0.
    explicit FoamClock(const FvTimeCtl& c);

    // setInitialDeltaT.H: only with adjustTimeStep, at time index 0, CoNum > SMALL:
    // deltaT = min(maxCo deltaT / CoNum, min(deltaT, maxDeltaT)), then adjustDeltaT.
    void setInitialDeltaT(double coNum);
    // setDeltaT.H (adjustTimeStep): the 1.2 / (1 + 0.1 f) limited factor, then adjustDeltaT.
    void setDeltaT(double coNum);
    // Time::run(): value < endTime - 0.5 deltaT.
    bool running() const { return t_ < (c_.endTime - 0.5 * dt_); }
    // ++runTime: the time and index advance by the current deltaT; returns
    // Time::writeTime() of the new time.
    bool advance();

    double value() const { return t_; }
    double deltaT() const { return dt_; }
    long index() const { return index_; }
    // Time::timeName(): general format at timePrecision significant digits.
    // (The precision growth of Time::operator++ for clashing names is not built.)
    std::string timeName() const;

private:
    void adjustDeltaT();

    FvTimeCtl c_;
    double t_ = 0, dt_ = 0;
    long index_ = 0, writeIndex_ = 0;
};

} // namespace dns
