// internal::StatsCadence (gpu/internal/run_loop.hpp), driven exactly the way
// the five body/wall loops drive it: printDue is consulted INSIDE the row
// branch, printed() before logged(). The progress line must fire every
// 10 * statsEvery steps -- not only at tEnd, which is what a shared counter
// produced (lastLog resets every statsEvery steps, so `steps - lastLog >=
// 10 * every` was unreachable and dns-ui's live plots stayed empty).
// Pure host arithmetic; needs the HIP headers, never a device.
#include "dns/gpu/hip_runtime.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "check.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

using dnstest::check;

struct Trace
{
    std::vector<long> rows, prints;
};

// The runner loop shape (runner_comp.hip): step, then the row branch.
Trace drive(long every, double dt, double tEnd)
{
    dns::gpu::internal::StatsCadence cad{every};
    Trace tr;
    double t = 0.0;
    long steps = 0;
    while (t < tEnd)
    {
        t += dt;
        ++steps;
        if (cad.due(steps, t, tEnd))
        {
            tr.rows.push_back(steps);
            if (cad.printDue(steps, t, tEnd))
            {
                tr.prints.push_back(steps);
                cad.printed(steps);
            }
            cad.logged(steps);
        }
    }
    return tr;
}

void cadence(long every, long nSteps)
{
    // tEnd lands exactly on step nSteps; dt a power of two so t is exact.
    const double dt = 0.125, tEnd = dt * double(nSteps);
    const Trace tr = drive(every, dt, tEnd);
    const std::string tag = "every " + std::to_string(every) + ", "
                            + std::to_string(nSteps) + " steps";

    std::vector<long> rowsExp, printsExp;
    for (long s = every; s <= nSteps; s += every) rowsExp.push_back(s);
    if (rowsExp.empty() || rowsExp.back() != nSteps) rowsExp.push_back(nSteps);
    for (long s = 10 * every; s <= nSteps; s += 10 * every)
        printsExp.push_back(s);
    if (printsExp.empty() || printsExp.back() != nSteps)
        printsExp.push_back(nSteps);

    std::printf("%-26s rows %zu (expect %zu), prints %zu (expect %zu), first "
                "print at %ld\n",
                tag.c_str(), tr.rows.size(), rowsExp.size(), tr.prints.size(),
                printsExp.size(), tr.prints.empty() ? -1L : tr.prints.front());
    check(tr.rows == rowsExp, tag + ": a row every statsEvery steps + tEnd");
    check(tr.prints == printsExp,
          tag + ": the progress line every 10*statsEvery steps + tEnd");
}

} // namespace

int main()
{
    cadence(1, 35);
    cadence(5, 200);   // 4 progress lines before tEnd, then tEnd
    cadence(7, 150);   // tEnd off the cadence grid
    cadence(200, 5000);
    cadence(50, 30);   // shorter than one row: tEnd alone
    // The bug's signature, spelled out: a long run must print BEFORE tEnd.
    {
        const Trace tr = drive(10, 0.125, 0.125 * 1000);
        check(tr.prints.size() > 1 && tr.prints.front() == 100,
              "1000 steps at statsEvery 10: first progress line at step 100");
    }
    return dnstest::finish();
}
