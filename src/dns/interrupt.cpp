#include "dns/interrupt.hpp"

namespace dns {
namespace {

extern "C" void onStopSignal(int sig)
{
    // The ONLY thing done here. Everything else -- flushing the stats file,
    // unlinking the shared-memory segment, printing the epilogue -- happens on
    // the normal return path, where it is allowed to.
    g_stopRequested = sig;
}

} // namespace

void installInterruptHandler()
{
    struct sigaction sa {};
    sa.sa_handler = onStopSignal;
    sigemptyset(&sa.sa_mask);
    // No SA_RESTART: a blocking read in the middle of a step should come back
    // with EINTR rather than swallow the signal.
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
}

} // namespace dns
