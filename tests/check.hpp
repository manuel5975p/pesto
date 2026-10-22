// The three functions every test in this tree needs: a counting assertion, a
// main() exit code derived from that count, and the ctest skip convention.
//
// validation_util.hpp re-exports these under `valtest`, so val_* call sites
// are unchanged; they live here rather than there so a test can count failures
// without also dragging in CSV parsing, the anchor registry and <sys/wait.h>.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

namespace dnstest {

inline int gFailures = 0;

// Assert a named condition. Failures are COUNTED, not fatal, so one run
// reports every broken property rather than the first.
inline void check(bool ok, const std::string& what)
{
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++gFailures;
}

// Exit code for the whole test; call last in main. The ONLY correct epilogue
// for a counting test -- do not hand-roll a `return something ? 0 : 1`.
inline int finish()
{
    if (gFailures) std::printf("%d check(s) FAILED\n", gFailures);
    return gFailures ? EXIT_FAILURE : EXIT_SUCCESS;
}

// ctest SKIP_RETURN_CODE convention (no device, missing anchor, ...): the ONE
// place the code is spelled.
inline constexpr int kSkipExitCode = 77;

[[noreturn]] inline void skip(const std::string& why)
{
    std::printf("SKIP: %s\n", why.c_str());
    std::exit(kSkipExitCode);
}

} // namespace dnstest
