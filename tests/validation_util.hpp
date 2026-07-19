// Shared helpers for the physical-validation tests (test_val_*): tolerant CSV
// parsing, windowed statistics, Strouhal extraction, curve comparison, app
// driving, and the deferred-anchor registry (data/validation/anchors.csv).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <sys/wait.h>

namespace valtest {

inline int gFailures = 0;

// Assert a named condition; failures are counted, not fatal.
inline void check(bool ok, const std::string& what)
{
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++gFailures;
}

// Exit code for the whole test; call last in main.
inline int finish()
{
    if (gFailures) std::printf("%d check(s) FAILED\n", gFailures);
    return gFailures ? EXIT_FAILURE : EXIT_SUCCESS;
}

// ctest SKIP_RETURN_CODE convention (no device, missing anchor, ...).
[[noreturn]] inline void skip(const std::string& why)
{
    std::printf("SKIP: %s\n", why.c_str());
    std::exit(77);
}

// Columnar numeric table from comma- or whitespace-separated text.
// '#' lines are ignored; a leading line with alphabetic tokens names columns.
struct CsvTable {
    std::vector<std::string> names;
    std::vector<std::vector<double>> cols;

    // Column by name; empty vector reference never returned — dies if absent.
    const std::vector<double>& col(const std::string& name) const
    {
        auto it = std::find(names.begin(), names.end(), name);
        if (it == names.end()) {
            std::printf("CSV column '%s' missing\n", name.c_str());
            std::exit(EXIT_FAILURE);
        }
        return cols[std::size_t(it - names.begin())];
    }
    std::size_t rows() const { return cols.empty() ? 0 : cols[0].size(); }
};

inline std::vector<std::string> splitFields(const std::string& line)
{
    std::vector<std::string> out;
    std::string cur;
    for (char c : line) {
        if (c == ',' || c == ' ' || c == '\t' || c == '\r') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// Load a stats/reference table; returns nullopt if unreadable or empty.
// Ragged rows and non-numeric data rows are skipped.
inline std::optional<CsvTable> loadCsv(const std::string& path)
{
    std::ifstream f(path);
    if (!f) return std::nullopt;
    CsvTable tab;
    std::string line;
    bool sawHeader = false;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto fields = splitFields(line);
        if (fields.empty()) continue;
        const bool alpha = std::any_of(fields[0].begin(), fields[0].end(),
                                       [](char c) { return std::isalpha(unsigned(c)); });
        if (!sawHeader && alpha) {
            tab.names = fields;
            tab.cols.assign(fields.size(), {});
            sawHeader = true;
            continue;
        }
        if (tab.cols.empty()) {  // headerless: synthesize c0..cN
            for (std::size_t i = 0; i < fields.size(); ++i) tab.names.push_back("c" + std::to_string(i));
            tab.cols.assign(fields.size(), {});
        }
        if (fields.size() != tab.cols.size()) continue;
        std::vector<double> row(fields.size());
        bool ok = true;
        for (std::size_t i = 0; i < fields.size(); ++i) {
            char* end = nullptr;
            row[i] = std::strtod(fields[i].c_str(), &end);
            if (end == fields[i].c_str()) { ok = false; break; }
        }
        if (!ok) continue;
        for (std::size_t i = 0; i < fields.size(); ++i) tab.cols[i].push_back(row[i]);
    }
    if (tab.rows() == 0) return std::nullopt;
    return tab;
}

// Time-windowed trapezoidal mean of y(t) over [t0,t1]; t strictly increasing.
inline double windowMean(const std::vector<double>& t, const std::vector<double>& y,
                         double t0, double t1)
{
    double num = 0.0, den = 0.0;
    for (std::size_t i = 1; i < t.size(); ++i) {
        if (t[i - 1] < t0 || t[i] > t1) continue;
        const double dt = t[i] - t[i - 1];
        num += 0.5 * (y[i] + y[i - 1]) * dt;
        den += dt;
    }
    return den > 0.0 ? num / den : std::nan("");
}

// Windowed rms of y about its windowed mean.
inline double windowRms(const std::vector<double>& t, const std::vector<double>& y,
                        double t0, double t1)
{
    const double mean = windowMean(t, y, t0, t1);
    double num = 0.0, den = 0.0;
    for (std::size_t i = 1; i < t.size(); ++i) {
        if (t[i - 1] < t0 || t[i] > t1) continue;
        const double dt = t[i] - t[i - 1];
        const double d = 0.5 * (y[i] + y[i - 1]) - mean;
        num += d * d * dt;
        den += dt;
    }
    return den > 0.0 ? std::sqrt(num / den) : std::nan("");
}

// Shedding frequency of an oscillating signal by hysteresis zero-crossing.
// Chosen over FFT: samples are non-uniform in t (variable dt), no FFTW dep.
struct StrouhalResult {
    double freq = std::nan("");   // 1 / mean period
    int periods = 0;              // full periods captured in the window
    double periodScatter = 0.0;   // max relative deviation of any period
};

// Requires >= 3 upward crossings in [t0,t1]; deadband 0.2*rms rejects noise.
inline StrouhalResult strouhal(const std::vector<double>& t, const std::vector<double>& y,
                               double t0, double t1)
{
    const double mean = windowMean(t, y, t0, t1);
    const double band = 0.2 * windowRms(t, y, t0, t1);
    std::vector<double> up;                      // interpolated upward crossing times
    bool armed = false;                          // below -band since last crossing
    for (std::size_t i = 1; i < t.size(); ++i) {
        if (t[i] < t0 || t[i] > t1) continue;
        const double a = y[i - 1] - mean, b = y[i] - mean;
        if (a < -band) armed = true;
        if (armed && a <= band && b > band) {
            up.push_back(t[i - 1] + (t[i] - t[i - 1]) * (band - a) / (b - a));
            armed = false;
        }
    }
    StrouhalResult r;
    if (up.size() < 3) return r;
    r.periods = int(up.size()) - 1;
    const double meanT = (up.back() - up.front()) / r.periods;
    for (std::size_t i = 1; i < up.size(); ++i)
        r.periodScatter = std::max(r.periodScatter, std::abs((up[i] - up[i - 1]) - meanT) / meanT);
    r.freq = 1.0 / meanT;
    return r;
}

// Max |y - yRef| over refT in [t0,t1], scaled by max|yRef| in the window
// (scale-relative: pointwise-relative blows up at zero crossings).
inline double curveMaxRelDiff(const std::vector<double>& refT, const std::vector<double>& refY,
                              const std::vector<double>& t, const std::vector<double>& y,
                              double t0, double t1)
{
    double scale = 0.0;
    for (std::size_t i = 0; i < refT.size(); ++i)
        if (refT[i] >= t0 && refT[i] <= t1) scale = std::max(scale, std::abs(refY[i]));
    if (scale == 0.0) return std::nan("");
    double worst = 0.0;
    std::size_t j = 1;
    for (std::size_t i = 0; i < refT.size(); ++i) {
        const double tr = refT[i];
        if (tr < t0 || tr > t1 || tr < t.front() || tr > t.back()) continue;
        while (j < t.size() - 1 && t[j] < tr) ++j;
        const double w = (tr - t[j - 1]) / (t[j] - t[j - 1]);
        const double yi = y[j - 1] + w * (y[j] - y[j - 1]);
        worst = std::max(worst, std::abs(yi - refY[i]) / scale);
    }
    return worst;
}

// Location/value of the curve maximum, parabola-refined through 3 samples.
struct PeakResult {
    double t = std::nan("");
    double value = std::nan("");
};

inline PeakResult curvePeak(const std::vector<double>& t, const std::vector<double>& y)
{
    if (t.size() < 3) return {};
    std::size_t k = std::size_t(std::max_element(y.begin(), y.end()) - y.begin());
    if (k == 0 || k + 1 == y.size()) return {t[k], y[k]};
    const double d1 = y[k] - y[k - 1], d2 = y[k] - y[k + 1];
    const double denom = d1 + d2;
    PeakResult p{t[k], y[k]};
    if (denom > 0.0) {
        const double s = 0.5 * (d1 - d2) / denom;                 // in [-0.5,0.5]
        p.t = t[k] + s * 0.5 * (t[k + 1] - t[k - 1]);
        p.value = y[k] + 0.25 * s * (d2 - d1) + 0.5 * s * s * denom;
    }
    return p;
}

// Run a shell command, silenced; returns WEXITSTATUS or -1 on abnormal exit.
inline int runBinary(const std::string& cmd)
{
    const int rc = std::system((cmd + " > /dev/null 2>&1").c_str());
    if (rc == -1 || !WIFEXITED(rc)) return -1;
    return WEXITSTATUS(rc);
}

// As runBinary, but capture stdout to outPath (stderr silenced).
inline int runBinaryTo(const std::string& cmd, const std::string& outPath)
{
    const int rc = std::system((cmd + " > \"" + outPath + "\" 2> /dev/null").c_str());
    if (rc == -1 || !WIFEXITED(rc)) return -1;
    return WEXITSTATUS(rc);
}

// One row of data/validation/anchors.csv (regenerated per docs/validation.md).
struct Anchor {
    double value = std::nan("");
    double tolRel = std::nan("");
    std::string config;
};

// name,value,tol_rel,config,date,note — '#' rows are pending/TODO anchors.
inline std::map<std::string, Anchor> loadAnchors(const std::string& path)
{
    std::map<std::string, Anchor> out;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) fields.push_back(field);
        if (fields.size() < 3 || fields[0] == "name") continue;
        Anchor a;
        a.value = std::atof(fields[1].c_str());
        a.tolRel = std::atof(fields[2].c_str());
        if (fields.size() > 3) a.config = fields[3];
        out[fields[0]] = a;
    }
    return out;
}

// Anchor lookup for tests: absent anchor => deferred, not a failure.
inline std::optional<Anchor> findAnchor(const std::map<std::string, Anchor>& anchors,
                                        const std::string& name)
{
    auto it = anchors.find(name);
    if (it == anchors.end()) {
        std::printf("  [DEFER] anchor '%s' missing -- regenerate per docs/validation.md\n",
                    name.c_str());
        return std::nullopt;
    }
    return it->second;
}

}  // namespace valtest
