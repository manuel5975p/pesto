#include "dns/webframes.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace dns {

namespace {

std::string fmtDouble(double v)
{
    char b[32];
    std::snprintf(b, sizeof b, "%.9g", v);
    return b;
}

} // namespace

WebFrames::WebFrames(std::string dir, int w, int h, std::string simJson)
    : dir_(std::move(dir)), simJson_(std::move(simJson)), w_(w), h_(h)
{
    assert(w_ > 0 && h_ > 0);
    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
}

void WebFrames::addField(const std::string& name, const std::string& kind,
                         double lo, double hi)
{
    assert(hi > lo && frames_.empty());
    fields_.push_back({name, kind, lo, hi});
}

bool WebFrames::writeQuantized(const std::string& path, const float* data,
                               double lo, double hi, float& mn, float& mx) const
{
    const std::size_t n = std::size_t(w_) * h_;
    std::vector<std::uint16_t> q(n);
    const double s = 65535.0 / (hi - lo);
    mn = mx = data[0];
    for (std::size_t c = 0; c < n; ++c) {
        const float v = data[c];
        mn = v < mn ? v : mn;
        mx = v > mx ? v : mx;
        double t = (double(v) - lo) * s;
        t = t < 0.0 ? 0.0 : (t > 65535.0 ? 65535.0 : t);
        q[c] = std::uint16_t(t + 0.5);
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(q.data()),
            std::streamsize(n * sizeof(std::uint16_t)));
    // Close first: a buffered ENOSPC only surfaces at close, so testing the
    // stream while it is still open reports success on a truncated file.
    f.close();
    return bool(f);
}

bool WebFrames::writeStatic(const std::string& name, const float* data,
                            double lo, double hi)
{
    float mn, mx;
    if (!writeQuantized(dir_ + "/" + name + ".bin", data, lo, hi, mn, mx))
        return false;
    statics_.push_back({name, "static", lo, hi});
    return writeManifest();
}

bool WebFrames::writeFrame(double t, const std::vector<const float*>& data)
{
    assert(data.size() == fields_.size());
    Frame fr{t, {}};
    char path[4096];
    for (std::size_t q = 0; q < fields_.size(); ++q) {
        std::snprintf(path, sizeof path, "%s/%s_%05d.bin", dir_.c_str(),
                      fields_[q].name.c_str(), int(frames_.size()));
        float mn, mx;
        if (!writeQuantized(path, data[q], fields_[q].lo, fields_[q].hi, mn, mx))
            return false;
        fr.minmax.emplace_back(mn, mx);
    }
    frames_.push_back(std::move(fr));
    return writeManifest();
}

bool WebFrames::writeManifest() const
{
    std::string j = "{\"version\":1,\"grid\":{\"nx\":" + std::to_string(w_) +
                    ",\"ny\":" + std::to_string(h_) + "},\"sim\":" +
                    (simJson_.empty() ? "{}" : simJson_) + ",\"fields\":[";
    for (std::size_t q = 0; q < fields_.size(); ++q) {
        const auto& f = fields_[q];
        j += std::string(q ? "," : "") + "{\"name\":\"" + f.name +
             "\",\"file\":\"" + f.name + "_%05d.bin\",\"kind\":\"" + f.kind +
             "\",\"range\":[" + fmtDouble(f.lo) + "," + fmtDouble(f.hi) + "]}";
    }
    j += "],\"static\":[";
    for (std::size_t q = 0; q < statics_.size(); ++q) {
        const auto& f = statics_[q];
        j += std::string(q ? "," : "") + "{\"name\":\"" + f.name +
             "\",\"file\":\"" + f.name + ".bin\",\"range\":[" +
             fmtDouble(f.lo) + "," + fmtDouble(f.hi) + "]}";
    }
    j += "],\"frames\":[";
    for (std::size_t r = 0; r < frames_.size(); ++r) {
        j += std::string(r ? "," : "") + "{\"t\":" + fmtDouble(frames_[r].t) +
             ",\"minmax\":{";
        for (std::size_t q = 0; q < fields_.size(); ++q)
            j += std::string(q ? "," : "") + "\"" + fields_[q].name + "\":[" +
                 fmtDouble(frames_[r].minmax[q].first) + "," +
                 fmtDouble(frames_[r].minmax[q].second) + "]";
        j += "}}";
    }
    j += "]}\n";

    const std::string tmp = dir_ + "/manifest.json.tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        if (!f) return false;
        f.write(j.data(), std::streamsize(j.size()));
        if (!f) return false;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, dir_ + "/manifest.json", ec);
    return !ec;
}

} // namespace dns
