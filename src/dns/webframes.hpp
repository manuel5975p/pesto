// Live browser-viewer frame writer (docs/viewer-format.md v1): a per-run
// directory of u16-quantized field frames plus an atomically-rewritten
// manifest.json a web viewer polls. CPU-only.
#pragma once

#include <string>
#include <vector>

namespace dns {

class WebFrames
{
public:
    // Creates dir (parents included); w x h image dims shared by all fields;
    // simJson = app metadata inserted verbatim as the "sim" object ("{...}").
    WebFrames(std::string dir, int w, int h, std::string simJson);

    // Declare a field before the first writeFrame. kind "diverging" or
    // "sequential" (viewer rendering hint); [lo, hi] = fixed quantization
    // range (hi > lo).
    void addField(const std::string& name, const std::string& kind, double lo,
                  double hi);

    // Time-independent field (e.g. body mask), written once, same encoding.
    bool writeStatic(const std::string& name, const float* data, double lo,
                     double hi);

    // Append one frame: data[i] belongs to the i-th declared field, w*h
    // floats, row j = 0 at the bottom. Rewrites the manifest (commit point).
    bool writeFrame(double t, const std::vector<const float*>& data);

    int frames() const { return int(frames_.size()); }

private:
    struct Field
    {
        std::string name, kind;
        double lo, hi;
    };
    struct Frame
    {
        double t;
        std::vector<std::pair<float, float>> minmax; // per field
    };

    bool writeQuantized(const std::string& path, const float* data, double lo,
                        double hi, float& mn, float& mx) const;
    bool writeManifest() const;

    std::string dir_, simJson_;
    int w_, h_;
    std::vector<Field> fields_;
    std::vector<Field> statics_;
    std::vector<Frame> frames_;
};

} // namespace dns
