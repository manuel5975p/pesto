#include "ui/field.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>

#include <fcntl.h>
#include <csignal>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ui {
namespace {

std::uint64_t loadSeq(const dns::ShmHeader* h)
{
    return std::atomic_ref<std::uint64_t>(const_cast<std::uint64_t&>(h->seq))
        .load(std::memory_order_acquire);
}

float lerp(float a, float b, float t) { return a + (b - a) * t; }

} // namespace

std::size_t LiveSnapshot::bytes() const
{
    std::size_t n = sizeof(LiveSnapshot);
    for (const auto& p : planes)
        n += (p.value.capacity() + p.mask.capacity()) * sizeof(float);
    return n;
}

LiveField::~LiveField() { detach(); }

void LiveField::clearHistory()
{
    hist_.clear();
    histBytes_ = 0;
}

void LiveField::setHistoryBudget(std::size_t bytes)
{
    histBudget_ = bytes;
    while (hist_.size() > 1 && histBytes_ > histBudget_) {
        histBytes_ -= hist_.front().bytes();
        hist_.pop_front();
    }
}

LiveView LiveField::view() const
{
    LiveView v = geomView();
    v.planes = &planes_;
    v.volume = &volume_;
    v.volW = volW_;
    v.volH = volH_;
    v.volD = volD_;
    return v;
}

// A rewound frame carries no volume, so the cut plane falls back to the
// published centre plane and the fog draws nothing -- see LiveSnapshot.
LiveView LiveField::viewOf(const LiveSnapshot& s) const
{
    LiveView v = geomView();
    v.planes = &s.planes;
    return v;
}

LiveView LiveField::geomView() const
{
    LiveView v;
    v.i0 = i0_;
    v.j0 = j0_;
    v.k0 = k0_;
    v.winW = winW_;
    v.winH = winH_;
    v.winD = winD_;
    v.hx = hx_;
    v.hy = hy_;
    v.hz = hz_;
    return v;
}

const dns::ShmHeader* LiveField::header() const
{
    return reinterpret_cast<const dns::ShmHeader*>(base_);
}

std::string LiveField::attach(const std::string& name)
{
    detach();
    const std::string path = "/" + name;
    const int fd = shm_open(path.c_str(), O_RDONLY, 0);
    if (fd < 0) return std::string("no live run at ") + path;

    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size < off_t(sizeof(dns::ShmHeader))) {
        close(fd);
        return "live segment too small";
    }
    void* m = mmap(nullptr, std::size_t(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) {
        close(fd);
        return std::string("mmap: ") + std::strerror(errno);
    }
    base_ = static_cast<const unsigned char*>(m);
    bytes_ = std::size_t(st.st_size);
    fd_ = fd;
    name_ = name;

    const dns::ShmHeader* h = header();
    if (std::memcmp(h->magic, dns::kShmMagic, sizeof dns::kShmMagic) != 0 ||
        h->version != dns::kShmVersion) {
        detach();
        return "live segment is not a dns frame stream (or a different version)";
    }
    fields_.assign(reinterpret_cast<const dns::ShmField*>(base_ + sizeof(dns::ShmHeader)),
                   reinterpret_cast<const dns::ShmField*>(base_ + sizeof(dns::ShmHeader))
                       + h->fieldCount);
    frame_ = 0;
    sel_ = -1;
    return {};
}

void LiveField::detach()
{
    if (base_) munmap(const_cast<unsigned char*>(base_), bytes_);
    base_ = nullptr;
    bytes_ = 0;
    if (fd_ >= 0) close(fd_);
    fd_ = -1;
    name_.clear();
    fields_.clear();
    planes_.clear();
    volume_.clear();
    frame_ = 0;
    sel_ = -1;
    volW_ = volH_ = volD_ = 0;
    clearHistory();
}

bool LiveField::running() const
{
    if (!base_) return false;
    if (header()->running == 0) return false;
    // A run killed with SIGKILL never gets to clear `running` or unlink the
    // segment, so the flag alone would leave a dead run looking live forever.
    // The publisher's pid is the ground truth: kill(pid, 0) probes existence
    // without signalling anything.
    const int pid = header()->pid;
    if (pid > 0 && ::kill(pid_t(pid), 0) != 0 && errno == ESRCH) return false;
    return true;
}

int LiveField::publisherPid() const { return base_ ? header()->pid : 0; }

bool LiveField::isStale(const std::string& name)
{
    LiveField probe;
    if (!probe.attach(name).empty()) return true;   // unreadable: treat as junk
    const int pid = probe.publisherPid();
    if (pid <= 0) return false;
    return ::kill(pid_t(pid), 0) != 0 && errno == ESRCH;
}

bool LiveField::hasVolume() const
{
    return base_ && header()->volumeCount > 0;
}

std::string LiveField::stopPublisher()
{
    const int pid = publisherPid();
    if (pid <= 0) return "the run did not publish a pid";
    if (::kill(pid_t(pid), SIGTERM) != 0)
        return std::string("could not signal pid ") + std::to_string(pid) + ": "
               + std::strerror(errno);
    return {};
}

bool LiveField::poll()
{
    if (!base_) return false;
    const dns::ShmHeader* h = header();
    const int nf = h->fieldCount;
    if (nf <= 0) return false;
    const int sel = std::clamp(want_, 0, nf - 1);
    // Frame 0 means the publisher has not written anything yet: the segment
    // exists (it is created before the first step) but holds only zeros.
    if (h->frame == 0) return false;
    if (h->frame == frame_ && sel == sel_) return false;

    const auto* pArr = reinterpret_cast<const dns::ShmPlane*>(
        base_ + sizeof(dns::ShmHeader) + sizeof(dns::ShmField) * std::size_t(nf));
    const auto* vArr = reinterpret_cast<const dns::ShmVolume*>(pArr + h->planeCount);

    // Seqlock read: copy between two even, equal observations of `seq`. Three
    // attempts is plenty -- the writer holds the lock for one memcpy, so a
    // reader losing three times in a row means the run is publishing far faster
    // than the viewer can draw, and skipping this frame is the right answer.
    for (int attempt = 0; attempt < 3; ++attempt) {
        const std::uint64_t s0 = loadSeq(h);
        if (s0 & 1u) continue;   // a write is in progress

        const double t = h->t;
        const std::uint64_t fr = h->frame;
        std::vector<LivePlane> planes;
        planes.reserve(std::size_t(h->planeCount));
        for (int p = 0; p < h->planeCount; ++p) {
            const dns::ShmPlane& sp = pArr[p];
            const std::size_t n = std::size_t(sp.w) * std::size_t(sp.h);
            if (sp.dataOffset + (std::size_t(sel) + 1) * n * sizeof(float) > bytes_)
                return false;   // malformed: refuse rather than read out of range
            LivePlane lp;
            lp.axis = sp.axis;
            lp.w = sp.w;
            lp.h = sp.h;
            lp.value.resize(n);
            std::memcpy(lp.value.data(),
                        base_ + sp.dataOffset + std::size_t(sel) * n * sizeof(float),
                        n * sizeof(float));
            if (sp.maskOffset && sp.maskOffset + n * sizeof(float) <= bytes_) {
                lp.mask.resize(n);
                std::memcpy(lp.mask.data(), base_ + sp.maskOffset, n * sizeof(float));
            }
            planes.push_back(std::move(lp));
        }

        std::vector<float> vol;
        int vw = 0, vh = 0, vd = 0;
        if (wantVolume_ && h->volumeCount > 0) {
            vw = vArr->w;
            vh = vArr->h;
            vd = vArr->d;
            const std::size_t n = std::size_t(vw) * vh * vd;
            // fieldCount consecutive volumes; take the selected one.
            const std::uint64_t off =
                vArr->dataOffset + std::uint64_t(sel) * n * sizeof(float);
            if (off + n * sizeof(float) <= bytes_) {
                vol.resize(n);
                std::memcpy(vol.data(), base_ + off, n * sizeof(float));
            } else {
                vw = vh = vd = 0;
            }
        }

        std::atomic_thread_fence(std::memory_order_acquire);
        if (loadSeq(h) != s0) continue;   // torn: the writer moved under us

        // Switching the coloured field re-decodes the SAME frames as a
        // different quantity, so the frames already in the timeline are no
        // longer the same series. Start it over rather than splice two.
        if (sel != sel_ || fr < frame_) clearHistory();

        planes_ = std::move(planes);
        volume_ = std::move(vol);
        volW_ = vw;
        volH_ = vh;
        volD_ = vd;
        frame_ = fr;
        t_ = t;
        sel_ = sel;
        i0_ = h->i0;
        j0_ = h->j0;
        k0_ = h->k0;
        winW_ = h->winW;
        winH_ = h->winH;
        winD_ = h->winD;
        hx_ = h->hx;
        hy_ = h->hy;
        hz_ = h->hz;

        // Keep the planes for rewinding. The volume is NOT kept: see
        // LiveSnapshot. Evicting from the front leaves the newest frames, which
        // is the end of the timeline anyone scrubs from.
        LiveSnapshot snap;
        snap.frame = frame_;
        snap.t = t_;
        snap.planes = planes_;
        histBytes_ += snap.bytes();
        hist_.push_back(std::move(snap));
        while (hist_.size() > 1 && histBytes_ > histBudget_) {
            histBytes_ -= hist_.front().bytes();
            hist_.pop_front();
        }
        return true;
    }
    return false;
}

// Ramps authored for a dark viewport: the "nothing happening" end of each sinks
// into the background instead of glowing, so a wake reads as structure rather
// than as a bright rectangle.
void fieldColor(float t, bool diverging, unsigned char& r, unsigned char& g,
                unsigned char& b)
{
    t = std::clamp(t, 0.0f, 1.0f);
    float fr, fg, fb;
    if (diverging) {
        // blue <- dark -> red, with the zero crossing dark.
        if (t < 0.5f) {
            const float u = t * 2.0f;               // 0 = strong blue, 1 = zero
            fr = lerp(0.25f, 0.09f, u);
            fg = lerp(0.55f, 0.10f, u);
            fb = lerp(1.00f, 0.13f, u);
        } else {
            const float u = (t - 0.5f) * 2.0f;      // 0 = zero, 1 = strong red
            fr = lerp(0.09f, 1.00f, u);
            fg = lerp(0.10f, 0.35f, u);
            fb = lerp(0.13f, 0.20f, u);
        }
    } else {
        // dark -> teal -> amber -> white: a magnitude ramp with usable contrast
        // at both the low and high end.
        if (t < 0.33f) {
            const float u = t / 0.33f;
            fr = lerp(0.05f, 0.10f, u);
            fg = lerp(0.07f, 0.55f, u);
            fb = lerp(0.10f, 0.60f, u);
        } else if (t < 0.70f) {
            const float u = (t - 0.33f) / 0.37f;
            fr = lerp(0.10f, 0.95f, u);
            fg = lerp(0.55f, 0.72f, u);
            fb = lerp(0.60f, 0.25f, u);
        } else {
            const float u = (t - 0.70f) / 0.30f;
            fr = lerp(0.95f, 1.00f, u);
            fg = lerp(0.72f, 0.98f, u);
            fb = lerp(0.25f, 0.95f, u);
        }
    }
    r = static_cast<unsigned char>(std::clamp(fr, 0.0f, 1.0f) * 255.0f);
    g = static_cast<unsigned char>(std::clamp(fg, 0.0f, 1.0f) * 255.0f);
    b = static_cast<unsigned char>(std::clamp(fb, 0.0f, 1.0f) * 255.0f);
}

} // namespace ui
