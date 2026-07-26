#include "dns/shmframes.hpp"

#include <atomic>
#include <cerrno>
#include <cctype>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dns {
namespace {

// Every block starts on a 64-byte boundary: the plane data is memcpy'd every
// frame and an aligned destination is what lets that be a wide copy.
std::uint64_t align64(std::uint64_t v) { return (v + 63u) & ~std::uint64_t(63); }

// The seqlock counter, accessed atomically. The rest of the segment is plain
// memory: the release/acquire fences around the counter are what order it.
std::atomic_ref<std::uint64_t> seqOf(ShmHeader* h)
{
    return std::atomic_ref<std::uint64_t>(h->seq);
}

} // namespace

std::string ShmFrames::nameFor(const std::string& outPrefix)
{
    // Basename only: `--out runs/foo` must not try to create a nested shm name.
    std::string base = outPrefix;
    if (const auto slash = base.find_last_of('/'); slash != std::string::npos)
        base = base.substr(slash + 1);
    if (base.empty()) base = "run";
    std::string s = "dns_";
    for (char c : base)
        s += (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')
                 ? c
                 : '_';
    return s;
}

ShmFrames::~ShmFrames()
{
    finish();
    unmapAll();
}

void ShmFrames::unmapAll()
{
    if (base_) munmap(base_, bytes_);
    base_ = nullptr;
    if (fd_ >= 0) close(fd_);
    fd_ = -1;
    // The segment is unlinked here, not at create: a viewer that is still
    // mapped keeps its mapping alive (POSIX unlink semantics), it just cannot
    // be opened afresh. That is the behaviour we want when a run ends.
    if (!name_.empty()) shm_unlink(("/" + name_).c_str());
    name_.clear();
}

std::expected<void, std::string>
ShmFrames::create(const std::string& name, const std::vector<ShmField>& fields,
                  const std::vector<PlaneSpec>& planes, int volW, int volH,
                  int volD)
{
    unmapAll();
    if (fields.empty()) return std::unexpected("shm: no fields declared");

    fields_ = fields;
    planes_.clear();
    volume_ = {};

    // Lay the segment out once. Offsets never move again, so a frame is pure
    // memcpy and a reader can cache them.
    std::uint64_t off = align64(sizeof(ShmHeader) + sizeof(ShmField) * fields.size()
                                + sizeof(ShmPlane) * planes.size()
                                + sizeof(ShmVolume));
    for (const auto& p : planes) {
        ShmPlane sp;
        sp.axis = p.axis;
        sp.off = p.off;
        sp.w = p.w;
        sp.h = p.h;
        const std::uint64_t plane = std::uint64_t(p.w) * std::uint64_t(p.h)
                                    * sizeof(float);
        sp.dataOffset = off;
        off = align64(off + plane * fields.size());
        sp.maskOffset = off;
        off = align64(off + plane);
        planes_.push_back(sp);
    }
    if (volW > 0 && volH > 0 && volD > 0) {
        const std::uint64_t one =
            std::uint64_t(volW) * volH * volD * sizeof(float);
        // All fields, or none: a viewer that could fog one field and not
        // another would be a worse surprise than no fog at all.
        if (one * fields.size() <= kShmVolumeBudget) {
            volume_.w = volW;
            volume_.h = volH;
            volume_.d = volD;
            volume_.dataOffset = off;
            off = align64(off + one * fields.size());
        }
    }
    bytes_ = std::size_t(off);

    const std::string path = "/" + name;
    shm_unlink(path.c_str());   // a stale segment from a crashed run
    fd_ = shm_open(path.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd_ < 0)
        return std::unexpected(std::string("shm_open ") + path + ": "
                               + std::strerror(errno));
    if (ftruncate(fd_, off_t(bytes_)) != 0) {
        const std::string e = std::strerror(errno);
        close(fd_);
        fd_ = -1;
        shm_unlink(path.c_str());
        return std::unexpected("shm ftruncate: " + e);
    }
    void* m = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (m == MAP_FAILED) {
        const std::string e = std::strerror(errno);
        close(fd_);
        fd_ = -1;
        shm_unlink(path.c_str());
        return std::unexpected("shm mmap: " + e);
    }
    base_ = static_cast<unsigned char*>(m);
    name_ = name;
    std::memset(base_, 0, bytes_);

    auto* h = reinterpret_cast<ShmHeader*>(base_);
    std::memcpy(h->magic, kShmMagic, sizeof kShmMagic);
    h->version = kShmVersion;
    h->headerBytes = std::uint32_t(sizeof(ShmHeader));
    h->fieldCount = std::int32_t(fields_.size());
    h->planeCount = std::int32_t(planes_.size());
    h->volumeCount = volume_.w > 0 ? 1 : 0;
    h->running = 1;
    h->pid = std::int32_t(getpid());
    h->totalBytes = bytes_;

    auto* fArr = reinterpret_cast<ShmField*>(base_ + sizeof(ShmHeader));
    for (std::size_t i = 0; i < fields_.size(); ++i) fArr[i] = fields_[i];
    auto* pArr = reinterpret_cast<ShmPlane*>(fArr + fields_.size());
    for (std::size_t i = 0; i < planes_.size(); ++i) pArr[i] = planes_[i];
    auto* vArr = reinterpret_cast<ShmVolume*>(pArr + planes_.size());
    *vArr = volume_;
    return {};
}

void ShmFrames::setGrid(int nx, int ny, int nz, int i0, int j0, int k0, int wW,
                        int wH, int wD, double hx, double hy, double hz, double cx,
                        double cy, double cz, double D)
{
    if (!base_) return;
    auto* h = reinterpret_cast<ShmHeader*>(base_);
    h->nx = nx;
    h->ny = ny;
    h->nz = nz;
    h->i0 = i0;
    h->j0 = j0;
    h->k0 = k0;
    h->winW = wW;
    h->winH = wH;
    h->winD = wD;
    h->hx = hx;
    h->hy = hy;
    h->hz = hz;
    h->cx = cx;
    h->cy = cy;
    h->cz = cz;
    h->D = D;
}

void ShmFrames::publish(double t,
                        const std::vector<std::vector<const float*>>& planeData,
                        const std::vector<const float*>& planeMask,
                        const std::vector<const float*>& volumes)
{
    if (!base_) return;
    auto* h = reinterpret_cast<ShmHeader*>(base_);
    auto seq = seqOf(h);

    // Odd: a write is in progress. Release so the reader cannot observe the
    // odd counter after the payload stores it is meant to guard.
    seq.fetch_add(1, std::memory_order_acq_rel);
    std::atomic_thread_fence(std::memory_order_release);

    h->t = t;
    for (std::size_t p = 0; p < planes_.size() && p < planeData.size(); ++p) {
        const ShmPlane& sp = planes_[p];
        const std::size_t n = std::size_t(sp.w) * std::size_t(sp.h);
        for (std::size_t f = 0; f < fields_.size() && f < planeData[p].size(); ++f)
            if (planeData[p][f])
                std::memcpy(base_ + sp.dataOffset + f * n * sizeof(float),
                            planeData[p][f], n * sizeof(float));
        if (p < planeMask.size() && planeMask[p])
            std::memcpy(base_ + sp.maskOffset, planeMask[p], n * sizeof(float));
    }
    if (volume_.w > 0) {
        const std::size_t n =
            std::size_t(volume_.w) * volume_.h * volume_.d;
        for (std::size_t f = 0; f < fields_.size() && f < volumes.size(); ++f)
            if (volumes[f])
                std::memcpy(base_ + volume_.dataOffset + f * n * sizeof(float),
                            volumes[f], n * sizeof(float));
    }

    std::atomic_thread_fence(std::memory_order_release);
    h->frame += 1;
    seq.fetch_add(1, std::memory_order_acq_rel);   // even again: readable
}

void ShmFrames::finish()
{
    if (!base_) return;
    auto* h = reinterpret_cast<ShmHeader*>(base_);
    h->running = 0;
    std::atomic_thread_fence(std::memory_order_release);
}

} // namespace dns
