// The live bridge: solver -> viewer, in shared memory, with no disk between.
//
// `--emit web` already streams frames, but through a directory the reader has
// to poll and the writer has to fsync-ish its way through; for a viewer running
// on the same machine that is a round trip through the page cache for data that
// never needed a filename. `--emit shm` publishes the same planes (and, in 3D,
// the volume) into a POSIX shared-memory segment the viewer maps read-only.
// Nothing is written to disk, so a run can be watched live with `--emit none`
// alongside it and leave no artefacts at all.
//
// CONCURRENCY is a seqlock, not a mutex: the writer bumps `seq` to odd, copies,
// then bumps it to even. A reader snapshots seq, copies, and re-reads seq; if it
// changed or was odd, the frame was torn and it retries. That means the solver
// NEVER blocks on the viewer -- a stalled or killed viewer cannot slow a run
// down, and a viewer that attaches mid-run simply sees the next complete frame.
// It also means the reader must copy out before using the data, which it does.
//
// The segment is fixed-size and allocated once from the grid, so no frame ever
// reallocates or moves an offset.
#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace dns {

inline constexpr char kShmMagic[8] = {'D', 'N', 'S', 'S', 'H', 'M', '0', '1'};
inline constexpr std::uint32_t kShmVersion = 1;

// All offsets are byte offsets from the start of the segment. All field data is
// float32, row-major, i fastest, row j = 0 at the BOTTOM -- the same convention
// as the web format and the PNG export, so a plane means the same thing
// whichever way it left the solver.
struct ShmField
{
    char name[24]{};      // "vort", "speed", ...
    float lo = 0, hi = 1; // fixed display range; a per-frame range flickers
    std::int32_t diverging = 0;
    std::int32_t pad = 0;
};

struct ShmPlane
{
    std::int32_t axis = 2;      // 0 = x (yz plane), 1 = y (xz), 2 = z (xy)
    float off = 0;              // offset from the body centre, in D
    std::int32_t w = 0, h = 0;
    std::uint64_t dataOffset = 0;   // fieldCount consecutive w*h float planes
    std::uint64_t maskOffset = 0;   // one w*h float plane, 1 = fluid; 0 = absent
};

struct ShmVolume
{
    std::int32_t w = 0, h = 0, d = 0, pad = 0;
    // fieldCount consecutive w*h*d float volumes, in the same order as the
    // field table -- so a viewer can slice or fog ANY published field, not just
    // the first one, and can move a cut plane without asking the solver to
    // publish a different one.
    std::uint64_t dataOffset = 0;
};

// Volumes are the expensive part of the segment: one field of a 512^3 run is
// 512 MB. Above this the run publishes planes only, and a viewer degrades to
// the centre planes rather than the machine degrading to swap.
inline constexpr std::uint64_t kShmVolumeBudget = 512ull << 20;

struct ShmHeader
{
    char magic[8]{};
    std::uint32_t version = 0;
    std::uint32_t headerBytes = 0;
    // Seqlock. Odd means a write is in progress; even and unchanged across a
    // read means the copy is coherent.
    std::uint64_t seq = 0;
    std::uint64_t frame = 0;
    double t = 0;

    std::int32_t nx = 0, ny = 0, nz = 0, pad0 = 0;   // full grid, ghosts included
    // The WRITTEN WINDOW in full-grid indices: planes and the volume cover
    // [i0, i0+w) x [j0, j0+h) x [k0, k0+d). An incompressible run passes the
    // interior, so the ghost ring never reaches a frame -- and a viewer that
    // assumed the window started at 0 would draw every plane one cell off.
    std::int32_t i0 = 0, j0 = 0, k0 = 0, pad1 = 0;
    // ...and its SIZE. A viewer needs the depth of the window to place an xy
    // plane at mid-depth; deriving it from the volume fails when the volume is
    // not being copied, which put the plane on the floor of the box.
    std::int32_t winW = 0, winH = 0, winD = 0, pad2 = 0;
    double hx = 0, hy = 0, hz = 0;
    double cx = 0, cy = 0, cz = 0, D = 0;            // body placement

    std::int32_t fieldCount = 0;
    std::int32_t planeCount = 0;
    std::int32_t volumeCount = 0;
    std::int32_t running = 1;      // cleared on a clean shutdown
    // The publishing process. A viewer that did not spawn the run still needs
    // to be able to STOP it -- a run is a GPU and an hour of someone's time,
    // and "detach and go find the terminal" is not an answer.
    std::int32_t pid = 0;
    std::int32_t pad3 = 0;
    std::uint64_t totalBytes = 0;
    // Followed by: ShmField[fieldCount], ShmPlane[planeCount],
    // ShmVolume[volumeCount], then the float data at the recorded offsets.
};

// The writer side. Owned by RunOutput; one per run.
class ShmFrames
{
public:
    struct PlaneSpec
    {
        int axis = 2;
        float off = 0;
        int w = 0, h = 0;
    };

    ~ShmFrames();
    ShmFrames() = default;
    ShmFrames(const ShmFrames&) = delete;
    ShmFrames& operator=(const ShmFrames&) = delete;

    // `name` is the POSIX object name without the leading slash. Sized once
    // from the specs; every later frame writes into the same offsets.
    std::expected<void, std::string> create(const std::string& name,
                                            const std::vector<ShmField>& fields,
                                            const std::vector<PlaneSpec>& planes,
                                            int volW, int volH, int volD);

    void setGrid(int nx, int ny, int nz, int i0, int j0, int k0, int wW, int wH,
                 int wD, double hx, double hy, double hz, double cx, double cy,
                 double cz, double D);

    bool valid() const { return base_ != nullptr; }
    int planeCount() const { return int(planes_.size()); }
    int fieldCount() const { return int(fields_.size()); }

    // One frame. `planeData[p][f]` is the w*h plane for plane p, field f;
    // `planeMask[p]` may be null. `volumes[f]` is the w*h*d volume of field f,
    // empty or short when the run publishes none. Bumps the seqlock.
    void publish(double t, const std::vector<std::vector<const float*>>& planeData,
                 const std::vector<const float*>& planeMask,
                 const std::vector<const float*>& volumes);

    // True when the volumes fit the budget and are being published.
    bool hasVolume() const { return volume_.w > 0; }

    // Marks the run finished so a viewer can say so instead of waiting.
    void finish();

    // The name the viewer should open, without the slash.
    static std::string nameFor(const std::string& outPrefix);

private:
    void unmapAll();

    std::string name_;
    unsigned char* base_ = nullptr;
    std::size_t bytes_ = 0;
    int fd_ = -1;
    std::vector<ShmField> fields_;
    std::vector<ShmPlane> planes_;
    ShmVolume volume_{};
};

} // namespace dns
