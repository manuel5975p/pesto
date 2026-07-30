// Reading a running solver's live fields, straight out of shared memory.
//
// The solver publishes with `--emit shm` (src/dns/shmframes.hpp): a POSIX
// segment holding the cut planes, the 3D volume and a body mask, guarded by a
// seqlock. There is no file, no directory to poll and no page cache in the
// path -- the viewer maps the same pages the solver writes.
//
// This is also why dns-ui can show live results while still linking dns_core
// only and solving nothing itself: it is a READER of another process's memory,
// not a solver.
//
// A read is: snapshot seq, copy out, re-read seq. If the counter moved or was
// odd, the frame was torn mid-write and the copy is discarded -- so the solver
// never waits for the viewer, and the viewer never shows half a frame.
#pragma once

#include "dns/shmframes.hpp"

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace ui {

// One decoded plane, in the viewer's OWN storage. Never a pointer into the
// shared segment: that memory changes under us at any instant.
struct LivePlane
{
    int axis = 2;               // 0 = yz, 1 = xz, 2 = xy
    int w = 0, h = 0;
    std::vector<float> value;   // w*h, row j = 0 at the BOTTOM
    std::vector<float> mask;    // w*h, 1 = fluid; empty when absent
};

// One past frame, kept so the run can be rewound.
//
// PLANES ONLY, deliberately. A single 512^3 volume is 512 MB; a history of
// volumes would be the machine's memory, and there is nothing to spend it on --
// the plane is what a rewind is for. So scrubbing back shows the published
// centre planes, and the fog and the off-centre cut plane follow the live
// frame, which the panel says out loud rather than silently showing stale data.
struct LiveSnapshot
{
    std::uint64_t frame = 0;
    double t = 0;
    std::vector<LivePlane> planes;

    std::size_t bytes() const;
};

// What the mesh builders read. Bundled rather than passed as a LiveField so the
// SAME code draws the live frame and a rewound one: a snapshot supplies the
// planes with no volume, the live segment supplies both, and the geometry that
// places them in world coordinates is identical either way.
struct LiveView
{
    const std::vector<LivePlane>* planes = nullptr;
    const std::vector<float>* volume = nullptr;   // null or empty when rewinding
    int volW = 0, volH = 0, volD = 0;
    int i0 = 0, j0 = 0, k0 = 0;
    int winW = 0, winH = 0, winD = 0;
    double hx = 1, hy = 1, hz = 1;

    bool hasVolume() const
    {
        return volume && !volume->empty() && volW > 0 && volH > 0 && volD > 0;
    }
};

class LiveField
{
public:
    ~LiveField();

    // Attach by name (no leading slash). Returns "" on success. Attaching to a
    // name that does not exist yet is a normal, quiet failure while a run is
    // still starting up.
    std::string attach(const std::string& name);
    void detach();
    bool attached() const { return base_ != nullptr; }
    const std::string& name() const { return name_; }

    // Copies the newest COMPLETE frame out. True when a new frame arrived.
    // Cheap when the frame counter has not moved.
    bool poll();

    // True while the publishing run is alive; false once it exits cleanly.
    bool running() const;
    // The publishing process, so a viewer can stop a run it did not start.
    int publisherPid() const;
    // True when the segment belongs to a process that no longer exists: a run
    // killed with SIGKILL never gets to unlink it. Attaching to one of these
    // shows a dead run as live and blocks starting a new one.
    static bool isStale(const std::string& name);
    // SIGTERM to the publisher. Returns "" when the signal was sent. The run
    // shuts down cleanly on it: RunOutput's destructor clears `running` and
    // unlinks the segment, exactly as an interactive Ctrl-C would.
    std::string stopPublisher();

    std::uint64_t frame() const { return frame_; }
    double time() const { return t_; }
    const std::vector<LivePlane>& planes() const { return planes_; }
    const std::vector<dns::ShmField>& fields() const { return fields_; }

    void select(int i) { want_ = i; }
    int selected() const { return sel_; }

    // --- rewind ------------------------------------------------------------
    // Every frame this viewer decoded, oldest first, the newest being the one
    // the live view shows. Dropped from the front once the budget is reached,
    // and CLEARED whenever the selected field changes: a timeline that mixed
    // vort and speed frames would be a timeline of two different runs.
    const std::deque<LiveSnapshot>& history() const { return hist_; }
    void setHistoryBudget(std::size_t bytes);
    std::size_t historyBudget() const { return histBudget_; }
    std::size_t historyBytes() const { return histBytes_; }
    void clearHistory();

    // The live frame, and one rewound to a snapshot. Same geometry either way:
    // the grid does not move during a run.
    LiveView view() const;
    LiveView viewOf(const LiveSnapshot&) const;

    // The 3D volume of the SELECTED field; empty for a 2D run, when the run
    // exceeded the volume budget, or when volume copying is off.
    int volW() const { return volW_; }
    int volH() const { return volH_; }
    int volD() const { return volD_; }
    const std::vector<float>& volume() const { return volume_; }
    // True when the run publishes volumes at all -- i.e. when an arbitrary cut
    // plane and volumetric rendering are possible.
    bool hasVolume() const;
    // Copying the volume is the expensive part of a frame; off by default.
    void wantVolume(bool on) { wantVolume_ = on; }

    // Grid geometry, so the viewer can place a plane in world coordinates.
    int i0() const { return i0_; }
    int j0() const { return j0_; }
    int k0() const { return k0_; }
    // The written window's size, which is what places a plane at mid-depth.
    int winW() const { return winW_; }
    int winH() const { return winH_; }
    int winD() const { return winD_; }
    double hx() const { return hx_; }
    double hy() const { return hy_; }
    double hz() const { return hz_; }

private:
    const dns::ShmHeader* header() const;
    LiveView geomView() const;   // the grid geometry, with no data attached

    std::string name_;
    const unsigned char* base_ = nullptr;
    std::size_t bytes_ = 0;
    int fd_ = -1;

    std::vector<dns::ShmField> fields_;
    std::vector<LivePlane> planes_;
    std::vector<float> volume_;
    std::uint64_t frame_ = 0;
    double t_ = 0;
    int want_ = 0, sel_ = -1;
    int i0_ = 0, j0_ = 0, k0_ = 0;
    int winW_ = 0, winH_ = 0, winD_ = 0;
    int volW_ = 0, volH_ = 0, volD_ = 0;
    double hx_ = 1, hy_ = 1, hz_ = 1;
    bool wantVolume_ = false;

    std::deque<LiveSnapshot> hist_;
    std::size_t histBytes_ = 0;
    // 256 MB of planes is thousands of frames of a 512^2 run -- generous for
    // what it costs, and a hard ceiling so a long run cannot grow without end.
    std::size_t histBudget_ = 256ull << 20;
};

// Colour for a normalized value in [0,1], matching the published `diverging`
// hint: signed fields (vorticity) get a blue-dark-red ramp whose middle sinks
// into the background, magnitudes (speed) get a dark-to-hot sequential ramp.
// Both are built for a dark viewport, the house convention for artifacts.
void fieldColor(float t, bool diverging, unsigned char& r, unsigned char& g,
                unsigned char& b);

} // namespace ui
