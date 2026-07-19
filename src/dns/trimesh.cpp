#include "dns/trimesh.hpp"

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>

namespace dns {

namespace {

// Merges bit-identical vertices so the mesh has shared edges/vertices.
struct MeshBuilder
{
    TriMesh m;
    std::map<std::array<double, 3>, int> ids;

    int vert(std::array<double, 3> v)
    {
        const auto [it, fresh] = ids.try_emplace(v, int(m.verts.size()));
        if (fresh) m.verts.push_back(v);
        return it->second;
    }

    void tri(int a, int b, int c)
    {
        if (a != b && b != c && c != a) m.tris.push_back({a, b, c});
    }
};

std::expected<std::vector<char>, std::string> slurp(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return std::unexpected("cannot open " + path);
    const auto n = f.tellg();
    std::vector<char> buf(static_cast<std::size_t>(n), '\0');
    f.seekg(0);
    f.read(buf.data(), n);
    if (!f) return std::unexpected("read failed: " + path);
    return buf;
}

TriMesh parseStlBinary(const std::vector<char>& buf, std::uint32_t n)
{
    MeshBuilder b;
    const char* p = buf.data() + 84;
    for (std::uint32_t t = 0; t < n; ++t)
    {
        p += 12; // facet normal: recomputed from winding, ignored
        int id[3];
        for (int v = 0; v < 3; ++v, p += 12)
        {
            float f[3];
            std::memcpy(f, p, sizeof f);
            id[v] = b.vert({double(f[0]), double(f[1]), double(f[2])});
        }
        p += 2; // attribute byte count
        b.tri(id[0], id[1], id[2]);
    }
    return std::move(b.m);
}

std::expected<TriMesh, std::string> parseStlAscii(const std::vector<char>& buf)
{
    MeshBuilder b;
    std::istringstream ss(std::string(buf.begin(), buf.end()));
    std::string tok;
    int id[3];
    int nv = 0;
    while (ss >> tok)
    {
        if (tok != "vertex") continue;
        double x, y, z;
        if (!(ss >> x >> y >> z))
            return std::unexpected("stl: malformed ASCII vertex");
        id[nv++] = b.vert({x, y, z});
        if (nv == 3)
        {
            b.tri(id[0], id[1], id[2]);
            nv = 0;
        }
    }
    if (nv != 0) return std::unexpected("stl: vertex count not a multiple of 3");
    if (b.m.tris.empty()) return std::unexpected("stl: no triangles");
    return std::move(b.m);
}

std::string lowerExt(const std::string& path)
{
    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos) return {};
    std::string e = path.substr(dot);
    std::ranges::transform(e, e.begin(),
                           [](unsigned char c) { return char(std::tolower(c)); });
    return e;
}

} // namespace

Aabb bounds(const TriMesh& m)
{
    Aabb bb{{1e300, 1e300, 1e300}, {-1e300, -1e300, -1e300}};
    for (const auto& v : m.verts)
        for (int a = 0; a < 3; ++a)
        {
            bb.lo[a] = std::min(bb.lo[a], v[a]);
            bb.hi[a] = std::max(bb.hi[a], v[a]);
        }
    return bb;
}

std::expected<TriMesh, std::string> loadStl(const std::string& path)
{
    auto buf = slurp(path);
    if (!buf) return std::unexpected(std::move(buf.error()));
    // Binary iff the 50-byte-record layout matches the file size exactly.
    if (buf->size() >= 84)
    {
        std::uint32_t n{};
        std::memcpy(&n, buf->data() + 80, sizeof n);
        if (n > 0 && buf->size() == 84 + 50ull * n)
        {
            auto m = parseStlBinary(*buf, n);
            if (m.tris.empty()) return std::unexpected("stl: no triangles");
            return m;
        }
    }
    return parseStlAscii(*buf);
}

std::expected<TriMesh, std::string> loadGltf(const std::string& path)
{
    const cgltf_options opt{};
    cgltf_data* raw = nullptr;
    if (cgltf_parse_file(&opt, path.c_str(), &raw) != cgltf_result_success)
        return std::unexpected("gltf: parse failed: " + path);
    const std::unique_ptr<cgltf_data, void (*)(cgltf_data*)> data(raw, cgltf_free);
    if (cgltf_load_buffers(&opt, raw, path.c_str()) != cgltf_result_success)
        return std::unexpected("gltf: buffer load failed: " + path);

    MeshBuilder b;
    for (cgltf_size ni = 0; ni < raw->nodes_count; ++ni)
    {
        const cgltf_node& node = raw->nodes[ni];
        if (!node.mesh) continue;
        cgltf_float wm[16]; // column-major world transform
        cgltf_node_transform_world(&node, wm);
        for (cgltf_size pi = 0; pi < node.mesh->primitives_count; ++pi)
        {
            const cgltf_primitive& prim = node.mesh->primitives[pi];
            if (prim.type != cgltf_primitive_type_triangles) continue;
            const cgltf_accessor* pos = nullptr;
            for (cgltf_size ai = 0; ai < prim.attributes_count; ++ai)
                if (prim.attributes[ai].type == cgltf_attribute_type_position)
                    pos = prim.attributes[ai].data;
            if (!pos) continue;

            std::vector<int> id(pos->count);
            for (cgltf_size v = 0; v < pos->count; ++v)
            {
                cgltf_float f[3]{};
                if (!cgltf_accessor_read_float(pos, v, f, 3))
                    return std::unexpected("gltf: POSITION read failed");
                id[v] = b.vert(
                    {double(wm[0]) * f[0] + double(wm[4]) * f[1] + double(wm[8]) * f[2] + double(wm[12]),
                     double(wm[1]) * f[0] + double(wm[5]) * f[1] + double(wm[9]) * f[2] + double(wm[13]),
                     double(wm[2]) * f[0] + double(wm[6]) * f[1] + double(wm[10]) * f[2] + double(wm[14])});
            }
            if (prim.indices)
                for (cgltf_size t = 0; t + 2 < prim.indices->count; t += 3)
                    b.tri(id[cgltf_accessor_read_index(prim.indices, t)],
                          id[cgltf_accessor_read_index(prim.indices, t + 1)],
                          id[cgltf_accessor_read_index(prim.indices, t + 2)]);
            else
                for (cgltf_size t = 0; t + 2 < pos->count; t += 3)
                    b.tri(id[t], id[t + 1], id[t + 2]);
        }
    }
    if (b.m.tris.empty()) return std::unexpected("gltf: no triangles");
    return std::move(b.m);
}

std::expected<TriMesh, std::string> loadMesh(const std::string& path)
{
    const std::string e = lowerExt(path);
    if (e == ".stl") return loadStl(path);
    if (e == ".gltf" || e == ".glb") return loadGltf(path);
    return std::unexpected("unsupported mesh extension: " + path);
}

void transform(TriMesh& m, double scale, double pitchRad, double yawRad,
               std::array<double, 3> center)
{
    const Aabb bb = bounds(m);
    const std::array<double, 3> c0{0.5 * (bb.lo[0] + bb.hi[0]),
                                   0.5 * (bb.lo[1] + bb.hi[1]),
                                   0.5 * (bb.lo[2] + bb.hi[2])};
    const double cp = std::cos(pitchRad), sp = std::sin(pitchRad);
    const double cy = std::cos(yawRad), sy = std::sin(yawRad);
    for (auto& v : m.verts)
    {
        double x = (v[0] - c0[0]) * scale;
        double y = (v[1] - c0[1]) * scale;
        double z = (v[2] - c0[2]) * scale;
        const double y1 = y * cp - z * sp, z1 = y * sp + z * cp; // pitch about x
        const double x2 = x * cy - y1 * sy, y2 = x * sy + y1 * cy; // yaw about z
        v = {x2 + center[0], y2 + center[1], z1 + center[2]};
    }
}

} // namespace dns
