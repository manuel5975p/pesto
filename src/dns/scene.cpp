#include "dns/scene.hpp"

#include "dns/bodies.hpp"
#include "dns/geometry.hpp"
#include "dns/meshsdf.hpp"
#include "dns/trimesh.hpp"

#define TOML_EXCEPTIONS 0
#include <tomlplusplus/toml.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <numbers>
#include <sstream>

namespace dns {
namespace {

// --- scene extraction --------------------------------------------------------
// toml++ (third_party/tomlplusplus, v3.4.0) does the lexing; the strictness
// the CLI has everywhere else -- unknown keys, wrong types, missing required
// fields are hard errors, never silent defaults -- is enforced here by
// iterating every table key instead of looking up the expected ones.

std::string bad(const std::string& where, const std::string& what)
{
    return "scene: " + where + ": " + what;
}

// Finite number out of a TOML integer or float. TOML admits `inf` and `nan`
// literals, so finiteness is checked here, once.
bool finiteNum(const toml::node& n, double& out)
{
    if (const auto v = n.value<double>(); v && std::isfinite(*v))
    {
        out = *v;
        return true;
    }
    return false;
}

std::expected<SceneObject, std::string> parseObject(const toml::table& o,
                                                    std::size_t i)
{
    const std::string where = "objects[" + std::to_string(i) + "]";

    const toml::node* kind = o.get("kind");
    if (!kind || !kind->is_string())
        return std::unexpected(bad(where, "requires a string `kind`"));

    SceneObject ob;
    const std::string_view ks = kind->value_or(std::string_view{});
    if (ks == "cylinder") ob.kind = SceneObjectKind::Cylinder;
    else if (ks == "sphere") ob.kind = SceneObjectKind::Sphere;
    else if (ks == "cube") ob.kind = SceneObjectKind::Cube;
    else if (ks == "lens") ob.kind = SceneObjectKind::Lens;
    else if (ks == "mesh") ob.kind = SceneObjectKind::Mesh;
    else
        return std::unexpected(bad(where, "unknown kind '" + std::string(ks)
                                   + "' (cylinder, sphere, cube, lens, mesh)"));

    const bool mesh = ob.kind == SceneObjectKind::Mesh;
    for (const auto& [key, v] : o)
    {
        const std::string k(key.str());
        if (k == "kind") continue;
        if (k == "name")
        {
            const auto s = v.value<std::string>();
            if (!s || s->empty())
                return std::unexpected(bad(where, "`name` must be a non-empty "
                                                  "string"));
            // The label becomes a CSV column suffix: keep it token-shaped.
            if (!validSceneName(*s))
                return std::unexpected(bad(where, "`name` must be "
                                                  "[A-Za-z0-9_-]"));
            ob.label = *s;
            continue;
        }
        if (k == "path")
        {
            if (!mesh)
                return std::unexpected(bad(where, "`path` applies to mesh "
                                                  "objects only"));
            const auto s = v.value<std::string>();
            if (!s || s->empty())
                return std::unexpected(bad(where, "`path` must be a non-empty "
                                                  "string"));
            ob.meshPath = *s;
            continue;
        }
        if (k == "robust")
        {
            if (!mesh || !v.is_boolean())
                return std::unexpected(bad(where, "`robust` is a bool for mesh "
                                                  "objects only"));
            ob.robust = v.value_or(false);
            continue;
        }
        double* num = k == "x" ? &ob.x
                    : k == "y" ? &ob.y
                    : k == "z" ? &ob.z
                    : k == "d" ? &ob.d
                    : k == "aspect" && ob.kind == SceneObjectKind::Lens ? &ob.aspect
                    : mesh && k == "scale" ? &ob.scale
                    : mesh && k == "pitch" ? &ob.pitch
                    : mesh && k == "yaw"   ? &ob.yaw
                                           : nullptr;
        if (!num)
            return std::unexpected(bad(where, "unknown key `" + k + "`"));
        if (!finiteNum(v, *num))
            return std::unexpected(bad(where, "`" + k + "` must be a finite "
                                                        "number"));
    }

    if (!(ob.d > 0))
        return std::unexpected(bad(where, "`d` must be > 0 (units of D)"));
    if (ob.kind == SceneObjectKind::Lens && !(ob.aspect > 1.0))
        return std::unexpected(bad(where, "lens `aspect` must be > 1"));
    if (mesh && ob.meshPath.empty())
        return std::unexpected(bad(where, "mesh objects require `path`"));
    if (mesh && ob.scale < 0)
        return std::unexpected(bad(where, "`scale` must be >= 0 (0 = fit d)"));
    return ob;
}

std::expected<void, std::string> parseModel(const toml::table& m, Scene& sc)
{
    const toml::node* type = m.get("type");
    if (!type || !type->is_string())
        return std::unexpected(bad("model", "requires a string `type`"));
    const std::string_view ts = type->value_or(std::string_view{});
    if (ts == "dns") sc.model = SceneModel::Dns;
    else if (ts == "les") sc.model = SceneModel::Les;
    else
        return std::unexpected(bad("model", "unknown type '" + std::string(ts)
                                   + "' (dns, les)"));
    for (const auto& [key, v] : m)
    {
        const std::string k(key.str());
        if (k == "type") continue;
        if (k == "cs" && sc.model == SceneModel::Les)
        {
            double cs{};
            if (!finiteNum(v, cs) || cs <= 0)
                return std::unexpected(bad("model", "`cs` must be a finite "
                                                    "number > 0"));
            sc.cs = cs;
            continue;
        }
        if (k == "sgs" && sc.model == SceneModel::Les)
        {
            const std::string_view m =
                v.is_string() ? v.value_or(std::string_view{})
                              : std::string_view{};
            if (m == "smag") sc.sgs = SgsModel::Smagorinsky;
            else if (m == "sigma") sc.sgs = SgsModel::Sigma;
            else
                return std::unexpected(bad("model", "`sgs` must be \"smag\" or "
                                                    "\"sigma\""));
            continue;
        }
        return std::unexpected(bad("model", "unknown key `" + k + "`"));
    }
    if (sc.model == SceneModel::Les && sc.cs == 0)
        return std::unexpected(bad("model", "type \"les\" requires `cs`"));
    return {};
}

std::array<double, 6> merge(const std::array<double, 6>& a,
                            const std::array<double, 6>& b)
{
    return {std::min(a[0], b[0]), std::min(a[1], b[1]), std::min(a[2], b[2]),
            std::max(a[3], b[3]), std::max(a[4], b[4]), std::max(a[5], b[5])};
}

} // namespace

const char* name(SceneObjectKind k)
{
    switch (k)
    {
    case SceneObjectKind::Cylinder: return "cylinder";
    case SceneObjectKind::Sphere: return "sphere";
    case SceneObjectKind::Cube: return "cube";
    case SceneObjectKind::Lens: return "lens";
    case SceneObjectKind::Mesh: return "mesh";
    }
    return "?";
}

RotatedCube sceneCube(const Grid& g, double cx, double cy, double cz, double r)
{
    // 2D: no third dimension to be a cube in, so the prism reading stands.
    if (g.nz == 1) return RotatedCube{cx, cy, r, 0.0, g.hz};
    // Half the SIDE of the rotated square, whose half-DIAGONAL is r.
    const double hh = r / std::numbers::sqrt2;
    return RotatedCube{cx, cy, r, cz - hh, cz + hh};
}

bool Scene::zInvariant() const
{
    return std::ranges::none_of(objects, [](const SceneObject& o) {
        return o.kind == SceneObjectKind::Sphere || o.kind == SceneObjectKind::Mesh
               || o.kind == SceneObjectKind::Cube;
    });
}

bool Scene::has3d() const
{
    return std::ranges::any_of(objects, needs3d);
}

std::string Scene::describe() const
{
    std::ostringstream os;
    os << "scene:      model " << (model == SceneModel::Les ? "LES" : "DNS");
    if (model == SceneModel::Les)
        os << " (" << (sgs == SgsModel::Sigma ? "sigma" : "smag")
           << ", Cs=" << cs << ")";
    os << ", " << objects.size() << " object"
       << (objects.size() == 1 ? "" : "s") << "\n";
    for (std::size_t i = 0; i < objects.size(); ++i)
    {
        const auto& o = objects[i];
        os << "  [" << i << "] " << name(o.kind);
        if (!o.label.empty()) os << " '" << o.label << "'";
        os << "  at (" << o.x << ", " << o.y << ", " << o.z << ") D"
           << "  d=" << o.d << " D";
        if (o.kind == SceneObjectKind::Lens) os << "  aspect=" << o.aspect;
        if (o.kind == SceneObjectKind::Mesh) os << "  " << o.meshPath;
        os << "\n";
    }
    return os.str();
}

std::expected<Scene, std::string> parseScene(std::string_view text)
{
    const auto res = toml::parse(text);
    if (!res)
    {
        const auto& e = res.error();
        return std::unexpected("scene: TOML parse error: "
                               + std::string(e.description()) + " (line "
                               + std::to_string(e.source().begin.line) + ")");
    }
    const toml::table& root = res.table();

    Scene sc;
    for (const auto& [key, v] : root)
    {
        const std::string k(key.str());
        if (k == "model")
        {
            const toml::table* m = v.as_table();
            if (!m) return std::unexpected(bad("model", "must be a [model] table"));
            if (auto r = parseModel(*m, sc); !r) return std::unexpected(r.error());
            continue;
        }
        if (k == "objects")
        {
            const toml::array* a = v.as_array();
            if (!a || !a->is_homogeneous(toml::node_type::table))
                return std::unexpected(bad("objects", "must be an array of "
                                                      "tables ([[objects]])"));
            for (std::size_t i = 0; i < a->size(); ++i)
            {
                auto ob = parseObject(*a->get(i)->as_table(), i);
                if (!ob) return std::unexpected(ob.error());
                sc.objects.push_back(std::move(*ob));
            }
            continue;
        }
        return std::unexpected("scene: unknown top-level key `" + k + "`");
    }

    if (sc.objects.empty())
        return std::unexpected("scene: at least one [[objects]] entry is "
                               "required");
    // Default labels, then require uniqueness (they become CSV columns).
    for (std::size_t i = 0; i < sc.objects.size(); ++i)
        if (sc.objects[i].label.empty())
            sc.objects[i].label = defaultSceneName(sc.objects[i].kind, i);
    if (auto n = checkSceneNames(sc); !n) return std::unexpected(n.error());
    return sc;
}

// --- rules shared with tools (scene.hpp) -------------------------------------

bool validSceneName(std::string_view s)
{
    if (s.empty()) return false;
    for (char c : s)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-'))
            return false;
    return true;
}

std::string defaultSceneName(SceneObjectKind k, std::size_t index)
{
    return std::string(name(k)) + std::to_string(index);
}

std::expected<void, std::string> checkSceneNames(const Scene& sc)
{
    for (const auto& o : sc.objects)
        if (!validSceneName(o.label))
            return std::unexpected("scene: object `name` must be non-empty and "
                                   "[A-Za-z0-9_-] (got '" + o.label + "')");
    for (std::size_t i = 0; i < sc.objects.size(); ++i)
        for (std::size_t j = i + 1; j < sc.objects.size(); ++j)
            if (sc.objects[i].label == sc.objects[j].label)
                return std::unexpected("scene: duplicate object name '"
                                       + sc.objects[i].label + "'");
    return {};
}

bool needs3d(const SceneObject& o)
{
    return o.kind == SceneObjectKind::Sphere || o.kind == SceneObjectKind::Mesh;
}

std::array<double, 6> sceneObjectBox(const Grid& g, const Placement& ref,
                                     const SceneObject& o)
{
    const double lzFull = g.nz == 1 ? g.hz : double(g.nz) * g.hz;
    const double cx = ref.cx + o.x * ref.D, cy = ref.cy + o.y * ref.D,
                 cz = ref.cz + o.z * ref.D, D = o.d * ref.D, r = D / 2.0;
    std::array<double, 6> bb{cx - r, cy - r, 0.0, cx + r, cy + r, lzFull};
    if (o.kind == SceneObjectKind::Sphere)
    {
        bb[2] = cz - r;
        bb[5] = cz + r;
    }
    else if (o.kind == SceneObjectKind::Cube)
    {
        // A cube is as tall as its side, not as tall as the box (sceneCube).
        const RotatedCube c = sceneCube(g, cx, cy, cz, r);
        bb[2] = c.z0;
        bb[5] = c.z1;
    }
    else if (o.kind == SceneObjectKind::Lens)
    {
        const double w = o.aspect * r;
        bb[0] = cx - w;
        bb[3] = cx + w;
    }
    return bb;
}

bool sceneBoxInside(const Grid& g, const std::array<double, 6>& bb)
{
    const double lzFull = g.nz == 1 ? g.hz : double(g.nz) * g.hz;
    // z is exempt for z-invariant objects: they span the box by design.
    const bool zSpans = bb[2] == 0.0 && bb[5] == lzFull;
    const double lx = double(g.nx) * g.hx, ly = double(g.ny) * g.hy;
    return !(bb[0] <= g.hx || bb[3] >= lx - g.hx || bb[1] <= g.hy
             || bb[4] >= ly - g.hy
             || (!zSpans && g.nz > 1 && (bb[2] <= g.hz || bb[5] >= lzFull - g.hz)));
}

std::expected<Scene, std::string> loadScene(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::unexpected("cannot read scene file '" + path + "'");
    std::ostringstream os;
    os << f.rdbuf();
    auto sc = parseScene(os.str());
    if (!sc) return std::unexpected(sc.error() + " ('" + path + "')");
    return sc;
}

std::expected<SceneGeometry, std::string>
buildSceneGeometry(const Grid& g, const Placement& ref, const Scene& scene)
{
    using std::numbers::pi;

    SceneGeometry out;
    std::vector<Sdf> parts;
    parts.reserve(scene.objects.size());

    for (const auto& o : scene.objects)
    {
        const double cx = ref.cx + o.x * ref.D, cy = ref.cy + o.y * ref.D,
                     cz = ref.cz + o.z * ref.D, D = o.d * ref.D, r = D / 2.0;
        // Shared with the editor (scene.hpp): one definition of an object's
        // box. The mesh case overwrites it once the file is loaded.
        std::array<double, 6> bb = sceneObjectBox(g, ref, o);
        switch (o.kind)
        {
        case SceneObjectKind::Cylinder:
            parts.push_back(sdfOf(CylinderZ{cx, cy, r}));
            break;
        case SceneObjectKind::Sphere:
            if (g.nz == 1)
                return std::unexpected("scene object '" + o.label
                                       + "' is 3D (sphere); nz == 1 leaves it "
                                         "with no solid cells -- pass --nz");
            parts.push_back(sdfOf(Sphere{cx, cy, cz, r}));
            break;
        case SceneObjectKind::Cube:
            parts.push_back(sdfOf(sceneCube(g, cx, cy, cz, r)));
            break;
        case SceneObjectKind::Lens:
        {
            const double hh = r, w = o.aspect * hh;
            parts.push_back(sdfOf(Lens{cx, cy, w, hh}));
            break;
        }
        case SceneObjectKind::Mesh:
        {
            if (g.nz == 1)
                return std::unexpected("scene object '" + o.label
                                       + "' is 3D (mesh); nz == 1 leaves it "
                                         "with no solid cells -- pass --nz");
            auto meshOr = loadMesh(o.meshPath);
            if (!meshOr)
                return std::unexpected("scene object '" + o.label + "': "
                                       + meshOr.error());
            TriMesh mesh = std::move(*meshOr);
            if (mesh.tris.empty())
                return std::unexpected("scene object '" + o.label
                                       + "': mesh has no triangles");
            const Aabb bb0 = bounds(mesh);
            const double ext0 =
                std::max({bb0.hi[0] - bb0.lo[0], bb0.hi[1] - bb0.lo[1],
                          bb0.hi[2] - bb0.lo[2]});
            if (!(ext0 > 0))
                return std::unexpected("scene object '" + o.label
                                       + "': degenerate mesh bounds");
            const double sc = o.scale > 0 ? o.scale : D / ext0;
            transform(mesh, sc, o.pitch * pi / 180.0, o.yaw * pi / 180.0,
                      {cx, cy, cz});
            const Aabb b = bounds(mesh);
            bb = {b.lo[0], b.lo[1], b.lo[2], b.hi[0], b.hi[1], b.hi[2]};
            auto sdfp = std::make_shared<MeshSdf>(std::move(mesh), o.robust);
            if (!sdfp->watertight() && !o.robust)
                std::fprintf(stderr, "dns: warning: scene object '%s' is not "
                                     "watertight; consider `robust = true`\n",
                             o.label.c_str());
            parts.push_back([sdfp](double x, double y, double z) {
                return (*sdfp)(x, y, z);
            });
            break;
        }
        }

        // The solid must sit strictly inside the interior; a body poking
        // through the box boundary is the fence/floor rig, which a scene does
        // not model (their wall sealing and mass-balance plumbing is special).
        // z is exempt for z-invariant objects (they span the box by design).
        if (!sceneBoxInside(g, bb))
            return std::unexpected("scene object '" + o.label
                                   + "' reaches the domain boundary; shrink d, "
                                     "move it, or raise --dr");
        out.objects.push_back({o.label, bb});
        out.bbox = out.objects.size() == 1 ? bb : merge(out.bbox, bb);
    }

    // Union of solids: fluid is where EVERY object says fluid (> 0), so the
    // union sdf is the min. Exact distance outside; inside overlaps it is
    // merely a bound, which the sampler never relies on.
    out.sdf = [parts](double x, double y, double z) {
        double d = 1e9;
        for (const auto& f : parts) d = std::min(d, f(x, y, z));
        return d;
    };

    // One generic sampled cut for the whole scene: the mesh/golf {8,8} budget,
    // on a single replicated x-y layer when every object is z-invariant.
    auto cut = sampleCut(g, out.sdf, {8, 8, scene.zInvariant()});
    if (!cut) return std::unexpected(cut.error());
    out.cut = std::move(*cut);
    return out;
}

} // namespace dns
