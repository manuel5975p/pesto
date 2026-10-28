// Job server, pure parts: SHA-256 vectors, strict JSON, job-spec validation
// and its hard caps, and the admission rule. No processes, no sockets.

#include "check.hpp"

#include "jobs/json.hpp"
#include "jobs/sched.hpp"
#include "jobs/sha256.hpp"
#include "jobs/spec.hpp"

#include <format>
#include <string>
#include <vector>

using dnstest::check;
using namespace jobs;

namespace {

std::string hashOf(const std::string& s)
{
    Sha256 h;
    h.update({reinterpret_cast<const std::byte*>(s.data()), s.size()});
    const auto d = h.finish();
    return toHex(d);
}

void sha()
{
    std::printf("sha256\n");
    check(hashOf("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty");
    check(hashOf("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "abc");
    check(hashOf("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
          "448-bit two-block vector");
    check(hashOf(std::string(1000000, 'a')) ==
              "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
          "one million a");
    // 55/56/64-byte boundaries against the one-shot of the same bytes split up.
    for (std::size_t n : {55u, 56u, 63u, 64u, 65u}) {
        const std::string s(n, 'x');
        Sha256 h;
        h.update({reinterpret_cast<const std::byte*>(s.data()), 7});
        h.update({reinterpret_cast<const std::byte*>(s.data()) + 7, n - 7});
        const auto d = h.finish();
        check(toHex(d) == hashOf(s), std::format("split update == one shot at {} bytes", n));
    }
}

void jsonTests()
{
    std::printf("json\n");
    const auto v = json::parse(R"( {"a":[1,2.5,-3e2,true,false,null],"s":"x\"\\\n\u00e9\ud83d\ude00","o":{}} )");
    check(v.has_value(), "parses a document");
    if (v) {
        check(json::dump(*v) == "{\"a\":[1,2.5,-300,true,false,null],\"s\":\"x\\\"\\\\\\n\xc3\xa9\xf0\x9f\x98\x80\",\"o\":{}}",
              "round trip, integers compact, UTF-8 from \\u and surrogates");
        const auto again = json::parse(json::dump(*v));
        check(again && json::dump(*again) == json::dump(*v), "dump parses back to itself");
    }
    for (const char* bad : {"{\"a\":1,\"a\":2}", "[1,]", "{} x", "\"\\q\"", "\"\\ud83d\"", "01", "1.", "-",
                            "\"tab\there\"", "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[1]]]]]]]]"
                            "]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]", ""})
        check(!json::parse(bad).has_value(), std::format("rejects {}", std::string(bad).substr(0, 24)));
}

json::Value spec(int runtime, long long mem, const std::string& extraKey = "")
{
    json::Object step{{"exe", "/bin/true"}, {"sha256", std::string(64, 'a')}, {"args", json::Array{"x"}}};
    json::Object o{{"resource", "gpu"}, {"max_runtime_s", runtime}, {"max_mem_mb", mem},
                   {"cwd", "/tmp"}, {"steps", json::Array{step}}};
    if (!extraKey.empty()) o.emplace_back(extraKey, 1);
    return o;
}

void specTests()
{
    std::printf("spec\n");
    check(parseSpec(spec(1800, 16384)).has_value(), "30 min / 16 GiB accepted");
    check(!parseSpec(spec(1801, 1024)).has_value(), "1801 s rejected");
    check(!parseSpec(spec(0, 1024)).has_value(), "0 s rejected");
    check(!parseSpec(spec(60, 16385)).has_value(), "16385 MB rejected");
    check(!parseSpec(spec(60, kMinMemMb - 1)).has_value(), "below the runner's own floor rejected");
    check(!parseSpec(spec(60, 1024, "nice")).has_value(), "unknown key rejected");
    auto relExe = spec(60, 1024);
    std::get<json::Object>(std::get<json::Array>(std::get<json::Object>(relExe.v)[4].second.v)[0].v)[0].second = "bin/true";
    check(!parseSpec(relExe).has_value(), "relative exe rejected");
    auto badHash = spec(60, 1024);
    std::get<json::Object>(std::get<json::Array>(std::get<json::Object>(badHash.v)[4].second.v)[0].v)[1].second = "ABC";
    check(!parseSpec(badHash).has_value(), "malformed sha256 rejected");
    const auto ok = parseSpec(spec(60, 1024));
    if (ok) {
        const auto back = parseSpec(toJson(*ok));
        check(back && json::dump(toJson(*back)) == json::dump(toJson(*ok)), "spec round-trips through its JSON");
        check(ok->resource == Resource::gpu && !ok->benchmark && ok->steps[0].args == std::vector<std::string>{"x"},
              "fields read");
    }
    check(exitCode(State::done, 0) == 0 && exitCode(State::failed, 3) == 3 && exitCode(State::failed, 0) == 1 &&
              exitCode(State::timeout, 0) == 124 && exitCode(State::oom, 0) == 137 &&
              exitCode(State::rejected, 0) == 3,
          "exit-code mapping");
}

std::vector<std::int64_t> run(std::vector<Pending> q, Occupancy busy = {}, bool gpuFree = true)
{
    return admit(q, busy, gpuFree);
}

void schedTests()
{
    using V = std::vector<std::int64_t>;
    constexpr auto G = Resource::gpu, C = Resource::cpu;
    std::printf("admit\n");
    check(run({{1, G}, {2, G}, {3, C}, {4, C}}) == V{1, 3}, "one gpu + one cpu at a time");
    check(run({{1, G}, {2, C}}, {.gpu = true}) == V{2}, "busy gpu still lets cpu run");
    check(run({{1, G}, {2, G}}, {}, false) == V{}, "external gpu lock holds gpu jobs");
    check(run({{1, G}, {2, C}}, {}, false) == V{2}, "... but not cpu jobs");
    check(run({{1, G}, {2, G}, {3, C}}, {.gpu = true}) == V{3}, "FIFO per resource: 2 waits behind running gpu");
    check(run({{1, C, true}}) == V{1}, "benchmark starts on an idle machine");
    check(run({{1, C, true}}, {.gpu = true}) == V{}, "benchmark waits for any running job");
    check(run({{1, C, true}, {2, C}, {3, G}}, {.gpu = true}) == V{}, "nothing overtakes a waiting benchmark");
    check(run({{1, C}, {2, G, true}, {3, G}}) == V{1}, "earlier job starts, benchmark then waits for it");
    check(run({{1, G}, {2, C}}, {.benchmark = true}) == V{}, "nothing starts beside a running benchmark");
    check(run({{1, C}, {2, G}, {3, C, true}}, {.cpu = true}) == V{2}, "jobs ahead of a benchmark still start");
    check(run({{1, G, true}}, {}, false) == V{}, "benchmark needs the external gpu lock too");
}

} // namespace

int main()
{
    sha();
    jsonTests();
    specTests();
    schedTests();
    return dnstest::finish();
}
