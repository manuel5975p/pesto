// jobs_hog MB: allocate and touch MB megabytes, then exit 0. The live jobserver
// test runs it under a smaller memory cap to provoke a cgroup OOM kill.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "usage: jobs_hog MB\n");
        return 2;
    }
    const long mb = std::strtol(argv[1], nullptr, 10);
    std::vector<char*> blocks;
    for (long i = 0; i < mb; ++i) {
        char* p = static_cast<char*>(std::malloc(1 << 20));
        if (!p) return 3;
        std::memset(p, int(i), 1 << 20);
        blocks.push_back(p);
    }
    std::printf("touched %ld MB\n", mb);
    return 0;
}
