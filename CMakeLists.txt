cmake_minimum_required(VERSION 3.25)
project(dns CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall$<SEMICOLON>-Wextra$<SEMICOLON>-Wpedantic>)

option(DNS_WITH_FFTW "Use FFTW3 for the pressure Poisson solve" ON)

# The GPU backend, in two flavours of the SAME sources. src/dns/gpu/*.hip is
# written in HIP and compiled by hipcc for AMD; for NVIDIA the identical files
# are compiled as CUDA (LANGUAGE CUDA => nvcc -x cu) against the header shim in
# src/compat/cuda, which renames the ~40 hip*/rocprim names onto their CUDA
# equivalents. There is no second implementation and no per-backend #if in any
# solver: see src/compat/cuda/README.md.
option(DNS_WITH_HIP "Build the GPU backend with HIP (ROCm / AMD)" OFF)
option(DNS_WITH_CUDA "Build the GPU backend with CUDA (NVIDIA)" OFF)
if(DNS_WITH_HIP AND DNS_WITH_CUDA)
    message(FATAL_ERROR
        "DNS_WITH_HIP and DNS_WITH_CUDA are mutually exclusive: they build the "
        "same sources into the same dns_gpu target. Pick one backend.")
endif()
if(DNS_WITH_HIP OR DNS_WITH_CUDA)
    set(DNS_WITH_GPU ON)
endif()

find_package(ZLIB REQUIRED) # fieldpng.cpp (PNG frame export)

add_library(dns_core
    src/dns/blockband.cpp
    src/dns/bodies.cpp
    src/dns/bunsen.cpp
    src/dns/channel_line.cpp
    src/dns/chem.cpp
    src/dns/cli.cpp
    src/dns/casegeom.cpp
    src/dns/chamber.cpp
    src/dns/config.cpp
    src/dns/diffusion.cpp
    src/dns/foamdict.cpp
    src/dns/foamio.cpp
    src/dns/framemap.cpp
    src/dns/fvcase.cpp
    src/dns/fvgrad.cpp
    src/dns/fvsimple.cpp
    src/dns/fvwalldist.cpp
    src/dns/fvref.cpp
    src/dns/cutcell.cpp
    src/dns/diagnostics.cpp
    src/dns/facemesh.cpp
    src/dns/fieldpng.cpp
    src/dns/flame1d.cpp
    src/dns/turb_ic.cpp
    src/dns/flow_ic.cpp
    src/dns/geometry.cpp
    src/dns/operators.cpp
    src/dns/output.cpp
    src/dns/comp.cpp
    src/dns/curvmesh.cpp
    src/dns/interrupt.cpp
    src/dns/la/amg.cpp
    src/dns/la/ell.cpp
    src/dns/la/graph.cpp
    src/dns/la/lines.cpp
    src/dns/la/structured.cpp
    src/dns/meshio.cpp
    src/dns/meshsdf.cpp
    src/dns/ogrid.cpp
    src/dns/poisson.cpp
    src/dns/polymesh.cpp
    src/dns/prop/spec.cpp
    src/dns/pseudo_time.cpp
    src/dns/rans_channel1d.cpp
    src/dns/rans_label.cpp
    src/dns/rotor.cpp
    src/dns/scene.cpp
    src/dns/shmframes.cpp
    src/dns/steady_stop.cpp
    src/dns/timestepper.cpp
    src/dns/trimesh.cpp
    src/dns/vtk_writer.cpp
    src/dns/walldist.cpp
    src/dns/webframes.cpp
    src/dns/wick.cpp
)
target_include_directories(dns_core PUBLIC src)
# cgltf (trimesh.cpp) + toml++ (scene.cpp); SYSTEM mutes third-party warnings.
target_include_directories(dns_core SYSTEM PRIVATE third_party)
# toml++ is built without exceptions in every TU that includes it (house rule:
# std::expected, no throw). PRIVATE: third_party is not on consumers' path.
target_compile_definitions(dns_core PRIVATE TOML_EXCEPTIONS=0)
target_link_libraries(dns_core PRIVATE ZLIB::ZLIB)

if(DNS_WITH_FFTW)
    find_path(FFTW3_INCLUDE_DIR fftw3.h REQUIRED)
    find_library(FFTW3_LIBRARY fftw3 REQUIRED)
    target_sources(dns_core PRIVATE src/dns/hit_ic.cpp)
    target_include_directories(dns_core PRIVATE ${FFTW3_INCLUDE_DIR})
    target_link_libraries(dns_core PRIVATE ${FFTW3_LIBRARY})
    target_compile_definitions(dns_core PUBLIC DNS_WITH_FFTW=1)
endif()

# `dns prop`: the propeller thrust-curve codes (BEMT, lifting line). Host-only
# physics on top of dns_core's config layer, so dns-ui (dns_core only) never
# carries them, and they never link dns_gpu.
find_package(Threads REQUIRED)
add_library(dns_prop
    src/dns/prop/bemt.cpp
    src/dns/prop/blade.cpp
    src/dns/prop/csv.cpp
    src/dns/prop/curve.cpp
    src/dns/prop/liftline.cpp
    src/dns/prop/numeric.cpp
    src/dns/prop/polar.cpp
)
target_link_libraries(dns_prop PUBLIC dns_core PRIVATE Threads::Threads)

# The one executable. Everything else in apps/ was deleted by the
# recomposition; `dns <subcommand>` covers the whole former 15-binary surface.
add_executable(dns apps/dns_main.cpp)
target_link_libraries(dns PRIVATE dns_core dns_prop)

# The simulation job server and its client (docs/jobs.md). Linux + systemd
# user session; links nothing of the solver, so it never goes stale with it.
add_library(simjobs STATIC
    src/jobs/io.cpp
    src/jobs/json.cpp
    src/jobs/runner.cpp
    src/jobs/sched.cpp
    src/jobs/server.cpp
    src/jobs/sha256.cpp
    src/jobs/spec.cpp
)
target_include_directories(simjobs PUBLIC src)
add_executable(simjobd apps/simjobd.cpp)
target_link_libraries(simjobd PRIVATE simjobs)
add_executable(simjob apps/simjob.cpp)
target_link_libraries(simjob PRIVATE simjobs)

# The one GPU source list, shared by both backends.
set(DNS_GPU_SOURCES
    src/dns/gpu/axis_bc.hip
    src/dns/gpu/nk.hip
    src/dns/gpu/bodyforce.hip
    src/dns/gpu/boussinesq.hip
    src/dns/gpu/chem.hip
    src/dns/gpu/comp.hip
    src/dns/gpu/comp_channel.hip
    src/dns/gpu/comp_conv.hip
    src/dns/gpu/comp_far.hip
    src/dns/gpu/comp_ghost.hip
    src/dns/gpu/comp_nscbc.hip
    src/dns/gpu/comp_sink.hip
    src/dns/gpu/nscbc_modal.hip
    src/dns/gpu/cutcell.hip
    src/dns/gpu/cutflux.hip
    src/dns/gpu/cutpoisson.hip
    src/dns/gpu/derived.hip
    src/dns/gpu/flame_turb.hip
    src/dns/gpu/device_util.hip
    src/dns/gpu/fv_simple.hip
    src/dns/gpu/fv_coupled.hip
    src/dns/gpu/fv_ras.hip
    src/dns/gpu/ghostfill.hip
    src/dns/gpu/kernels.hip
    src/dns/gpu/la/amg.hip
    src/dns/gpu/la/blockell.hip
    src/dns/gpu/la/krylov.hip
    src/dns/gpu/la/smooth.hip
    src/dns/gpu/les.hip
    src/dns/gpu/ogrid_bc.hip
    src/dns/gpu/poisson.hip
    src/dns/gpu/rans.hip
    src/dns/gpu/reductions.hip
    src/dns/gpu/rotortable.hip
    src/dns/gpu/runner_chamber.hip
    src/dns/gpu/runner_channel.hip
    src/dns/gpu/runner_comp.hip
    src/dns/gpu/runner_conv.hip
    src/dns/gpu/runner_flame.hip
    src/dns/gpu/runner_fv.hip
    src/dns/gpu/runner_ogrid.hip
    src/dns/gpu/runner_rb.hip
    src/dns/gpu/runner_incomp.hip
    src/dns/gpu/runner_periodic.hip
    src/dns/gpu/runner_wick.hip
    src/dns/gpu/species.hip
    src/dns/gpu/tfles.hip
    src/dns/gpu/timestepper.hip
    src/dns/gpu/tunnel.hip
)

if(DNS_WITH_HIP)
    # GPU arch is flexible: an explicit -DCMAKE_HIP_ARCHITECTURES wins;
    # otherwise auto-detect the installed GPU(s) at configure time and fall
    # back to a fat multi-arch list if no device is visible (e.g. CI, headless
    # build hosts). gfx1100 = RX 7900 (RDNA3), gfx1201 = RX 9070 (RDNA4).
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(_dns_detected_arch "")
        find_program(DNS_AMDGPU_ARCH NAMES amdgpu-arch
                     HINTS /opt/rocm/bin /opt/rocm/llvm/bin)
        if(DNS_AMDGPU_ARCH)
            execute_process(COMMAND ${DNS_AMDGPU_ARCH}
                            OUTPUT_VARIABLE _dns_arch_out
                            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
            string(REPLACE "\n" ";" _dns_arch_list "${_dns_arch_out}")
            list(REMOVE_DUPLICATES _dns_arch_list)
            list(FILTER _dns_arch_list INCLUDE REGEX "^gfx")
            set(_dns_detected_arch "${_dns_arch_list}")
        endif()
        if(_dns_detected_arch)
            set(CMAKE_HIP_ARCHITECTURES "${_dns_detected_arch}")
            message(STATUS "DNS: auto-detected GPU arch: ${CMAKE_HIP_ARCHITECTURES}")
        else()
            set(CMAKE_HIP_ARCHITECTURES "gfx1100;gfx1201")
            message(STATUS "DNS: no GPU detected, building fat binary for "
                           "${CMAKE_HIP_ARCHITECTURES} "
                           "(override with -DCMAKE_HIP_ARCHITECTURES)")
        endif()
    else()
        message(STATUS "DNS: using requested GPU arch: ${CMAKE_HIP_ARCHITECTURES}")
    endif()
    enable_language(HIP)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
    find_package(hip REQUIRED)
    find_package(hipfft REQUIRED)
    find_package(rocprim REQUIRED)

    add_library(dns_gpu ${DNS_GPU_SOURCES})
    target_link_libraries(dns_gpu
        PUBLIC dns_core hip::host
        PRIVATE hip::hipfft roc::rocprim)
    target_compile_definitions(dns_gpu PUBLIC DNS_WITH_GPU=1 DNS_WITH_HIP=1
                                              DNS_GPU_BACKEND_NAME="hip")
    target_compile_options(dns_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-Wall$<SEMICOLON>-Wextra>)

    # dns picks up the GPU backend and the runners when HIP is available.
    # (Target predates enable_language(HIP), so set its HIP arch explicitly.)
    target_link_libraries(dns PRIVATE dns_gpu)
    set_property(TARGET dns PROPERTY HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})
endif()

if(DNS_WITH_CUDA)
    # Same policy as the HIP branch above: an explicit
    # -DCMAKE_CUDA_ARCHITECTURES wins; otherwise ask the installed driver what
    # is in the machine ("native", which nvcc resolves at compile time) and fall
    # back to a fat binary when no device answers (CI, headless build hosts).
    # 75 = Turing, 86 = Ampere consumer (RTX 30xx), 89 = Ada, 120 = Blackwell.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(_dns_have_nv "")
        find_program(DNS_NVIDIA_SMI NAMES nvidia-smi)
        if(DNS_NVIDIA_SMI)
            execute_process(COMMAND ${DNS_NVIDIA_SMI} -L
                            OUTPUT_VARIABLE _dns_nv_out RESULT_VARIABLE _dns_nv_rc
                            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(_dns_nv_rc EQUAL 0 AND _dns_nv_out MATCHES "GPU 0")
                set(_dns_have_nv ON)
            endif()
        endif()
        if(_dns_have_nv)
            set(CMAKE_CUDA_ARCHITECTURES native)
            message(STATUS "DNS: CUDA arch: native (device detected)")
        else()
            set(CMAKE_CUDA_ARCHITECTURES 75 86 89 120)
            message(STATUS "DNS: no NVIDIA device detected, building fat binary "
                           "for ${CMAKE_CUDA_ARCHITECTURES} "
                           "(override with -DCMAKE_CUDA_ARCHITECTURES)")
        endif()
    else()
        message(STATUS "DNS: using requested CUDA arch: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # CMake 4.3's NVIDIA-CUDA compiler module stops at the c++20 dialect, so a
    # target inheriting CXX_STANDARD 23 is rejected as "does not know the flags"
    # -- even though nvcc has accepted -std=c++23 for several releases. The
    # library is C++23 by necessity (std::expected is the error channel, and it
    # is in the GPU headers), so downgrading the GPU TUs is not an option.
    # Teach CMake the flag, but only after CHECKING it: a version guess would be
    # wrong in one direction or the other, and a bad guess turns into a wall of
    # template errors instead of one clear message. Guarded on the variable
    # being absent, so a future CMake that knows CUDA23 wins.
    if(NOT CMAKE_CUDA23_STANDARD_COMPILE_OPTION)
        include(CheckSourceCompiles)
        set(CMAKE_REQUIRED_FLAGS "-std=c++23")
        check_source_compiles(CUDA
            "#include <expected>\nint main(){ std::expected<int,int> e{1}; return *e - 1; }"
            DNS_NVCC_HAS_CXX23)
        unset(CMAKE_REQUIRED_FLAGS)
        if(NOT DNS_NVCC_HAS_CXX23)
            message(FATAL_ERROR
                "nvcc ${CMAKE_CUDA_COMPILER_VERSION} cannot compile C++23 "
                "(<expected>), which dns_core's headers require. Use a newer "
                "CUDA toolkit, or build the HIP backend instead.")
        endif()
        set(CMAKE_CUDA23_STANDARD_COMPILE_OPTION "-std=c++23")
        set(CMAKE_CUDA23_EXTENSION_COMPILE_OPTION "-std=c++23")
        set(CMAKE_CUDA_STANDARD_LATEST 23)
    endif()

    add_library(dns_gpu ${DNS_GPU_SOURCES})
    # The .hip extension is not a CUDA extension; this is what routes the
    # unchanged sources through nvcc (-x cu) instead of the C++ compiler.
    set_source_files_properties(${DNS_GPU_SOURCES} PROPERTIES LANGUAGE CUDA)
    # PUBLIC: dns_main.cpp and the test_gpu_*.cpp tests are CXX TUs that include
    # gpu/device_util.hpp, so they need <hip/hip_runtime.h> to resolve too.
    target_include_directories(dns_gpu PUBLIC src/compat/cuda)
    target_link_libraries(dns_gpu
        PUBLIC dns_core CUDA::cudart
        PRIVATE CUDA::cufft)
    target_compile_definitions(dns_gpu PUBLIC DNS_WITH_GPU=1 DNS_WITH_CUDA=1
                                              DNS_GPU_BACKEND_NAME="cuda")
    # Same warnings as the HIP branch; nvcc needs them handed to the host
    # compiler explicitly.
    target_compile_options(dns_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)

    target_link_libraries(dns PRIVATE dns_gpu)
endif()

# --- dns-ui: the interactive obstacle editor / grid viewer -------------------
#
# A SECOND binary, and the only exception to the one-binary rule: it solves
# nothing. It is a pre-run inspector -- place bodies, choose a grid, and see
# the cut cells the solver would actually get -- so it links dns_core only
# (cli/config + geometry) and never dns_gpu. Everything it shows comes from the
# same sampleCut()/sanitizeCut() the runners consume; it does not reimplement
# geometry.
#
# Deps are SDL3 (window + input), Vulkan (system loader) and HarfBuzz (text
# shaping AND glyph outlines -- there is no FreeType here). SDL3 and HarfBuzz
# are fetched and built in-tree so the UI cannot drift against whatever the
# distro ships; Vulkan comes from the system, since the ICD does anyway.
option(DNS_WITH_UI "Build dns-ui (SDL3 + Vulkan + HarfBuzz)" OFF)
if(DNS_WITH_UI)
    include(FetchContent)
    find_package(Vulkan REQUIRED)

    # Shared SDL3 (its best-tested configuration; it dlopens X11/Wayland
    # itself), static HarfBuzz. BUILD_SHARED_LIBS is saved and restored so the
    # setting cannot leak into dns_core.
    set(_dns_saved_shared ${BUILD_SHARED_LIBS})
    set(SDL_SHARED ON CACHE BOOL "" FORCE)
    set(SDL_STATIC OFF CACHE BOOL "" FORCE)
    set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
    set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SDL_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(SDL3
        URL https://github.com/libsdl-org/SDL/releases/download/release-3.4.0/SDL3-3.4.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

    # HarfBuzz with every optional integration off: we want the shaper and
    # hb_font_draw_glyph(), nothing that drags in glib/icu/cairo/freetype.
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(HB_BUILD_SUBSET OFF CACHE BOOL "" FORCE)
    set(HB_BUILD_UTILS OFF CACHE BOOL "" FORCE)
    set(HB_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(HB_HAVE_FREETYPE OFF CACHE BOOL "" FORCE)
    set(HB_HAVE_GLIB OFF CACHE BOOL "" FORCE)
    set(HB_HAVE_ICU OFF CACHE BOOL "" FORCE)
    set(HB_HAVE_CORETEXT OFF CACHE BOOL "" FORCE)
    set(HB_HAVE_GDI OFF CACHE BOOL "" FORCE)
    set(HB_HAVE_DIRECTWRITE OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(harfbuzz
        URL https://github.com/harfbuzz/harfbuzz/archive/refs/tags/14.2.1.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

    FetchContent_MakeAvailable(SDL3 harfbuzz)
    set(BUILD_SHARED_LIBS ${_dns_saved_shared} CACHE BOOL "" FORCE)

    # GLSL -> SPIR-V, embedded as a C array header at build time. No runtime
    # shader files to lose, and a shader syntax error is a BUILD error.
    find_program(DNS_GLSLC NAMES glslc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)
    set(_dns_spv_dir ${CMAKE_CURRENT_BINARY_DIR}/ui_shaders)
    file(MAKE_DIRECTORY ${_dns_spv_dir})
    set(DNS_UI_SPV "")
    foreach(s scene3d.vert scene3d.frag line3d.vert line3d.frag
              ui2d.vert ui2d.frag)
        string(REPLACE "." "_" _sym ${s})
        set(_out ${_dns_spv_dir}/${_sym}.spv.h)
        add_custom_command(OUTPUT ${_out}
            COMMAND ${DNS_GLSLC} -O --target-env=vulkan1.1 -mfmt=c
                    -o ${_out} ${CMAKE_CURRENT_SOURCE_DIR}/src/ui/shaders/${s}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/ui/shaders/${s}
            COMMENT "glslc ${s}"
            VERBATIM)
        list(APPEND DNS_UI_SPV ${_out})
    endforeach()
    add_custom_target(dns_ui_shaders DEPENDS ${DNS_UI_SPV})

    add_executable(dns-ui
        apps/dns_ui.cpp
        src/ui/app.cpp
        src/ui/doc.cpp
        src/ui/field.cpp
        src/ui/livemesh.cpp
        src/ui/pickobj.cpp
        src/ui/draw.cpp
        src/ui/gizmo.cpp
        src/ui/gui.cpp
        src/ui/guicheck.cpp
        src/ui/runner.cpp
        src/ui/text.cpp
        src/ui/view.cpp
        src/ui/vk.cpp
    )
    add_dependencies(dns-ui dns_ui_shaders)
    target_include_directories(dns-ui PRIVATE src ${_dns_spv_dir})
    # --check opens every shipped scene, so it needs to find them.
    target_compile_definitions(dns-ui PRIVATE
        DNS_EXAMPLES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/examples")
    target_link_libraries(dns-ui PRIVATE dns_core SDL3::SDL3 Vulkan::Vulkan harfbuzz)

    # `dns-ui --check` drives the whole non-graphical half (document build,
    # validator wiring, cut census, view geometry) over a small case matrix.
    # No window, no GPU, no font -- so it runs headless like every other test.
    add_test(NAME ui_check COMMAND dns-ui --check)
    set_tests_properties(ui_check PROPERTIES TIMEOUT 300)
endif()

include(CTest)
if(BUILD_TESTING)
    set(DNS_TESTS cli geometry output poisson projection tgv tgv3d cutcell cutcell_sphere comp comp_curv meshsdf
        fieldpng webframes channel_config conv_config scene sgs_op framemap ogrid transport
        rotor focal cli_defaults comp_axis_permute rans rans_channel1d
        steady_stop blockband channel_line pseudo_time flow_rans_config precond
        host_threads flame1d flame_config bunsen diffusion candle wick chamber_config
        polymesh facemesh fvref la amg fvcase fvwalldist tfles)
    if(DNS_WITH_FFTW)
        list(APPEND DNS_TESTS ke_conservation hit_mini turb_ic)
    endif()
    foreach(t IN LISTS DNS_TESTS)
        add_executable(test_${t} tests/test_${t}.cpp)
        target_link_libraries(test_${t} PRIVATE dns_core)
        add_test(NAME ${t} COMMAND test_${t})
        set_tests_properties(${t} PROPERTIES TIMEOUT 120)
    endforeach()
    target_link_libraries(test_fieldpng PRIVATE ZLIB::ZLIB) # decodes IDAT
    target_compile_definitions(test_host_threads PRIVATE
        DNS_ROTOR2_STL="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/rotor2.stl")
    # The Cantera reference of the chemistry oracle (tools/chem/cantera_flame.py).
    target_compile_definitions(test_flame1d PRIVATE
        DNS_CHEM_DATA="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/chem")
    # Charlette's efficiency against tools/chem/tfles_eff.py (docs/chem.md).
    target_compile_definitions(test_tfles PRIVATE
        DNS_CHEM_DATA="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/chem")
    # The 3D flame renderer's self-test (tools/chem/turb_render.py, numpy):
    # optically thin blob total = integral j dV at any orbit angle, the z
    # seam, byte determinism, the chemiluminescence colour. Host, seconds.
    find_program(DNS_PYTHON3 python3)
    if(DNS_PYTHON3)
        add_test(NAME turb_render_selftest COMMAND ${DNS_PYTHON3}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/chem/turb_render.py --selftest)
        set_tests_properties(turb_render_selftest PROPERTIES TIMEOUT 120)
    endif()
    # FaceMesh (docs/mesh.md): the checked-in polyMesh fixtures and their
    # checkMesh logs (tools/mesh/make_test_meshes.sh). facemesh_mb is the
    # 831k-cell snappy mesh, LOCAL only (runs/ is git-ignored): rc 77 when
    # absent.
    set(DNS_MESH_DATA ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/mesh)
    foreach(t polymesh facemesh fvref la amg)
        target_compile_definitions(test_${t} PRIVATE DNS_MESH_DATA="${DNS_MESH_DATA}")
    endforeach()
    target_compile_definitions(test_facemesh PRIVATE
        DNS_LOCAL_MESH_MB="${CMAKE_SOURCE_DIR}/runs/foam_mesh/mB")
    add_test(NAME facemesh_mb COMMAND test_facemesh mb)
    set_tests_properties(facemesh_mb PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    set_tests_properties(polymesh PROPERTIES TIMEOUT 60)
    # FV oracle (docs/mesh.md): G1-G3, G5-G7 + extras in `fvref`; the G4 MMS
    # march (3 nested meshes, threaded) on its own so ctest runs them apart.
    add_test(NAME fvref_mms COMMAND test_fvref mms)
    set_tests_properties(fvref_mms PROPERTIES TIMEOUT 300)
    # AMG setup gate (docs/la.md, masterplan_meshsolver P5 "Setup"): the host
    # hierarchy of a 6M-cell 7-point graph <= 3 s, two setups identical.
    add_test(NAME amg_setup COMMAND test_amg setup)
    set_tests_properties(amg_setup PROPERTIES TIMEOUT 120)
    # Incompressible FV path (docs/fv.md): the OpenFOAM case layer and the
    # simpleFoam parity references (tools/fv/make_test_data.sh).
    set(DNS_FV_DATA ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/fv)
    target_compile_definitions(test_fvcase PRIVATE DNS_FV_DATA="${DNS_FV_DATA}")
    target_compile_definitions(test_fvwalldist PRIVATE DNS_FV_DATA="${DNS_FV_DATA}")
    # The layered rotor2 L1 snappy mesh (tools/fv/make_rotor_bl.py, 123k cells,
    # runs/ is git-ignored: rc 77 when absent): meshWave vs simpleFoam's serial
    # wallDistance (checkMesh -writeFields, 12 digits).
    add_test(NAME fvwalldist_rotor COMMAND test_fvwalldist
        ${CMAKE_SOURCE_DIR}/runs/fv/bl/rotor2_L1_sst_np1 ${CMAKE_SOURCE_DIR}/runs/fv/bl/rotor2_L1_sst_np1/0/wallDistance)
    set_tests_properties(fvwalldist_rotor PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)

    # Job server: pure parts (sha256, json, spec, admission) and a live run of
    # the real simjobd + simjob binaries (skips without a systemd user session).
    add_executable(test_jobs tests/test_jobs.cpp)
    target_link_libraries(test_jobs PRIVATE simjobs)
    add_test(NAME jobs COMMAND test_jobs)
    set_tests_properties(jobs PROPERTIES TIMEOUT 30 LABELS fast)
    add_executable(jobs_hog tests/jobs_hog.cpp)
    add_executable(test_jobs_live tests/test_jobs_live.cpp)
    target_link_libraries(test_jobs_live PRIVATE simjobs)
    target_compile_definitions(test_jobs_live PRIVATE
        SIMJOBD_BIN="$<TARGET_FILE:simjobd>" SIMJOB_BIN="$<TARGET_FILE:simjob>"
        JOBS_HOG_BIN="$<TARGET_FILE:jobs_hog>")
    add_test(NAME jobs_live COMMAND test_jobs_live)
    set_tests_properties(jobs_live PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    set_tests_properties(cli_defaults comp_axis_permute blockband channel_line
                         pseudo_time PROPERTIES TIMEOUT 60)

    # The exit-code contract, driven through the real binary. GPU-free: every
    # case is reachable in phases 1-3a.
    add_executable(test_errors tests/test_errors.cpp)
    target_link_libraries(test_errors PRIVATE dns_core)
    target_compile_definitions(test_errors PRIVATE DNS_BIN="$<TARGET_FILE:dns>"
                               DNS_MESH_DATA="${DNS_MESH_DATA}" DNS_FV_DATA="${DNS_FV_DATA}")
    add_dependencies(test_errors dns)
    add_test(NAME errors COMMAND test_errors)
    set_tests_properties(errors PROPERTIES TIMEOUT 120)

    # dns prop: library level (closed forms, Python equivalence fixtures, CSV
    # and CLI rejections) and through the binary (exit codes, CSV). Host-only:
    # neither skips without a GPU.
    set(DNS_PROP_DATA ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/prop)
    add_executable(test_prop tests/test_prop.cpp)
    target_link_libraries(test_prop PRIVATE dns_prop)
    target_compile_definitions(test_prop PRIVATE DNS_PROP_DATA="${DNS_PROP_DATA}")
    add_test(NAME prop COMMAND test_prop)
    set_tests_properties(prop PROPERTIES TIMEOUT 300)
    add_executable(test_prop_cli tests/test_prop_cli.cpp)
    target_compile_definitions(test_prop_cli PRIVATE DNS_BIN="$<TARGET_FILE:dns>"
                               DNS_PROP_DATA="${DNS_PROP_DATA}")
    add_dependencies(test_prop_cli dns)
    add_test(NAME prop_cli COMMAND test_prop_cli)
    set_tests_properties(prop_cli PROPERTIES TIMEOUT 300)
    if(DNS_WITH_FFTW)
        # 3 + 3 eddy turnovers, ~22 s unloaded (gpu_hit_mini runs 6 + 6).
        set_tests_properties(hit_mini PROPERTIES TIMEOUT 180
            LABELS "slow;validation;literature")
    endif()
    if(DNS_WITH_GPU)
        # Every test registered in this block drives the device: they all take
        # RESOURCE_LOCK gpu (set at the block's end), so `ctest -j` overlaps
        # host tests with ONE GPU test at a time instead of contending.
        get_property(dns_tests_before_gpu DIRECTORY PROPERTY TESTS)
        foreach(t gpu_buffer gpu_operators gpu_reductions gpu_poisson
                  gpu_poisson_neumann gpu_poisson2d gpu_projection gpu_tgv
                  gpu_cutcell gpu_cutcell2d gpu_cutcell_sphere gpu_conservation
                  gpu_energy_stability gpu_ghostfill gpu_comp gpu_comp_curv
                  gpu_comp_ghost gpu_comp_ghost_focal gpu_comp_les
                  gpu_comp_ghost3d gpu_comp_mesh gpu_comp_ghost_closure
                  gpu_comp_far gpu_comp_nscbc gpu_nscbc_absorb gpu_nscbc_order
                  gpu_acoustics gpu_comp_sink
                  gpu_les_smag gpu_les_wall gpu_derived
                  gpu_poisson_wally gpu_rotor_wall gpu_rotor_table gpu_mrf
                  gpu_channel gpu_cutflux gpu_ogrid gpu_rans gpu_rans_mask
                  gpu_rans_ogrid gpu_rans_mms gpu_axisym gpu_axisym3d gpu_nk
                  gpu_rans_immersed
                  index_offsets_wrap stats_cadence
                  gpu_tally_couette_torque gpu_tally_conduction
                  gpu_tally_archimedes gpu_tally_kutta_joukowski
                  gpu_mrf_conservation gpu_mrf_stabledt gpu_ogrid_rotation
                  gpu_rb_modes
                  gpu_comp_ghost_closure_focal gpu_comp_ghost_couette_exact
                  gpu_comp_ghost_twoside gpu_comp_ghost_imperm
                  gpu_rotor_comoving_fixed_point gpu_comp_scale_invariance
                  gpu_ghost_translate gpu_rotor_swap_ledger
                  gpu_comp_axis_permute gpu_comp_curv_axis_permute
                  comp_time_order gpu_comp_viscous_modes
                  gpu_flame_passive gpu_flame_conservation)
            add_executable(test_${t} tests/test_${t}.cpp)
            target_link_libraries(test_${t} PRIVATE dns_gpu)
            add_test(NAME ${t} COMMAND test_${t})
            set_tests_properties(${t} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
        endforeach()
        # Small verification tests (seconds each): a hang is a failure.
        set_tests_properties(index_offsets_wrap stats_cadence
            gpu_tally_couette_torque gpu_tally_conduction gpu_tally_archimedes
            gpu_tally_kutta_joukowski gpu_mrf_conservation gpu_mrf_stabledt
            gpu_ogrid_rotation gpu_rb_modes gpu_comp_ghost_closure_focal
            gpu_comp_ghost_couette_exact gpu_comp_ghost_twoside
            gpu_comp_ghost_imperm gpu_rotor_comoving_fixed_point
            gpu_comp_scale_invariance gpu_ghost_translate gpu_comp_axis_permute
            gpu_comp_curv_axis_permute comp_time_order gpu_comp_viscous_modes
            gpu_rans_mask gpu_rans_ogrid gpu_rans_mms gpu_axisym gpu_axisym3d
            gpu_rans_immersed gpu_les_wall gpu_flame_passive
            gpu_flame_conservation
            PROPERTIES TIMEOUT 60)
        set_tests_properties(gpu_nk gpu_rotor_swap_ledger PROPERTIES TIMEOUT 120)
        # ~56 s alone (G10's 2.2e5-iteration compensated march, G9's 1.2e5
        # plain one). Its SST half is its own entry, ~32 s (S-G9's 2.2e5).
        # The 273 s once measured under ctest -j4 was GPU contention, which
        # RESOURCE_LOCK gpu now rules out.
        add_test(NAME gpu_rans_sst COMMAND test_gpu_rans sst)
        set_tests_properties(gpu_rans_sst PROPERTIES SKIP_RETURN_CODE 77)
        set_tests_properties(gpu_rans gpu_rans_sst PROPERTIES TIMEOUT 180)
        # gpu_mrf's bladed half: the BUILT binary on the rotor2 L1 grid for
        # 1 D/U, ct/cq pinned at 1%, deep-solid core asserted (W7; ~11 s).
        target_compile_definitions(test_gpu_mrf PRIVATE
            DNS_BIN="$<TARGET_FILE:dns>"
            DNS_ROTOR2_STL="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/rotor2.stl")
        add_dependencies(test_gpu_mrf dns)
        add_test(NAME gpu_mrf_rotor2 COMMAND test_gpu_mrf rotor2)
        set_tests_properties(gpu_mrf_rotor2 PROPERTIES SKIP_RETURN_CODE 77
            TIMEOUT 60)
        # The frozen rotor's steady march (masterplan_levers M1): rotor2 L1
        # --steady --lts-max 1 to its 10000 cap (T4, ~45 s), and the random-
        # lts fixed point of the runner's hook set on a 2D rig (T5).
        add_test(NAME gpu_mrf_steady COMMAND test_gpu_mrf steady)
        add_test(NAME gpu_mrf_steady_lts COMMAND test_gpu_mrf steadylts)
        # --precond (masterplan_levers M2 G3/G4): the O-grid cylinder's cd
        # unchanged by the preconditioner, M 0.05 K 1 stable (~13 s). The
        # random-lts fixed point with it on rides gpu_mrf_steady_lts (G2).
        add_test(NAME gpu_precond COMMAND test_gpu_mrf precond)
        set_tests_properties(gpu_mrf_steady gpu_mrf_steady_lts gpu_precond
            PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
        # --thin-seal (analysis_propperf S1/S2): `host` is the classification
        # alone (sealed count vs segment clipping, none on resolved bodies,
        # thread-count identity; no device); `gpu` the sealed membranes
        # holding a pressure jump and their chamber masses (~seconds).
        add_executable(test_gpu_thin_seal tests/test_gpu_thin_seal.cpp)
        target_link_libraries(test_gpu_thin_seal PRIVATE dns_gpu)
        add_test(NAME thin_seal COMMAND test_gpu_thin_seal host)
        add_test(NAME gpu_thin_seal COMMAND test_gpu_thin_seal gpu)
        set_tests_properties(thin_seal gpu_thin_seal
            PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
        # The incompressible 3D SGS wall tally against the eddy-diffusion
        # substep's own momentum drain (W1; ~1 s).
        add_test(NAME gpu_les_tally3d COMMAND test_gpu_les_smag tally3d)
        set_tests_properties(gpu_les_tally3d PROPERTIES SKIP_RETURN_CODE 77
            TIMEOUT 60)
        # Compressible TGV 32^3 to t = 12, DNS / smag / sigma, against the
        # 512^3 reference (W10).
        add_executable(test_gpu_les_tgv_comp tests/test_gpu_les_tgv_comp.cpp)
        target_link_libraries(test_gpu_les_tgv_comp PRIVATE dns_gpu)
        target_compile_definitions(test_gpu_les_tgv_comp PRIVATE
            DNS_VALIDATION_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/validation")
        add_test(NAME gpu_les_tgv_comp COMMAND test_gpu_les_tgv_comp)
        set_tests_properties(gpu_les_tgv_comp PROPERTIES SKIP_RETURN_CODE 77
            TIMEOUT 60)
        # gpu/la (docs/la.md): LA1 kernels vs the host FP64 restatement on
        # structured and polyhedral graphs, LA2 Krylov residual + determinism,
        # LA-perf (SpMV, batched lines; meaningful under simjob --bench).
        add_executable(test_gpu_la tests/test_gpu_la.cpp)
        target_link_libraries(test_gpu_la PRIVATE dns_gpu)
        target_compile_definitions(test_gpu_la PRIVATE DNS_MESH_DATA="${DNS_MESH_DATA}")
        add_test(NAME gpu_la1 COMMAND test_gpu_la la1)
        add_test(NAME gpu_la2 COMMAND test_gpu_la la2)
        add_test(NAME gpu_la_perf COMMAND test_gpu_la perf)
        set_tests_properties(gpu_la1 gpu_la2 gpu_la_perf PROPERTIES SKIP_RETURN_CODE 77
            TIMEOUT 180)
        # gpu/la AMG (docs/la.md, masterplan_meshsolver P5): refresh vs the host
        # Galerkin restatement on every shape + determinism; LA3 h-independence
        # and anisotropy; LA4 AMG-PCG vs the FFT-PCG cut-cell solve; AMG-PCG on
        # the checked-in polyhedral meshes (OpenFOAM's Laplacian coefficients).
        add_executable(test_gpu_amg tests/test_gpu_amg.cpp)
        target_link_libraries(test_gpu_amg PRIVATE dns_gpu)
        target_compile_definitions(test_gpu_amg PRIVATE DNS_MESH_DATA="${DNS_MESH_DATA}")
        # fp32: the FV path's p solve in float at the FP32 floor (docs/fv.md "Precision").
        foreach(m refresh la3 la4 meshes fp32)
            add_test(NAME gpu_amg_${m} COMMAND test_gpu_amg ${m})
        endforeach()
        set_tests_properties(gpu_amg_refresh gpu_amg_la3 gpu_amg_la4 gpu_amg_meshes gpu_amg_fp32
            PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
        # Incompressible FV path vs OpenFOAM simpleFoam on the identical
        # blockMesh (docs/fv.md, gate IF1): cavity, skewed cavity, channel.
        add_executable(test_gpu_fv_simple tests/test_gpu_fv_simple.cpp)
        target_link_libraries(test_gpu_fv_simple PRIVATE dns_gpu)
        target_compile_definitions(test_gpu_fv_simple PRIVATE DNS_FV_DATA="${DNS_FV_DATA}")
        add_test(NAME gpu_fv_simple COMMAND test_gpu_fv_simple)
        set_tests_properties(gpu_fv_simple PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
        # gpu/la performance kernels vs the pre-change kernels, kept verbatim in
        # the test as oracles (masterplan_performance): memcmp, float + double.
        # A HIP TU (it defines kernels), routed like the library under CUDA.
        add_executable(test_gpu_la_ref tests/test_gpu_la_ref.hip)
        if(DNS_WITH_CUDA)
            set_source_files_properties(tests/test_gpu_la_ref.hip PROPERTIES LANGUAGE CUDA)
        endif()
        target_link_libraries(test_gpu_la_ref PRIVATE dns_gpu)
        target_compile_definitions(test_gpu_la_ref PRIVATE DNS_MESH_DATA="${DNS_MESH_DATA}")
        add_test(NAME gpu_la_ref COMMAND test_gpu_la_ref)
        set_tests_properties(gpu_la_ref PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
        # The RANS / LES / MRF tier caps at 180 s (the val tier keeps its own).
        set_tests_properties(gpu_tgv gpu_comp_les gpu_les_smag gpu_rotor_wall
            gpu_rotor_table gpu_mrf gpu_channel PROPERTIES TIMEOUT 180)

        # These define their own kernels (the pre-recomposition kernel variants,
        # as identity oracles), so they are HIP TUs rather than plain .cpp --
        # and so they need the same .hip -> CUDA routing as the library.
        foreach(t gpu_tunnel_bc gpu_bodyforce)
            add_executable(test_${t} tests/test_${t}.hip)
            if(DNS_WITH_CUDA)
                set_source_files_properties(tests/test_${t}.hip
                    PROPERTIES LANGUAGE CUDA)
            endif()
            target_link_libraries(test_${t} PRIVATE dns_gpu)
            add_test(NAME ${t} COMMAND test_${t})
            set_tests_properties(${t} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
        endforeach()

        # A GPU test that drives the BUILT `dns` binary (handed in as DNS_BIN)
        # and skips (77) without a device.
        function(dns_gpu_bin_test name timeout)
            add_executable(test_${name} tests/test_${name}.cpp)
            target_link_libraries(test_${name} PRIVATE dns_gpu)
            target_compile_definitions(test_${name} PRIVATE
                DNS_BIN="$<TARGET_FILE:dns>")
            add_dependencies(test_${name} dns)
            add_test(NAME ${name} COMMAND test_${name})
            set_tests_properties(${name} PROPERTIES TIMEOUT ${timeout}
                SKIP_RETURN_CODE 77)
        endfunction()

        # dns conv is an app composition (IC + sink + runner), so it is driven
        # through the binary and read back from its stats CSV.
        dns_gpu_bin_test(gpu_conv 300)
        # The Boussinesq branch, including the cross-validation against the
        # compressible one -- also driven through the binary.
        dns_gpu_bin_test(gpu_rb 300)
        # Open far-field path is app-local; drive the built binary.
        dns_gpu_bin_test(gpu_open_incomp 600)
        # Smoke coverage for the three runners, incl. --far tunnel WITH walls
        # (the incompressible default) and runner_comp, which had none (~30 s).
        dns_gpu_bin_test(gpu_runners 180)
        # SST small cases: the freestream decay ODE and the L1 sphere-wedge
        # Re 1e3 certification against OpenFOAM on the identical mesh (~35 s).
        dns_gpu_bin_test(gpu_sst 180)
        # Its SA-neg twin: the same wedge certified, pinned to its measured
        # state; the OpenFOAM comparison is printed, not gated (~30 s).
        dns_gpu_bin_test(gpu_sa_wedge 120)
        # The sphere's full revolution end to end: N 36 == the 10-deg wedge,
        # and the m = 1 bifurcation at Re 250 with --ogrid-trip (~40 s).
        dns_gpu_bin_test(gpu_axisym3d_run 180)
        # GPU twin of hit_mini: dns hit 64^3, 6 + 6 eddy turnovers (~2 s).
        if(DNS_WITH_FFTW)
            dns_gpu_bin_test(gpu_hit_mini 60)
        endif()
        # Sponge-free open tunnel: library-level freestream cycles + a short
        # end-to-end run of the built binary.
        dns_gpu_bin_test(gpu_tunnel_open 600)
        # The rotor tip-Mach guard: library replay of the runner's measurement,
        # then the binary's exit 2 (the binary half skips without a device).
        dns_gpu_bin_test(rotor_tip_guard 60)
        # Every runner twice: the stats CSVs must be byte-identical.
        dns_gpu_bin_test(gpu_determinism 120)
        # Linear Rayleigh-Benard onset (Ra_c 1708) on both conv branches.
        dns_gpu_bin_test(gpu_rb_onset 60)
        # The implicit steady RANS march: in-process device checks against the
        # host line operator, and the runner through the binary (I4/I5/I7).
        dns_gpu_bin_test(gpu_rans_implicit 60)
        # Channel LES (W11): the odd mu_t wall image, the label, sigma == 0 on
        # the laminar state, --les 0 byte-identical (~2 s).
        dns_gpu_bin_test(gpu_channel_les 60)
        # dns flame (docs/chem.md): the 1D premixed flame against its FP64
        # oracle -- grid convergence (3 runs), the Mach ladder + dt check
        # (4 runs), Sutherland (1 run). Each run < 1 min.
        add_executable(test_gpu_flame1d tests/test_gpu_flame1d.cpp)
        target_link_libraries(test_gpu_flame1d PRIVATE dns_gpu)
        target_compile_definitions(test_gpu_flame1d PRIVATE
            DNS_BIN="$<TARGET_FILE:dns>")
        add_dependencies(test_gpu_flame1d dns)
        add_test(NAME gpu_flame1d_grid COMMAND test_gpu_flame1d grid)
        add_test(NAME gpu_flame1d_mach COMMAND test_gpu_flame1d mach)
        add_test(NAME gpu_flame1d_suth COMMAND test_gpu_flame1d suth)
        # The planar flame at nz 16 against nz 1 (docs/chem.md, 3D; ~20 s).
        add_test(NAME gpu_flame3d_planar COMMAND test_gpu_flame1d 3d)
        set_tests_properties(gpu_flame1d_grid gpu_flame1d_mach gpu_flame1d_suth
            gpu_flame3d_planar PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
        # dns flame 3D (docs/chem.md, "3D and turbulence"): the z-extruded 2D
        # flame bitwise per k plane at a fixed dt (extrude); the turbulent
        # inflow instantiation, zero rows bit for bit, the row lerp bitwise,
        # rows relaxed onto the inflow row (inflow). Seconds each.
        add_executable(test_gpu_flame_turb tests/test_gpu_flame_turb.cpp)
        target_link_libraries(test_gpu_flame_turb PRIVATE dns_gpu)
        add_test(NAME gpu_flame3d_extrude COMMAND test_gpu_flame_turb extrude)
        add_test(NAME gpu_flame_turb_inflow COMMAND test_gpu_flame_turb inflow)
        set_tests_properties(gpu_flame3d_extrude gpu_flame_turb_inflow
            PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
        # dns flame LES species + TFLES (docs/chem.md, "LES species and
        # TFLES"): --tfles 1 bit for bit, charlette == none on a planar
        # flame (bits, binary); the sensor's coverage and the max filter
        # (sensor); the species SGS flux exact / conservative, the open-row
        # mu_t image (les_species); the Shvab-Zeldovich teeth (teeth); the
        # laminar TFLES flame speed, 1D and 3D LES (1d, binary, ~3 min).
        add_executable(test_gpu_tfles tests/test_gpu_tfles.cpp)
        target_link_libraries(test_gpu_tfles PRIVATE dns_gpu)
        target_compile_definitions(test_gpu_tfles PRIVATE
            DNS_BIN="$<TARGET_FILE:dns>")
        add_dependencies(test_gpu_tfles dns)
        add_test(NAME gpu_tfles_bits COMMAND test_gpu_tfles bits)
        add_test(NAME gpu_tfles_sensor COMMAND test_gpu_tfles sensor)
        add_test(NAME gpu_flame_les_species COMMAND test_gpu_tfles species)
        add_test(NAME gpu_tfles_teeth COMMAND test_gpu_tfles teeth)
        add_test(NAME gpu_tfles_1d COMMAND test_gpu_tfles lam)
        set_tests_properties(gpu_tfles_bits gpu_tfles_sensor
            gpu_flame_les_species gpu_tfles_teeth
            PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
        set_tests_properties(gpu_tfles_1d PROPERTIES TIMEOUT 600
            SKIP_RETURN_CODE 77)
        # dns flame --mode bunsen (docs/chem.md): the cone at U_jet 3 and 5
        # (8 cells, ~20 s each), the anchoring guard (cold coflow exits 8,
        # the fast burnt strip anchors), the per-row inflow bit for bit.
        add_executable(test_gpu_bunsen tests/test_gpu_bunsen.cpp)
        target_link_libraries(test_gpu_bunsen PRIVATE dns_gpu)
        target_compile_definitions(test_gpu_bunsen PRIVATE
            DNS_BIN="$<TARGET_FILE:dns>")
        add_dependencies(test_gpu_bunsen dns)
        add_test(NAME gpu_bunsen_cone COMMAND test_gpu_bunsen cone)
        add_test(NAME gpu_bunsen_anchor COMMAND test_gpu_bunsen anchor)
        # 3D slot Bunsen (docs/chem.md, "3D turbulent slot Bunsen"): the
        # burner extruded to nz 8 equals the 2D run byte for byte (frames,
        # span-mean cone), DNS and TFLES + LES sigma. Seconds.
        add_test(NAME gpu_bunsen3d_extrude COMMAND test_gpu_bunsen extrude3d)
        set_tests_properties(gpu_bunsen_cone gpu_bunsen_anchor
            gpu_bunsen3d_extrude
            PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
        # dns flame --mode diffusion (docs/chem.md): the two-reactant step on
        # an analytic diffusion layer -- the coupling function to rounding,
        # bounds, the Burke-Schumann sheet position and flux (mix), the flame
        # temperature against T_ad (heat); and the jet through the binary.
        add_executable(test_gpu_diffusion_layer
            tests/test_gpu_diffusion_layer.cpp)
        target_link_libraries(test_gpu_diffusion_layer PRIVATE dns_gpu)
        add_test(NAME gpu_diffusion_mix COMMAND test_gpu_diffusion_layer mix)
        add_test(NAME gpu_diffusion_heat COMMAND test_gpu_diffusion_layer heat)
        set_tests_properties(gpu_diffusion_mix gpu_diffusion_heat
            PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77)
        add_executable(test_gpu_diffusion_jet tests/test_gpu_diffusion_jet.cpp)
        target_link_libraries(test_gpu_diffusion_jet PRIVATE dns_gpu)
        target_compile_definitions(test_gpu_diffusion_jet PRIVATE
            DNS_BIN="$<TARGET_FILE:dns>")
        add_dependencies(test_gpu_diffusion_jet dns)
        add_test(NAME gpu_diffusion_jet COMMAND test_gpu_diffusion_jet jet)
        add_test(NAME gpu_diffusion_guard COMMAND test_gpu_diffusion_jet guard)
        set_tests_properties(gpu_diffusion_jet gpu_diffusion_guard
            PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77)
        # dns flame --mode candle (docs/chem.md): the diffusion flame under
        # gravity -- the quiescent hydrostatic box stays at rest (rest), g = 0
        # is the diffusion jet bit for bit (g0), the flame height vs the fuel
        # flow rate against Roper's buoyant slot law (height).
        add_executable(test_gpu_candle tests/test_gpu_candle.cpp)
        target_link_libraries(test_gpu_candle PRIVATE dns_gpu)
        target_compile_definitions(test_gpu_candle PRIVATE
            DNS_BIN="$<TARGET_FILE:dns>")
        add_dependencies(test_gpu_candle dns)
        add_test(NAME gpu_candle_rest COMMAND test_gpu_candle rest)
        add_test(NAME gpu_candle_g0 COMMAND test_gpu_candle g0)
        add_test(NAME gpu_candle_height COMMAND test_gpu_candle height)
        set_tests_properties(gpu_candle_rest gpu_candle_g0 gpu_candle_height
            PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)
        # --mode chamber (docs/chem.md, "Chamber"): the closed 2D combustion
        # chamber -- inert wall impermeability (fuel = injected to rounding),
        # then the reacting fuel/energy balance, the nozzle against the
        # isentropic estimate, and anchoring. Two runs, ~2 min.
        dns_gpu_bin_test(gpu_chamber 600)
        # --mode candle --wick (docs/chem.md, "Wick candle"): the immersed
        # wick + candle body, fuel from the wick surface, soot. inert: the
        # walls leak no fuel (~5 s); flame: anchored, fuel and soot balances,
        # soot burned out at the tip, the wick heated (~30 s).
        add_executable(test_gpu_wick tests/test_gpu_wick.cpp)
        target_link_libraries(test_gpu_wick PRIVATE dns_gpu)
        target_compile_definitions(test_gpu_wick PRIVATE
            DNS_BIN="$<TARGET_FILE:dns>")
        add_dependencies(test_gpu_wick dns)
        add_test(NAME gpu_wick_inert COMMAND test_gpu_wick inert)
        add_test(NAME gpu_wick_flame COMMAND test_gpu_wick flame)
        set_tests_properties(gpu_wick_inert gpu_wick_flame
            PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)


        # --- Physical validation suite (docs/validation.md) -----------------
        # Tier A registers always; Tier B (heavy) behind DNS_HEAVY_VALIDATION.
        # Labels: validation + truth class (analytic/literature/anchor) + heavy.
        option(DNS_HEAVY_VALIDATION "Register the heavy (Tier B) validation tests" OFF)
        option(DNS_VALIDATION_BASELINE
            "Register validation tests whose bands predate the 2026-07-19 recomposition" OFF)
        set(DNS_VAL_SCRATCH ${CMAKE_CURRENT_BINARY_DIR}/valtmp)
        file(MAKE_DIRECTORY ${DNS_VAL_SCRATCH})
        set(DNS_VAL_DEFS
            DNS_VAL_SCRATCH="${DNS_VAL_SCRATCH}"
            DNS_ANCHORS_CSV="${CMAKE_CURRENT_SOURCE_DIR}/data/validation/anchors.csv")

        # THE QUARANTINE, in one place.
        #
        # Nine literature-band tests have bands measured BEFORE the 2026-07-19
        # recomposition (the kBcTunnel corner unification, the 3-direction
        # friction tally, the FP64 force tally). They are a DECISION, not a
        # failure: re-measure per docs/validation.md Tier C before citing one.
        #
        # Their executables are built unconditionally even when zero of them are
        # registered -- that is deliberate, and keeps the quarantined sources
        # compiling against the current library instead of rotting.
        #
        # dns_add_val(<name>
        #     [TARGET t]        executable, default test_<name>
        #     [ARG a]           argument appended to the test command
        #     [TIMEOUT n]       default 300
        #     [LABELS "a;b"]    ctest labels, "validation;" prefix included
        #     [QUARANTINE]      register only with DNS_VALIDATION_BASELINE=ON
        #     [SERIAL])         RUN_SERIAL (the heavy tier)
        function(dns_add_val name)
            # LABELS is a MULTI-value keyword, not a one-value one: ${ARGN} expands
            # unquoted, so a "validation;anchor" string arrives already split
            # into two arguments and a one-value slot would keep only the first.
            cmake_parse_arguments(V "QUARANTINE;SERIAL" "TARGET;ARG;TIMEOUT" "LABELS" ${ARGN})
            if(NOT V_TARGET)
                set(V_TARGET test_${name})
            endif()
            if(NOT V_TIMEOUT)
                set(V_TIMEOUT 300)
            endif()
            if(V_QUARANTINE AND NOT DNS_VALIDATION_BASELINE)
                return()
            endif()
            add_test(NAME ${name} COMMAND ${V_TARGET} ${V_ARG})
            # NB: set the properties DIRECTLY. Collecting them into a variable
            # and expanding ${props} flattens the semicolon-separated LABELS
            # string into separate arguments, so every test silently kept only
            # its first label ("validation") and lost its truth class.
            set_tests_properties(${name} PROPERTIES
                TIMEOUT ${V_TIMEOUT} SKIP_RETURN_CODE 77 LABELS "${V_LABELS}")
            if(V_SERIAL)
                set_tests_properties(${name} PROPERTIES RUN_SERIAL TRUE)
            endif()
        endfunction()

        # Builds a validation executable (always, see above); registration is
        # dns_add_val's job.
        function(dns_val_exe name)
            cmake_parse_arguments(E "" "SOURCE" "DEFS" ${ARGN})
            if(NOT E_SOURCE)
                set(E_SOURCE tests/test_${name}.cpp)
            endif()
            add_executable(test_${name} ${E_SOURCE})
            target_link_libraries(test_${name} PRIVATE dns_gpu)
            target_compile_definitions(test_${name} PRIVATE ${DNS_VAL_DEFS}
                DNS_BIN="$<TARGET_FILE:dns>" ${E_DEFS})
            add_dependencies(test_${name} dns)
        endfunction()

        add_executable(test_val_spindown tests/test_val_spindown.cpp)
        target_link_libraries(test_val_spindown PRIVATE dns_gpu)
        dns_add_val(val_spindown TARGET test_val_spindown TIMEOUT 120
            LABELS "validation;analytic")

        foreach(t val_cyl_re60 val_cyl_re100 val_cyl_re200 val_cyl_blockage
                  val_comp_cyl val_sphere_re300 val_mesh_sphere
                  val_comp_mesh_sphere val_ogrid_cyl)
            dns_val_exe(${t})
        endforeach()
        dns_val_exe(val_tgv3d DEFS
            DNS_VALIDATION_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/validation"
            DNS_SPECTRAL_REF="${CMAKE_CURRENT_SOURCE_DIR}/data/spectral_Re1600_512.gdiag")

        dns_add_val(val_cyl_re60  QUARANTINE LABELS "validation;anchor;literature")
        dns_add_val(val_cyl_re100 QUARANTINE LABELS "validation;literature")
        dns_add_val(val_tgv3d_64  QUARANTINE TARGET test_val_tgv3d ARG 64
            LABELS "validation;anchor;literature")
        dns_add_val(val_comp_cyl_re60 TARGET test_val_comp_cyl ARG re60
            LABELS "validation;anchor")
        dns_add_val(val_comp_mesh_sphere QUARANTINE TIMEOUT 600
            LABELS "validation;analytic")
        # Default tier (not quarantined): the body-fitted O-grid against the
        # 2D literature, Re 100 shedding + Re 40 steady, ~28 s (W12).
        dns_add_val(val_ogrid_cyl TIMEOUT 120 LABELS "validation;literature")

        if(DNS_HEAVY_VALIDATION)
            dns_add_val(val_cyl_re200 QUARANTINE SERIAL TIMEOUT 900
                LABELS "validation;literature;heavy")
            dns_add_val(val_cyl_blockage QUARANTINE SERIAL TIMEOUT 900
                LABELS "validation;literature;heavy")
            dns_add_val(val_tgv3d_128 QUARANTINE SERIAL TIMEOUT 900
                TARGET test_val_tgv3d ARG 128
                LABELS "validation;anchor;literature;heavy")
            dns_add_val(val_comp_cyl_re800 SERIAL TIMEOUT 900
                TARGET test_val_comp_cyl ARG re800
                LABELS "validation;anchor;heavy")
            dns_add_val(val_sphere_re300 QUARANTINE SERIAL TIMEOUT 1200
                LABELS "validation;anchor;literature;heavy")
            dns_add_val(val_mesh_sphere QUARANTINE SERIAL TIMEOUT 1200
                LABELS "validation;analytic;heavy")
        endif()

        get_property(dns_tests_all DIRECTORY PROPERTY TESTS)
        set(dns_tests_gpu ${dns_tests_all})
        list(REMOVE_ITEM dns_tests_gpu ${dns_tests_before_gpu})
        set_tests_properties(${dns_tests_gpu} PROPERTIES RESOURCE_LOCK gpu)
    endif()

    # Model-subset labels: `ctest -L rans`, `-L 'rans|les|mrf'`, `-L fast`.
    #   rans  the RANS closures, their oracle, stop rules and runners
    #   les   the SGS models and the resolved-turbulence references they are
    #         read against (TGV, forced HIT)
    #   mrf   rotors: moving wall, azimuth tables, MRF frame, tip guard
    #   fast  members of those three under ~6 s each (measured, gfx1100)
    # A name not registered in this configuration (no GPU, no FFTW) is skipped.
    function(dns_label label)
        foreach(t IN LISTS ARGN)
            if(TEST ${t})
                set_property(TEST ${t} APPEND PROPERTY LABELS ${label})
            endif()
        endforeach()
    endfunction()
    dns_label(rans channel_config transport rans rans_channel1d steady_stop
        blockband channel_line pseudo_time flow_rans_config gpu_channel gpu_rans
        gpu_rans_sst gpu_rans_mask gpu_rans_ogrid gpu_rans_mms gpu_axisym
        gpu_axisym3d gpu_nk gpu_rans_immersed gpu_runners gpu_sst gpu_sa_wedge
        gpu_axisym3d_run gpu_determinism gpu_rans_implicit)
    dns_label(les tgv tgv3d sgs_op hit_mini gpu_tgv gpu_comp_les gpu_les_smag
        gpu_hit_mini gpu_les_wall gpu_les_tally3d gpu_les_tgv_comp
        gpu_channel_les)
    dns_label(mrf rotor gpu_rotor_wall gpu_rotor_table gpu_mrf gpu_mrf_rotor2
        gpu_mrf_steady gpu_mrf_steady_lts gpu_precond gpu_mrf_conservation gpu_mrf_stabledt gpu_rotor_comoving_fixed_point
        gpu_ghost_translate rotor_tip_guard gpu_rotor_swap_ledger)
    dns_label(fast channel_config transport rans rans_channel1d steady_stop precond
        blockband channel_line pseudo_time flow_rans_config gpu_channel
        gpu_rans_mask gpu_rans_ogrid gpu_rans_mms gpu_axisym gpu_axisym3d gpu_nk
        gpu_rans_immersed gpu_determinism gpu_rans_implicit tgv tgv3d sgs_op
        gpu_tgv gpu_comp_les gpu_les_smag gpu_hit_mini gpu_les_wall
        gpu_les_tally3d gpu_les_tgv_comp gpu_channel_les rotor gpu_rotor_wall
        gpu_rotor_table gpu_mrf_conservation gpu_mrf_stabledt
        gpu_rotor_comoving_fixed_point gpu_ghost_translate rotor_tip_guard)
endif()
