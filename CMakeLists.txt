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
    src/dns/bodies.cpp
    src/dns/cli.cpp
    src/dns/casegeom.cpp
    src/dns/config.cpp
    src/dns/framemap.cpp
    src/dns/cutcell.cpp
    src/dns/diagnostics.cpp
    src/dns/fieldpng.cpp
    src/dns/flow_ic.cpp
    src/dns/geometry.cpp
    src/dns/operators.cpp
    src/dns/output.cpp
    src/dns/comp.cpp
    src/dns/curvmesh.cpp
    src/dns/interrupt.cpp
    src/dns/meshsdf.cpp
    src/dns/ogrid.cpp
    src/dns/poisson.cpp
    src/dns/rotor.cpp
    src/dns/scene.cpp
    src/dns/shmframes.cpp
    src/dns/timestepper.cpp
    src/dns/trimesh.cpp
    src/dns/vtk_writer.cpp
    src/dns/webframes.cpp
)
target_include_directories(dns_core PUBLIC src)
# cgltf (trimesh.cpp) + toml++ (scene.cpp); SYSTEM mutes third-party warnings.
target_include_directories(dns_core SYSTEM PRIVATE third_party)
target_link_libraries(dns_core PRIVATE ZLIB::ZLIB)

if(DNS_WITH_FFTW)
    find_path(FFTW3_INCLUDE_DIR fftw3.h REQUIRED)
    find_library(FFTW3_LIBRARY fftw3 REQUIRED)
    target_sources(dns_core PRIVATE src/dns/hit_ic.cpp)
    target_include_directories(dns_core PRIVATE ${FFTW3_INCLUDE_DIR})
    target_link_libraries(dns_core PRIVATE ${FFTW3_LIBRARY})
    target_compile_definitions(dns_core PUBLIC DNS_WITH_FFTW=1)
endif()

# The one executable. Everything else in apps/ was deleted by the
# recomposition; `dns <subcommand>` covers the whole former 15-binary surface.
add_executable(dns apps/dns_main.cpp)
target_link_libraries(dns PRIVATE dns_core)

# The one GPU source list, shared by both backends.
set(DNS_GPU_SOURCES
    src/dns/gpu/bodyforce.hip
    src/dns/gpu/boussinesq.hip
    src/dns/gpu/comp.hip
    src/dns/gpu/comp_channel.hip
    src/dns/gpu/comp_conv.hip
    src/dns/gpu/comp_far.hip
    src/dns/gpu/comp_ghost.hip
    src/dns/gpu/comp_nscbc.hip
    src/dns/gpu/nscbc_modal.hip
    src/dns/gpu/cutcell.hip
    src/dns/gpu/cutflux.hip
    src/dns/gpu/cutpoisson.hip
    src/dns/gpu/derived.hip
    src/dns/gpu/device_util.hip
    src/dns/gpu/ghostfill.hip
    src/dns/gpu/kernels.hip
    src/dns/gpu/les.hip
    src/dns/gpu/ogrid_bc.hip
    src/dns/gpu/poisson.hip
    src/dns/gpu/reductions.hip
    src/dns/gpu/rotortable.hip
    src/dns/gpu/runner_channel.hip
    src/dns/gpu/runner_comp.hip
    src/dns/gpu/runner_conv.hip
    src/dns/gpu/runner_ogrid.hip
    src/dns/gpu/runner_rb.hip
    src/dns/gpu/runner_incomp.hip
    src/dns/gpu/runner_periodic.hip
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
    target_compile_definitions(dns_gpu PUBLIC DNS_WITH_GPU=1 DNS_WITH_HIP=1)
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
    target_compile_definitions(dns_gpu PUBLIC DNS_WITH_GPU=1 DNS_WITH_CUDA=1)
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
        rotor)
    if(DNS_WITH_FFTW)
        list(APPEND DNS_TESTS ke_conservation hit_mini)
    endif()
    foreach(t IN LISTS DNS_TESTS)
        add_executable(test_${t} tests/test_${t}.cpp)
        target_link_libraries(test_${t} PRIVATE dns_core)
        add_test(NAME ${t} COMMAND test_${t})
        set_tests_properties(${t} PROPERTIES TIMEOUT 120)
    endforeach()
    target_link_libraries(test_fieldpng PRIVATE ZLIB::ZLIB) # decodes IDAT

    # The exit-code contract, driven through the real binary. GPU-free: every
    # case is reachable in phases 1-3a.
    add_executable(test_errors tests/test_errors.cpp)
    target_link_libraries(test_errors PRIVATE dns_core)
    target_compile_definitions(test_errors PRIVATE DNS_BIN="$<TARGET_FILE:dns>")
    add_dependencies(test_errors dns)
    add_test(NAME errors COMMAND test_errors)
    set_tests_properties(errors PROPERTIES TIMEOUT 120)
    if(DNS_WITH_FFTW)
        set_tests_properties(hit_mini PROPERTIES TIMEOUT 900
            LABELS "slow;validation;literature")
    endif()
    if(DNS_WITH_GPU)
        foreach(t gpu_buffer gpu_operators gpu_reductions gpu_poisson
                  gpu_poisson_neumann gpu_poisson2d gpu_projection gpu_tgv
                  gpu_cutcell gpu_cutcell2d gpu_cutcell_sphere gpu_conservation
                  gpu_energy_stability gpu_ghostfill gpu_comp gpu_comp_curv
                  gpu_comp_ghost gpu_comp_ghost_focal gpu_comp_les
                  gpu_comp_ghost3d gpu_comp_mesh
                  gpu_comp_far gpu_comp_nscbc gpu_nscbc_absorb gpu_nscbc_order
                  gpu_acoustics
                  gpu_les_smag gpu_derived
                  gpu_poisson_wally gpu_rotor_wall gpu_rotor_table
                  gpu_channel gpu_cutflux gpu_ogrid)
            add_executable(test_${t} tests/test_${t}.cpp)
            target_link_libraries(test_${t} PRIVATE dns_gpu)
            add_test(NAME ${t} COMMAND test_${t})
            set_tests_properties(${t} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
        endforeach()

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

        # dns conv is an app composition (IC + sink + runner), so it is driven
        # through the binary and read back from its stats CSV.
        add_executable(test_gpu_conv tests/test_gpu_conv.cpp)
        target_link_libraries(test_gpu_conv PRIVATE dns_gpu)
        target_compile_definitions(test_gpu_conv PRIVATE
            DNS_BIN="$<TARGET_FILE:dns>")
        add_dependencies(test_gpu_conv dns)
        add_test(NAME gpu_conv COMMAND test_gpu_conv)
        set_tests_properties(gpu_conv PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)

        # The Boussinesq branch, including the cross-validation against the
        # compressible one -- also driven through the binary.
        add_executable(test_gpu_rb tests/test_gpu_rb.cpp)
        target_link_libraries(test_gpu_rb PRIVATE dns_gpu)
        target_compile_definitions(test_gpu_rb PRIVATE
            DNS_BIN="$<TARGET_FILE:dns>")
        add_dependencies(test_gpu_rb dns)
        add_test(NAME gpu_rb COMMAND test_gpu_rb)
        set_tests_properties(gpu_rb PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)

        # Open far-field path is app-local; drive the built binary.
        add_executable(test_gpu_open_incomp tests/test_gpu_open_incomp.cpp)
        target_link_libraries(test_gpu_open_incomp PRIVATE dns_gpu)
        target_compile_definitions(test_gpu_open_incomp PRIVATE
            DNS_BIN="$<TARGET_FILE:dns>")
        add_dependencies(test_gpu_open_incomp dns)
        add_test(NAME gpu_open_incomp COMMAND test_gpu_open_incomp)
        set_tests_properties(gpu_open_incomp PROPERTIES TIMEOUT 600
            SKIP_RETURN_CODE 77)

        # Smoke coverage for the three runners, incl. --far tunnel WITH walls
        # (the incompressible default) and runner_comp, which had none.
        add_executable(test_gpu_runners tests/test_gpu_runners.cpp)
        target_link_libraries(test_gpu_runners PRIVATE dns_gpu)
        target_compile_definitions(test_gpu_runners PRIVATE
            DNS_BIN="$<TARGET_FILE:dns>")
        add_dependencies(test_gpu_runners dns)
        add_test(NAME gpu_runners COMMAND test_gpu_runners)
        set_tests_properties(gpu_runners PROPERTIES TIMEOUT 600
            SKIP_RETURN_CODE 77)

        # Sponge-free open tunnel: library-level freestream cycles + a short
        # end-to-end run of the built binary.
        add_executable(test_gpu_tunnel_open tests/test_gpu_tunnel_open.cpp)
        target_link_libraries(test_gpu_tunnel_open PRIVATE dns_gpu)
        target_compile_definitions(test_gpu_tunnel_open PRIVATE
            DNS_BIN="$<TARGET_FILE:dns>")
        add_dependencies(test_gpu_tunnel_open dns)
        add_test(NAME gpu_tunnel_open COMMAND test_gpu_tunnel_open)
        set_tests_properties(gpu_tunnel_open PROPERTIES TIMEOUT 600
            SKIP_RETURN_CODE 77)


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
                  val_comp_mesh_sphere)
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
    endif()
endif()
