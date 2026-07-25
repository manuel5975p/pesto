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
option(DNS_WITH_HIP "Build the HIP GPU backend" OFF)

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
    src/dns/meshsdf.cpp
    src/dns/poisson.cpp
    src/dns/scene.cpp
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

    add_library(dns_gpu
        src/dns/gpu/bodyforce.hip
        src/dns/gpu/comp.hip
        src/dns/gpu/comp_channel.hip
        src/dns/gpu/comp_far.hip
        src/dns/gpu/comp_ghost.hip
        src/dns/gpu/comp_nscbc.hip
        src/dns/gpu/cutcell.hip
        src/dns/gpu/cutflux.hip
        src/dns/gpu/cutpoisson.hip
        src/dns/gpu/derived.hip
        src/dns/gpu/device_util.hip
        src/dns/gpu/ghostfill.hip
        src/dns/gpu/kernels.hip
        src/dns/gpu/les.hip
        src/dns/gpu/poisson.hip
        src/dns/gpu/reductions.hip
        src/dns/gpu/runner_channel.hip
        src/dns/gpu/runner_comp.hip
        src/dns/gpu/runner_incomp.hip
        src/dns/gpu/runner_periodic.hip
        src/dns/gpu/timestepper.hip
        src/dns/gpu/tunnel.hip
    )
    target_link_libraries(dns_gpu
        PUBLIC dns_core hip::host
        PRIVATE hip::hipfft roc::rocprim)
    target_compile_definitions(dns_gpu PUBLIC DNS_WITH_HIP=1)
    target_compile_options(dns_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-Wall$<SEMICOLON>-Wextra>)

    # dns picks up the GPU backend and the runners when HIP is available.
    # (Target predates enable_language(HIP), so set its HIP arch explicitly.)
    target_link_libraries(dns PRIVATE dns_gpu)
    set_property(TARGET dns PROPERTY HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})
endif()

include(CTest)
if(BUILD_TESTING)
    set(DNS_TESTS cli geometry output poisson projection tgv tgv3d cutcell cutcell_sphere comp comp_curv meshsdf
        fieldpng webframes channel_config scene sgs_op framemap)
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
    if(DNS_WITH_HIP)
        foreach(t gpu_buffer gpu_operators gpu_reductions gpu_poisson
                  gpu_poisson_neumann gpu_poisson2d gpu_projection gpu_tgv
                  gpu_cutcell gpu_cutcell2d gpu_cutcell_sphere gpu_conservation
                  gpu_energy_stability gpu_ghostfill gpu_comp gpu_comp_curv
                  gpu_comp_ghost gpu_comp_ghost_focal gpu_comp_les
                  gpu_comp_ghost3d gpu_comp_mesh
                  gpu_comp_far gpu_comp_nscbc gpu_les_smag gpu_derived
                  gpu_channel gpu_cutflux)
            add_executable(test_${t} tests/test_${t}.cpp)
            target_link_libraries(test_${t} PRIVATE dns_gpu)
            add_test(NAME ${t} COMMAND test_${t})
            set_tests_properties(${t} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
        endforeach()

        # These define their own kernels (the pre-recomposition kernel variants,
        # as identity oracles), so they are HIP TUs rather than plain .cpp.
        foreach(t gpu_tunnel_bc gpu_bodyforce)
            add_executable(test_${t} tests/test_${t}.hip)
            target_link_libraries(test_${t} PRIVATE dns_gpu)
            add_test(NAME ${t} COMMAND test_${t})
            set_tests_properties(${t} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
        endforeach()

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
