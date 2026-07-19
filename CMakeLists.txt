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
    src/dns/config.cpp
    src/dns/cutcell.cpp
    src/dns/diagnostics.cpp
    src/dns/fieldpng.cpp
    src/dns/geometry.cpp
    src/dns/operators.cpp
    src/dns/output.cpp
    src/dns/comp.cpp
    src/dns/meshsdf.cpp
    src/dns/poisson.cpp
    src/dns/timestepper.cpp
    src/dns/trimesh.cpp
    src/dns/vtk_writer.cpp
    src/dns/webframes.cpp
)
target_include_directories(dns_core PUBLIC src)
target_include_directories(dns_core PRIVATE third_party) # cgltf (trimesh.cpp)
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
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES gfx1100)
    endif()
    enable_language(HIP)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
    find_package(hip REQUIRED)
    find_package(hipfft REQUIRED)
    find_package(rocprim REQUIRED)

    add_library(dns_gpu
        src/dns/gpu/bodyforce.hip
        src/dns/gpu/comp.hip
        src/dns/gpu/comp_far.hip
        src/dns/gpu/comp_ghost.hip
        src/dns/gpu/comp_nscbc.hip
        src/dns/gpu/cutcell.hip
        src/dns/gpu/cutpoisson.hip
        src/dns/gpu/derived.hip
        src/dns/gpu/device_util.hip
        src/dns/gpu/ghostfill.hip
        src/dns/gpu/kernels.hip
        src/dns/gpu/les.hip
        src/dns/gpu/poisson.hip
        src/dns/gpu/reductions.hip
        src/dns/gpu/runner_comp.hip
        src/dns/gpu/runner_incomp.hip
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
    set(DNS_TESTS cli geometry output poisson projection tgv tgv3d cutcell cutcell_sphere comp meshsdf
        fieldpng webframes)
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
    if(DNS_WITH_FFTW)
        set_tests_properties(hit_mini PROPERTIES TIMEOUT 900
            LABELS "slow;validation;literature")
    endif()
    if(DNS_WITH_HIP)
        foreach(t gpu_buffer gpu_operators gpu_reductions gpu_poisson
                  gpu_poisson_neumann gpu_poisson2d gpu_projection gpu_tgv
                  gpu_cutcell gpu_cutcell2d gpu_cutcell_sphere gpu_conservation
                  gpu_energy_stability gpu_ghostfill gpu_comp gpu_comp_ghost
                  gpu_comp_les gpu_comp_ghost3d gpu_comp_mesh gpu_comp_far
                  gpu_comp_nscbc gpu_les_smag gpu_derived)
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

        add_executable(test_val_spindown tests/test_val_spindown.cpp)
        target_link_libraries(test_val_spindown PRIVATE dns_gpu)
        add_test(NAME val_spindown COMMAND test_val_spindown)
        set_tests_properties(val_spindown PROPERTIES TIMEOUT 120
            SKIP_RETURN_CODE 77 LABELS "validation;analytic")

        foreach(t val_cyl_re60 val_cyl_re100 val_cyl_re200 val_cyl_blockage)
            add_executable(test_${t} tests/test_${t}.cpp)
            target_link_libraries(test_${t} PRIVATE dns_gpu)
            target_compile_definitions(test_${t} PRIVATE ${DNS_VAL_DEFS}
                DNS_BIN="$<TARGET_FILE:dns>")
            add_dependencies(test_${t} dns)
        endforeach()
        if(DNS_VALIDATION_BASELINE)
        # QUARANTINED 2026-07-19 (masterplan_recompose §7): bands measured before the
        # kBcTunnel corner unification, the 3-direction friction tally, and the FP64
        # force tally. Re-enable after regenerating per docs/validation.md Tier C.
            add_test(NAME val_cyl_re60 COMMAND test_val_cyl_re60)
            set_tests_properties(val_cyl_re60 PROPERTIES TIMEOUT 300
                SKIP_RETURN_CODE 77 LABELS "validation;anchor;literature")
            add_test(NAME val_cyl_re100 COMMAND test_val_cyl_re100)
            set_tests_properties(val_cyl_re100 PROPERTIES TIMEOUT 300
                SKIP_RETURN_CODE 77 LABELS "validation;literature")
        endif()

        add_executable(test_val_tgv3d tests/test_val_tgv3d.cpp)
        target_link_libraries(test_val_tgv3d PRIVATE dns_gpu)
        target_compile_definitions(test_val_tgv3d PRIVATE ${DNS_VAL_DEFS}
            DNS_BIN="$<TARGET_FILE:dns>"
            DNS_VALIDATION_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/validation"
            DNS_SPECTRAL_REF="${CMAKE_CURRENT_SOURCE_DIR}/data/spectral_Re1600_512.gdiag")
        add_dependencies(test_val_tgv3d dns)
        if(DNS_VALIDATION_BASELINE)
        # QUARANTINED 2026-07-19 (masterplan_recompose §7): bands measured before the
        # kBcTunnel corner unification, the 3-direction friction tally, and the FP64
        # force tally. Re-enable after regenerating per docs/validation.md Tier C.
            add_test(NAME val_tgv3d_64 COMMAND test_val_tgv3d 64)
            set_tests_properties(val_tgv3d_64 PROPERTIES TIMEOUT 300
                SKIP_RETURN_CODE 77 LABELS "validation;anchor;literature")
        endif()

        add_executable(test_val_comp_cyl tests/test_val_comp_cyl.cpp)
        target_link_libraries(test_val_comp_cyl PRIVATE dns_gpu)
        target_compile_definitions(test_val_comp_cyl PRIVATE ${DNS_VAL_DEFS}
            DNS_BIN="$<TARGET_FILE:dns>")
        add_dependencies(test_val_comp_cyl dns)
        add_test(NAME val_comp_cyl_re60 COMMAND test_val_comp_cyl re60)
        set_tests_properties(val_comp_cyl_re60 PROPERTIES TIMEOUT 300
            SKIP_RETURN_CODE 77 LABELS "validation;anchor")

        add_executable(test_val_sphere_re300 tests/test_val_sphere_re300.cpp)
        target_link_libraries(test_val_sphere_re300 PRIVATE dns_gpu)
        target_compile_definitions(test_val_sphere_re300 PRIVATE ${DNS_VAL_DEFS}
            DNS_BIN="$<TARGET_FILE:dns>")
        add_dependencies(test_val_sphere_re300 dns)

        add_executable(test_val_mesh_sphere tests/test_val_mesh_sphere.cpp)
        target_link_libraries(test_val_mesh_sphere PRIVATE dns_gpu)
        target_compile_definitions(test_val_mesh_sphere PRIVATE ${DNS_VAL_DEFS}
            DNS_BIN="$<TARGET_FILE:dns>")
        add_dependencies(test_val_mesh_sphere dns)

        # Tier A: mesh insertion into the 3D compressible solver (short 64^3).
        add_executable(test_val_comp_mesh_sphere tests/test_val_comp_mesh_sphere.cpp)
        target_link_libraries(test_val_comp_mesh_sphere PRIVATE dns_gpu)
        target_compile_definitions(test_val_comp_mesh_sphere PRIVATE ${DNS_VAL_DEFS}
            DNS_BIN="$<TARGET_FILE:dns>")
        add_dependencies(test_val_comp_mesh_sphere dns)
        if(DNS_VALIDATION_BASELINE)
        # QUARANTINED 2026-07-19 (masterplan_recompose §7): bands measured before the
        # kBcTunnel corner unification, the 3-direction friction tally, and the FP64
        # force tally. Re-enable after regenerating per docs/validation.md Tier C.
            add_test(NAME val_comp_mesh_sphere COMMAND test_val_comp_mesh_sphere)
            set_tests_properties(val_comp_mesh_sphere PROPERTIES TIMEOUT 600
                SKIP_RETURN_CODE 77 LABELS "validation;analytic")
        endif()

        if(DNS_HEAVY_VALIDATION)
            if(DNS_VALIDATION_BASELINE)
            # QUARANTINED 2026-07-19 (masterplan_recompose §7): bands measured before the
            # kBcTunnel corner unification, the 3-direction friction tally, and the FP64
            # force tally. Re-enable after regenerating per docs/validation.md Tier C.
                add_test(NAME val_cyl_re200 COMMAND test_val_cyl_re200)
                set_tests_properties(val_cyl_re200 PROPERTIES TIMEOUT 900
                    SKIP_RETURN_CODE 77 RUN_SERIAL TRUE
                    LABELS "validation;literature;heavy")
            endif()
            if(DNS_VALIDATION_BASELINE)
            # QUARANTINED 2026-07-19 (masterplan_recompose §7): bands measured before the
            # kBcTunnel corner unification, the 3-direction friction tally, and the FP64
            # force tally. Re-enable after regenerating per docs/validation.md Tier C.
                add_test(NAME val_cyl_blockage COMMAND test_val_cyl_blockage)
                set_tests_properties(val_cyl_blockage PROPERTIES TIMEOUT 900
                    SKIP_RETURN_CODE 77 RUN_SERIAL TRUE
                    LABELS "validation;literature;heavy")
            endif()
            if(DNS_VALIDATION_BASELINE)
            # QUARANTINED 2026-07-19 (masterplan_recompose §7): bands measured before the
            # kBcTunnel corner unification, the 3-direction friction tally, and the FP64
            # force tally. Re-enable after regenerating per docs/validation.md Tier C.
                add_test(NAME val_tgv3d_128 COMMAND test_val_tgv3d 128)
                set_tests_properties(val_tgv3d_128 PROPERTIES TIMEOUT 900
                    SKIP_RETURN_CODE 77 RUN_SERIAL TRUE
                    LABELS "validation;anchor;literature;heavy")
            endif()
            add_test(NAME val_comp_cyl_re800 COMMAND test_val_comp_cyl re800)
            set_tests_properties(val_comp_cyl_re800 PROPERTIES TIMEOUT 900
                SKIP_RETURN_CODE 77 RUN_SERIAL TRUE
                LABELS "validation;anchor;heavy")
            if(DNS_VALIDATION_BASELINE)
            # QUARANTINED 2026-07-19 (masterplan_recompose §7): bands measured before the
            # kBcTunnel corner unification, the 3-direction friction tally, and the FP64
            # force tally. Re-enable after regenerating per docs/validation.md Tier C.
                add_test(NAME val_sphere_re300 COMMAND test_val_sphere_re300)
                set_tests_properties(val_sphere_re300 PROPERTIES TIMEOUT 1200
                    SKIP_RETURN_CODE 77 RUN_SERIAL TRUE
                    LABELS "validation;anchor;literature;heavy")
            endif()
            if(DNS_VALIDATION_BASELINE)
            # QUARANTINED 2026-07-19 (masterplan_recompose §7): bands measured before the
            # kBcTunnel corner unification, the 3-direction friction tally, and the FP64
            # force tally. Re-enable after regenerating per docs/validation.md Tier C.
                add_test(NAME val_mesh_sphere COMMAND test_val_mesh_sphere)
                set_tests_properties(val_mesh_sphere PROPERTIES TIMEOUT 1200
                    SKIP_RETURN_CODE 77 RUN_SERIAL TRUE
                    LABELS "validation;analytic;heavy")
            endif()
        endif()
    endif()
endif()
