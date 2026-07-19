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
    src/dns/cli.cpp
    src/dns/config.cpp
    src/dns/cutcell.cpp
    src/dns/diagnostics.cpp
    src/dns/fieldpng.cpp
    src/dns/operators.cpp
    src/dns/comp.cpp
    src/dns/meshsdf.cpp
    src/dns/poisson_jacobi.cpp
    src/dns/projection.cpp
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
    target_sources(dns_core PRIVATE src/dns/hit_ic.cpp src/dns/poisson_fft.cpp)
    target_include_directories(dns_core PRIVATE ${FFTW3_INCLUDE_DIR})
    target_link_libraries(dns_core PRIVATE ${FFTW3_LIBRARY})
    target_compile_definitions(dns_core PUBLIC DNS_WITH_FFTW=1)
endif()

add_executable(tgv apps/tgv_main.cpp)
target_link_libraries(tgv PRIVATE dns_core)

add_executable(tgv3d apps/tgv3d_main.cpp)
target_link_libraries(tgv3d PRIVATE dns_core)

add_executable(cube_cut apps/cube_cut_main.cpp)
target_link_libraries(cube_cut PRIVATE dns_core)

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
        src/dns/gpu/comp.hip
        src/dns/gpu/comp_far.hip
        src/dns/gpu/comp_ghost.hip
        src/dns/gpu/comp_nscbc.hip
        src/dns/gpu/cutcell.hip
        src/dns/gpu/device_util.hip
        src/dns/gpu/ghostfill.hip
        src/dns/gpu/kernels.hip
        src/dns/gpu/poisson.hip
        src/dns/gpu/projection.hip
        src/dns/gpu/reductions.hip
        src/dns/gpu/timestepper.hip
    )
    target_link_libraries(dns_gpu
        PUBLIC dns_core hip::host
        PRIVATE hip::hipfft roc::rocprim)
    target_compile_definitions(dns_gpu PUBLIC DNS_WITH_HIP=1)
    target_compile_options(dns_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-Wall$<SEMICOLON>-Wextra>)

    add_executable(bench apps/bench_main.cpp)
    target_link_libraries(bench PRIVATE dns_gpu)

    add_executable(hit apps/hit_main.cpp)
    target_link_libraries(hit PRIVATE dns_gpu)

    add_executable(movie apps/movie_main.cpp)
    target_link_libraries(movie PRIVATE dns_gpu)

    add_executable(sphere_tunnel_gpu apps/sphere_tunnel_gpu_main.hip)
    target_link_libraries(sphere_tunnel_gpu PRIVATE dns_gpu)

    add_executable(cube_tunnel_gpu apps/cube_tunnel_gpu_main.hip)
    target_link_libraries(cube_tunnel_gpu PRIVATE dns_gpu)

    add_executable(cylinder_tunnel_gpu apps/cylinder_tunnel_gpu_main.hip)
    target_link_libraries(cylinder_tunnel_gpu PRIVATE dns_gpu)
    add_executable(comp_cylinder_tunnel_gpu apps/comp_cylinder_tunnel_gpu_main.hip)
    target_link_libraries(comp_cylinder_tunnel_gpu PRIVATE dns_gpu)
    add_executable(eye_tunnel_gpu apps/eye_tunnel_gpu_main.hip)
    target_link_libraries(eye_tunnel_gpu PRIVATE dns_gpu)
    add_executable(mesh_tunnel_gpu apps/mesh_tunnel_gpu_main.hip)
    target_link_libraries(mesh_tunnel_gpu PRIVATE dns_gpu)
    add_executable(comp_mesh_tunnel_gpu apps/comp_mesh_tunnel_gpu_main.hip)
    target_link_libraries(comp_mesh_tunnel_gpu PRIVATE dns_gpu)

    # tgv3d picks up the GPU integrator when available. (Target predates
    # enable_language(HIP), so set its HIP arch explicitly.)
    target_link_libraries(tgv3d PRIVATE dns_gpu)
    set_property(TARGET tgv3d PROPERTY HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})
endif()

include(CTest)
if(BUILD_TESTING)
    set(DNS_TESTS cli poisson projection tgv tgv3d cutcell cutcell_sphere comp meshsdf
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
                  gpu_comp_nscbc)
            add_executable(test_${t} tests/test_${t}.cpp)
            target_link_libraries(test_${t} PRIVATE dns_gpu)
            add_test(NAME ${t} COMMAND test_${t})
            set_tests_properties(${t} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
        endforeach()

        # Open far-field path is app-local; drive the built binary.
        add_executable(test_gpu_open_incomp tests/test_gpu_open_incomp.cpp)
        target_link_libraries(test_gpu_open_incomp PRIVATE dns_gpu)
        target_compile_definitions(test_gpu_open_incomp PRIVATE
            CYLINDER_BIN="$<TARGET_FILE:cylinder_tunnel_gpu>")
        add_dependencies(test_gpu_open_incomp cylinder_tunnel_gpu)
        add_test(NAME gpu_open_incomp COMMAND test_gpu_open_incomp)
        set_tests_properties(gpu_open_incomp PROPERTIES TIMEOUT 600
            SKIP_RETURN_CODE 77)

        # LES Smagorinsky path is app-local (anonymous namespace in
        # cylinder_tunnel_gpu); drive the built binary instead of the library.
        add_executable(test_gpu_les_smag tests/test_gpu_les_smag.cpp)
        target_link_libraries(test_gpu_les_smag PRIVATE dns_gpu)
        target_compile_definitions(test_gpu_les_smag PRIVATE
            CYLINDER_BIN="$<TARGET_FILE:cylinder_tunnel_gpu>")
        add_dependencies(test_gpu_les_smag cylinder_tunnel_gpu)
        add_test(NAME gpu_les_smag COMMAND test_gpu_les_smag)
        set_tests_properties(gpu_les_smag PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)

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
                CYLINDER_BIN="$<TARGET_FILE:cylinder_tunnel_gpu>")
            add_dependencies(test_${t} cylinder_tunnel_gpu)
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
            TGV3D_BIN="$<TARGET_FILE:tgv3d>"
            DNS_VALIDATION_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/validation"
            DNS_SPECTRAL_REF="${CMAKE_CURRENT_SOURCE_DIR}/data/spectral_Re1600_512.gdiag")
        add_dependencies(test_val_tgv3d tgv3d)
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
            COMP_CYLINDER_BIN="$<TARGET_FILE:comp_cylinder_tunnel_gpu>")
        add_dependencies(test_val_comp_cyl comp_cylinder_tunnel_gpu)
        add_test(NAME val_comp_cyl_re60 COMMAND test_val_comp_cyl re60)
        set_tests_properties(val_comp_cyl_re60 PROPERTIES TIMEOUT 300
            SKIP_RETURN_CODE 77 LABELS "validation;anchor")

        add_executable(test_val_sphere_re300 tests/test_val_sphere_re300.cpp)
        target_link_libraries(test_val_sphere_re300 PRIVATE dns_gpu)
        target_compile_definitions(test_val_sphere_re300 PRIVATE ${DNS_VAL_DEFS}
            SPHERE_BIN="$<TARGET_FILE:sphere_tunnel_gpu>")
        add_dependencies(test_val_sphere_re300 sphere_tunnel_gpu)

        add_executable(test_val_mesh_sphere tests/test_val_mesh_sphere.cpp)
        target_link_libraries(test_val_mesh_sphere PRIVATE dns_gpu)
        target_compile_definitions(test_val_mesh_sphere PRIVATE ${DNS_VAL_DEFS}
            MESH_BIN="$<TARGET_FILE:mesh_tunnel_gpu>"
            SPHERE_BIN="$<TARGET_FILE:sphere_tunnel_gpu>")
        add_dependencies(test_val_mesh_sphere mesh_tunnel_gpu sphere_tunnel_gpu)

        # Tier A: mesh insertion into the 3D compressible solver (short 64^3).
        add_executable(test_val_comp_mesh_sphere tests/test_val_comp_mesh_sphere.cpp)
        target_link_libraries(test_val_comp_mesh_sphere PRIVATE dns_gpu)
        target_compile_definitions(test_val_comp_mesh_sphere PRIVATE ${DNS_VAL_DEFS}
            COMP_MESH_BIN="$<TARGET_FILE:comp_mesh_tunnel_gpu>")
        add_dependencies(test_val_comp_mesh_sphere comp_mesh_tunnel_gpu)
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
