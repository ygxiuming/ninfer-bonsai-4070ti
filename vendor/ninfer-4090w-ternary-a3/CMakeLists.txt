cmake_minimum_required(VERSION 3.28)

# The reference implementation is compiled for sm_120a (RTX 5090). sm_89 (RTX 4090)
# is supported for the architecture-generic kernel set (bf16/fp8/k8v4 paths); the
# warp-specialized SM120 NVFP4 W4A4 TMA kernels are excluded from non-120 builds
# (see src/CMakeLists.txt). Validate the architecture before CUDA compiler detection
# so CMake never injects a generic host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(120a|120|89)$")
  message(FATAL_ERROR
    "NInfer supports CMAKE_CUDA_ARCHITECTURES=120a or 89; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

# --- fork/package patch (2026-09-27) ------------------------------------------
# -DNINFER_SM_COUNT=<n> must reach the compiler as a MACRO, otherwise the branch in
# src/core/device.h is dead code and a "successful" rebuild silently keeps the
# per-arch default (this is exactly the "compile succeeded != compiled correctly"
# trap our own docs warn about). 0 = no override, use the per-arch default.
set(NINFER_SM_COUNT "0" CACHE STRING
  "Streaming-multiprocessor count of the card this build targets (0 = per-arch default)")
if(NOT NINFER_SM_COUNT MATCHES "^[0-9]+$")
  message(FATAL_ERROR
    "NINFER_SM_COUNT='${NINFER_SM_COUNT}' is not a number; read yours with gpu-probe.exe")
endif()
if(NINFER_SM_COUNT EQUAL 0)
  if(CMAKE_CUDA_ARCHITECTURES MATCHES "^(120|120a)")
    set(NINFER_SM_COUNT 170)   # RTX 5090 -- matches the shipped sm_120 engine
  else()
    set(NINFER_SM_COUNT 128)   # RTX 4090 -- matches the shipped sm_89 engine
  endif()
  message(STATUS
    "NINFER_SM_COUNT not given -> per-arch default ${NINFER_SM_COUNT}. Read your own card "
    "with gpu-probe.exe and pass -DNINFER_SM_COUNT=<n> to target it exactly.")
endif()
add_compile_definitions(NINFER_SM_COUNT=${NINFER_SM_COUNT})
message(STATUS "NInfer target: CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} SM_COUNT=${NINFER_SM_COUNT}")

project(ninfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# sm_89 (RTX 4090) builds enable the capped-shared-memory w8 kernel variants (see
# src/ops/linear/w8/w8_config.h); the 120a build keeps its default high-memory schedules.
if(CMAKE_CUDA_ARCHITECTURES MATCHES "89")
  add_compile_definitions(NINFER_SM89=1)
elseif(CMAKE_CUDA_ARCHITECTURES MATCHES "^(120a|120)$")
  # NOTE: this package tree carries the upstream 89/86 kernel set. Configuring for 120a works
  # and yields the right SM constant, but the NVFP4 / k8v4 sm_120 kernels are NOT in this tree.
  # For a real 50-series engine use the buildkit / gpu-samples tree.
  add_compile_definitions(NINFER_SM120=1)
endif()

if(MSVC)
  add_compile_definitions(NOMINMAX UTF8PROC_STATIC)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:C>:/FS>
    $<$<COMPILE_LANGUAGE:C>:/utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CXX>:/FS>
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/FS>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    # fork: MSVC 从 C++20 起默认开 /permissive-，导致 CUDA 生成代码里 lambda 访问
    # constexpr 静态成员报 C2326（见 SO 79894122）。关掉严格一致性即可，无需改源码。
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/permissive>
  )
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.1)
  message(FATAL_ERROR
    "NInfer requires CUDA 13.1 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()

option(NINFER_DISABLE_MEDIA "fork: build apps without FFmpeg (repo ships no ffmpeg/ dir)" OFF)
set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if((NINFER_BUILD_APPS OR BUILD_TESTING) AND NOT NINFER_DISABLE_MEDIA)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(CUDAToolkit REQUIRED)
if(NINFER_BUILD_MEDIA_ACQUIRE)
  if(MSVC)
    add_library(PkgConfig::FFMPEG INTERFACE IMPORTED)
    target_include_directories(PkgConfig::FFMPEG INTERFACE "${PROJECT_SOURCE_DIR}/ffmpeg/include")
    target_link_directories(PkgConfig::FFMPEG INTERFACE "${PROJECT_SOURCE_DIR}/ffmpeg/lib")
    target_link_libraries(PkgConfig::FFMPEG INTERFACE avcodec avformat avutil swscale swresample)
  else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
      libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
  endif()
endif()
find_package(Threads REQUIRED)

if(NINFER_BUILD_APPS OR BUILD_TESTING)
  # Product operational logging is an offline, repository-pinned dependency. Core-only builds do
  # not configure or link it.
  add_subdirectory(third_party/spdlog)
endif()

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
