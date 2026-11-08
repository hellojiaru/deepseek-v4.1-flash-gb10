cmake_minimum_required(VERSION 3.28)

# A separate CPU-only project is intentionally supported: no CUDA compiler,
# weights, network endpoints, fleet file, or inference service is required.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(dsgb10_host_tests LANGUAGES CXX)
  include(CTest)
endif()
get_filename_component(DSGB10_ENGINE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

function(dsgb10_host_test target source)
  add_executable(${target} "${DSGB10_ENGINE_DIR}/${source}")
  target_include_directories(${target} PRIVATE "${DSGB10_ENGINE_DIR}")
  target_compile_features(${target} PRIVATE cxx_std_20)
  if(MSVC)
    target_compile_options(${target} PRIVATE /UNDEBUG /EHsc)
  else()
    target_compile_options(${target} PRIVATE -UNDEBUG)
  endif()
  add_test(NAME ${target} COMMAND ${target})
endfunction()

if(BUILD_TESTING)
  dsgb10_host_test(vision_prefix_test server/tests/vision_prefix_test.cpp)
  dsgb10_host_test(vision_features_test server/tests/vision_features_test.cpp)

  # The actual batch-verdict header includes CUDA types, but this executable
  # calls only host helpers. CUDA headers are optional; no GPU is initialized.
  find_path(DSGB10_CUDA_INCLUDE_DIR cuda_runtime.h
    HINTS "$ENV{CUDA_PATH}/include" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
          /usr/local/cuda/include /usr/local/cuda-13.0/include)
  if(DSGB10_CUDA_INCLUDE_DIR)
    dsgb10_host_test(probability_host_test model/tests/probability_host_test.cpp)
    target_include_directories(probability_host_test PRIVATE "${DSGB10_CUDA_INCLUDE_DIR}")
  else()
    message(STATUS "CUDA headers absent: skipping probability_host_test (CPU helpers only)")
  endif()

  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME distribution_oracle
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/distribution_oracle.py")
  endif()

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(Threads REQUIRED)
    add_executable(runtime_metrics_test "${DSGB10_ENGINE_DIR}/server/tests/runtime_metrics_test.cpp")
    target_include_directories(runtime_metrics_test PRIVATE "${DSGB10_ENGINE_DIR}")
    target_compile_features(runtime_metrics_test PRIVATE cxx_std_20)
    target_compile_options(runtime_metrics_test PRIVATE -UNDEBUG)
    target_link_libraries(runtime_metrics_test PRIVATE Threads::Threads)
    add_test(NAME runtime_metrics_test
      COMMAND runtime_metrics_test "${CMAKE_CURRENT_BINARY_DIR}/runtime-metrics-test.json")

    # Uses the real restricted spool helper and /tmp path. Do not register it by
    # default where an inference frontend may own that path. Use a disposable
    # container with a private /tmp for this filesystem-only test.
    option(DSGB10_ENABLE_FILE_TESTS "Test real image spool files in an isolated /tmp" OFF)
    if(DSGB10_ENABLE_FILE_TESTS)
      dsgb10_host_test(vision_budget_test server/tests/vision_budget_test.cpp)
    endif()
  endif()
endif()

# Building this target is explicit; CTest never starts ranks, SSH, or GPUs.
option(DSGB10_BUILD_GPU_PROBE "Build the opt-in four-rank sampler probe" OFF)
if(DSGB10_BUILD_GPU_PROBE)
  if(NOT TARGET dsmodel)
    message(FATAL_ERROR "GPU probe requires the engine CUDA project, not the host-only project")
  endif()
  add_executable(probabilistic_sample_probe probabilistic_sample_probe.cu)
  target_include_directories(probabilistic_sample_probe PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
  target_link_libraries(probabilistic_sample_probe PRIVATE dsmodel)
  target_compile_options(probabilistic_sample_probe PRIVATE -UNDEBUG)
endif()
