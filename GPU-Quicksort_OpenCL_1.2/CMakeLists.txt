project(Quicksort)

message("-- CMAKE_SYSTEM_INFO_FILE: ${CMAKE_SYSTEM_INFO_FILE}")
message("-- CMAKE_SYSTEM_NAME:      ${CMAKE_SYSTEM_NAME}")
message("-- CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message("-- CMAKE_SYSTEM:           ${CMAKE_SYSTEM}")

string (REGEX MATCH "\\.el[1-9]" os_version_suffix ${CMAKE_SYSTEM})
message("-- os_version_suffix:      ${os_version_suffix}")

cmake_minimum_required(VERSION 2.8)

enable_testing()

set(bitness "x86")
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(bitness "x64")
endif()

option(NVIDIA_GPU "Compile for NVidia hardware: slightly different global/local WG sizes" OFF)

if(WIN32)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
include_directories($ENV{INTELOCLSDKROOT}/include)
link_directories($ENV{INTELOCLSDKROOT}/lib/${bitness})
else(WIN32)
add_definitions(-std=c++14)
link_directories(/opt/intel/inteloneapi/compiler/latest/linux/lib/ /opt/intel/inteloneapi/compiler/latest/linux/lib/x64)
endif(WIN32)
if(NVIDIA_GPU)
add_definitions(-DNVIDIA_GPU=1)
endif(NVIDIA_GPU)
add_executable(Quicksort QuicksortMain.cpp  Quicksort.h OpenCLUtils.h QuicksortKernels.cl)
if(WIN32)
target_link_libraries(Quicksort OpenCL)
else(WIN32)
target_link_libraries(Quicksort OpenCL tbb pthread stdc++)
endif(WIN32)

add_test(NAME Quicksort_smoke
         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
         COMMAND Quicksort 100 gpu intel 2048 2048 no_show_cl)
