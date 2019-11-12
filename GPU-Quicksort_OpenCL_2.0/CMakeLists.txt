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

if(WIN32)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
include_directories($ENV{INTELOCLSDKROOT}/include)
link_directories($ENV{INTELOCLSDKROOT}/lib/${bitness})
else(WIN32)
endif(WIN32)
add_executable(Quicksort QuicksortMain.cpp  Quicksort.h OpenCLUtils.h QuicksortKernels.cl)
if(WIN32)
  target_link_libraries(Quicksort OpenCL)
else(WIN32)
  if (NOT ${os_version_suffix} STREQUAL "")
    target_link_libraries(Quicksort OpenCL tbb imf)
  else()
    target_link_libraries(Quicksort OpenCL tbb)
  endif()
endif(WIN32)

add_test(NAME Quicksort_smoke
         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
         COMMAND Quicksort 100 gpu intel 2048 2048 no_show_cl)
