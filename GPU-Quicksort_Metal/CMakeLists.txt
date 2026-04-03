cmake_minimum_required(VERSION 3.19)
project(Quicksort LANGUAGES CXX OBJCXX)

enable_testing()

# Compile Metal shaders to .metallib
find_program(XCRUN xcrun REQUIRED)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/QuicksortKernels.air
    COMMAND ${XCRUN} -sdk macosx metal
            -I ${CMAKE_SOURCE_DIR}
            -c ${CMAKE_SOURCE_DIR}/QuicksortKernels.metal
            -o ${CMAKE_BINARY_DIR}/QuicksortKernels.air
    DEPENDS ${CMAKE_SOURCE_DIR}/QuicksortKernels.metal
            ${CMAKE_SOURCE_DIR}/QuicksortTypes.h
    COMMENT "Compiling Metal shaders"
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/QuicksortKernels.metallib
    COMMAND ${XCRUN} -sdk macosx metallib
            ${CMAKE_BINARY_DIR}/QuicksortKernels.air
            -o ${CMAKE_BINARY_DIR}/QuicksortKernels.metallib
    DEPENDS ${CMAKE_BINARY_DIR}/QuicksortKernels.air
    COMMENT "Linking Metal library"
)

add_custom_target(MetalShaders ALL
    DEPENDS ${CMAKE_BINARY_DIR}/QuicksortKernels.metallib
)

# Build the host executable
add_executable(Quicksort QuicksortMain.mm QuicksortTypes.h)
add_dependencies(Quicksort MetalShaders)

target_compile_definitions(Quicksort PRIVATE RUN_CPU_SORTS=1)

set_target_properties(Quicksort PROPERTIES
    CXX_STANDARD 17
    OBJCXX_STANDARD 17
)

target_link_libraries(Quicksort
    "-framework Metal"
    "-framework Foundation"
)

# Optional GCD parallel sort for CPU comparison
option(USE_GCD_SORT "Enable GCD-based parallel sort for CPU comparison" OFF)
if(USE_GCD_SORT)
    target_compile_definitions(Quicksort PRIVATE USE_GCD_SORT=1)
endif()

add_test(NAME Quicksort_smoke
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
         COMMAND Quicksort 3 2048 2048)
