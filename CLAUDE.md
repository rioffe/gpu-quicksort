# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU Quicksort: parallel quicksort implementations for GPU, progressing from OpenCL to SYCL/DPC++. Accompanies an Intel Parallel Universe Magazine article on porting GPU-Quicksort from OpenCL to Data Parallel C++.

## Build Commands

Requires Intel DPC++ compiler (`dpcpp`) and Intel oneAPI toolkit.

**Build a single variant (Linux/macOS):**
```bash
cd GPU-Quicksort_Full_SYCL   # or any variant directory
mkdir build && cd build
CXX=dpcpp cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

**Build for NVIDIA GPU:** add `-DNVIDIA_GPU=ON` to cmake.

**Windows (from Intel oneAPI Tools shell):**
```bat
dpcpp-cl /MD /GX /D_CRT_SECURE_NO_WARNINGS -o Quicksort.exe QuicksortMain.cpp OpenCL.lib
```

**Build and run all variants:** `./build_me.sh` (Linux) or `build_me.bat` (Windows).

**Run executable:**
```bash
./build/Quicksort <iterations> <gpu|cpu> <intel|nvidia> <array_size> <block_size> <show_cl|no_show_cl>
# Example: ./build/Quicksort 10 gpu intel 2048 2048 no_show_cl
```

**Run smoke test:** `cd build && ctest` (each variant has a `Quicksort_smoke` test).

## Architecture

Five independent implementations in separate directories, each a standalone CMake project producing the same `Quicksort` executable:

| Directory | API | Kernels | Notes |
|-----------|-----|---------|-------|
| `GPU-Quicksort_OpenCL_1.2` | OpenCL 1.2 | `.cl` files | Baseline; uses host-side work scheduling |
| `GPU-Quicksort_OpenCL_2.0` | OpenCL 2.0 | `.cl` files | Uses nested parallelism and work-group scan |
| `GPU-Quicksort_SYCL` | SYCL + OpenCL interop | `.cl` files | SYCL for platform/buffers, kernels remain in OpenCL C |
| `GPU-Quicksort_Full_SYCL` | Pure SYCL | C++ templates | Fully ported kernels; supports uint/float/double via templates |
| `GPU-Quicksort_Full_SYCL_2.0` | Modern SYCL 2.0 | C++ templates | Uses `get_native<backend::opencl>`, `make_kernel<backend::opencl>` APIs |

### Key Files Per Variant

- **`QuicksortMain.cpp`** - Entry point, device selection, benchmark loop, host-side orchestration of the two-kernel sort
- **`Quicksort.h`** - Shared data structures (`work_record`, `block_record`, `parent_record`), tuning constants, `median`/`median_select` functions
- **`OpenCLUtils.h`** - OpenCL error checking macros and build log utility (OpenCL/hybrid variants only)
- **`QuicksortKernels.cl`** - GPU kernels in OpenCL C (OpenCL and SYCL-interop variants only)
- **`CMakeLists.txt`** - Per-variant build; links against OpenCL, tbb, and optionally imf (RHEL)

### Algorithm Structure (Two-Kernel Design)

1. **`gqsort_kernel`** (Global Quicksort) - Partitions large blocks around a pivot using work-group parallel scan; produces sub-blocks for further processing
2. **`lqsort_kernel`** (Local Quicksort) - Sorts blocks smaller than `QUICKSORT_BLOCK_SIZE` using bitonic sort within a work-group

Host code repeatedly launches `gqsort_kernel` to subdivide the array, then `lqsort_kernel` to sort leaf blocks. Work records track pending partitions between kernel launches.

### Tuning Constants (in `Quicksort.h`)

Three preset configurations selected via preprocessor defines (`CPU_DEVICE`, `NVIDIA_GPU`, default=Intel GPU):
- `QUICKSORT_BLOCK_SIZE` - Threshold for switching from gqsort to lqsort
- `GQSORT_LOCAL_WORKGROUP_SIZE` / `LQSORT_LOCAL_WORKGROUP_SIZE` - Work-group sizes
- `SORT_THRESHOLD` - Must be 2x `LQSORT_LOCAL_WORKGROUP_SIZE` (bitonic sort requirement)

## Branches

- **`main`** - Stable release (default on GitHub)
- **`master`** - Development branch with additional `GPU-Quicksort_Full_SYCL_2.0` variant and modernization work not yet merged to main
