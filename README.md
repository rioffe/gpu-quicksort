# GPU Quicksort

GPU Quicksort implementation in OpenCL 1.2, 2.0, and SYCL

There are two SYCL implementations: GPU-Quicksort_SYCL, which only uses SYCL for platform initialization,
platform querying, buffer creation and kernel launch, but leaves the actual kernels in OpenCL C and uses
OpenCL API for building the program and selecting the kernels; and GPU-Quicksort_Full_SYCL, which fully
translates OpenCL C kernels to SYCL, and shows how to use templates to implement sorting for unsinged 
integers, floats and doubles.

This code base is going to accompany the upcoming article in the Parallel Universe Magazine tentatively
titled "GPU-Quicksort: from OpenCL to Data Parallel C++"

To build the SYCL portions of the repository you will need a compliant SYCL compiler, e.g. dpcpp from Intel.

Initial article is located here:
https://software.intel.com/en-us/articles/gpu-quicksort-in-opencl-20-using-nested-parallelism-and-work-group-scan-functions
