#!/bin/bash -x

for algorithm_name in GPU-Quicksort_OpenCL_1.2 GPU-Quicksort_OpenCL_2.0 GPU-Quicksort_SYCL GPU-Quicksort_Full_SYCL; do
cd $algorithm_name
mkdir build
cd build
CXX=dpcpp cmake .. -DCMAKE_BUILD_TYPE=Release
make 
cd ..
./build/Quicksort 10 gpu intel 256 256 show_cl
./build/Quicksort 10 gpu intel 512 256 no_show_cl
./build/Quicksort 10 gpu intel 512 512 no_show_cl
./build/Quicksort 10 gpu intel 1024 512 no_show_cl
./build/Quicksort 10 gpu intel 1024 1024 no_show_cl
./build/Quicksort 10 gpu intel 2048 1024 no_show_cl
./build/Quicksort 10 gpu intel 2048 2048 no_show_cl
./build/Quicksort 10 gpu intel 4096 2048 no_show_cl
./build/Quicksort 10 gpu intel 4096 4096 no_show_cl
./build/Quicksort 10 gpu intel 8192 4096 no_show_cl
./build/Quicksort 10 gpu intel 8192 8192 no_show_cl
./build/Quicksort 10 gpu intel 16384 8192 no_show_cl
./build/Quicksort 10 gpu intel 16384 16384 no_show_cl
./build/Quicksort 10 gpu intel 32768 16384 no_show_cl
cd ..
done
