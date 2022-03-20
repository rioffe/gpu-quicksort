#!/bin/bash 

for algorithm_name in GPU-Quicksort_OpenCL_1.2 GPU-Quicksort_OpenCL_2.0 GPU-Quicksort_SYCL GPU-Quicksort_Full_SYCL GPU-Quicksort_Full_SYCL_2.0; do
  echo --------------- Algorithm $algorithm_name ----------------
  cd $algorithm_name
  mkdir build
  cd build
  CXX=dpcpp cmake .. -DCMAKE_BUILD_TYPE=Release
  make
  cd ..
  for count in 256 512 1024 2048 4096 8192 16384; do
    for i in 1 2 3; do
      echo -------------- Trial $i --------------
      ./build/Quicksort 10 gpu intel $count $count show_cl
    done
    for i in 1 2 3; do
      echo -------------- Trial $i --------------
      ./build/Quicksort 10 gpu intel $((count*2)) $count show_cl
    done
  done
  cd ..
done
