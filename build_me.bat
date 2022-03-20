@echo off
setlocal EnableExtensions EnableDelayedExpansion
for %%a in (GPU-Quicksort_OpenCL_1.2 GPU-Quicksort_OpenCL_2.0 GPU-Quicksort_SYCL GPU-Quicksort_Full_SYCL GPU-Quicksort_Full_SYCL_2.0) do (
  echo --------------- Algorithm %%a ----------------
  cd %%a
  echo ... Building ...
  dpcpp-cl /MD /GX /D_CRT_SECURE_NO_WARNINGS /D__SYCL_INTERNAL_API -o Quicksort.exe QuicksortMain.cpp OpenCL.lib
  echo ... Done ...
  set "nmbr=0"
  for %%c in (256 512 1024 2048 4096 8192 16384) do call :run_loops %%c
  cd ..
)
goto :eof

:run_loops
for %%i in (1 2 3) do (
  echo -------------- Trial %%i ---- %1 %1 ----------
  .\Quicksort.exe 10 gpu intel %1 %1 show_cl
)
set /A D=2*%1
for %%i in (1 2 3) do (
  echo -------------- Trial %%i ----- %D% %1 ---------
  .\Quicksort.exe 10 gpu intel %D% %1 show_cl
)

goto :eof
