/*
Copyright (c) 2014-2019, Intel Corporation
Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions 
are met:
* Redistributions of source code must retain the above copyright 
      notice, this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above 
      copyright notice, this list of conditions and the following 
      disclaimer in the documentation and/or other materials provided 
      with the distribution.
      * Neither the name of Intel Corporation nor the names of its 
      contributors may be used to endorse or promote products 
      derived from this software without specific prior written 
      permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
*/

// QuicksortMain.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#ifdef _MSC_VER
// Windows
#include <windows.h>
#include <tchar.h>
#endif
#include <assert.h>
#include <string.h>
#ifndef _MSC_VER
// Linux
#include <time.h>
#include <unistd.h>
#endif
#include "OpenCLUtils.h"
#include <math.h>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <map>

#include "tbb/parallel_sort.h"
// Types:
typedef unsigned int uint;
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#define READ_ALIGNMENT  4096 // Intel recommended alignment
#define WRITE_ALIGNMENT 4096 // Intel recommended alignment

/// return a timestamp with sub-second precision 
/** QueryPerformanceCounter and clock_gettime have an undefined starting point (null/zero)     
 *  and can wrap around, i.e. be nulled again. **/ 
double seconds() { 
#ifdef _MSC_VER   
  static LARGE_INTEGER frequency;   
  if (frequency.QuadPart == 0)     ::QueryPerformanceFrequency(&frequency);   
  LARGE_INTEGER now;   
  ::QueryPerformanceCounter(&now);   
  return now.QuadPart / double(frequency.QuadPart); 
#else   
  struct timespec now;   
  clock_gettime(CLOCK_MONOTONIC, &now);   
  return now.tv_sec + now.tv_nsec / 1000000000.0; 
#endif 
}
 
typedef struct
{	
	// CL platform handles:
	cl_device_id		deviceID;
	cl_context			contextHdl;
	cl_program			programHdl;
	cl_command_queue	cmdQHdl;
    cl_kernel           relauncher_kernel;	
	// Create buffer objects for memory.
	// allocate buffers
	cl_mem db;
	cl_mem dnb;
	cl_mem blocksb;
	cl_mem parentsb;
	cl_mem newsb;
	cl_mem workb;
	cl_mem doneb;
} OCLResources;

// Globals:
cl_int		ciErrNum;

void Cleanup(OCLResources* pOCL, int iExitCode, bool bExit, const char* optionalErrorMessage)
{
	if (optionalErrorMessage) 
		printf ("%s\n", optionalErrorMessage);

	// release buffers: we are done	
	if (pOCL->workb) {
		ciErrNum = clReleaseMemObject(pOCL->workb);
		CheckCLError (ciErrNum, "Could not memory object.", "Memory object released successfully.");
		pOCL->workb = NULL;
	}
	if (pOCL->doneb) {
		ciErrNum = clReleaseMemObject(pOCL->doneb);
		CheckCLError (ciErrNum, "Could not memory object.", "Memory object released successfully.");
		pOCL->doneb = NULL;
	}
	if (pOCL->newsb) {
		ciErrNum = clReleaseMemObject(pOCL->newsb);
		CheckCLError (ciErrNum, "Could not memory object.", "Memory object released successfully.");
		pOCL->newsb = NULL;
	}
	if (pOCL->parentsb) {
		ciErrNum = clReleaseMemObject(pOCL->parentsb);
		CheckCLError (ciErrNum, "Could not memory object.", "Memory object released successfully.");
		pOCL->parentsb = NULL;
	}
	if (pOCL->blocksb) {
		ciErrNum = clReleaseMemObject(pOCL->blocksb);
		CheckCLError (ciErrNum, "Could not memory object.", "Memory object released successfully.");
		pOCL->blocksb = NULL;
	}
	if (pOCL->db) {
		ciErrNum = clReleaseMemObject(pOCL->db);
		CheckCLError (ciErrNum, "Could not memory object.", "Memory object released successfully.");
		pOCL->db = NULL;
	}
	if (pOCL->dnb) {
		ciErrNum = clReleaseMemObject(pOCL->dnb);
		CheckCLError (ciErrNum, "Could not memory object.", "Memory object released successfully.");	
		pOCL->dnb = NULL;
	}
	
	// Release all OpenCL kernel, program, command queue, context:
	if (pOCL->relauncher_kernel) {
		ciErrNum = clReleaseKernel(pOCL->relauncher_kernel);
		CheckCLError (ciErrNum, "Could not release kernel.", "Kernel released successfully.");
		pOCL->relauncher_kernel = NULL;	
	}
	if (pOCL->programHdl)		{ 
		ciErrNum = clReleaseProgram(pOCL->programHdl);
		CheckCLError (ciErrNum, "Could not release program.", "Program released successfully.");		
		pOCL->programHdl=NULL;	
	}
	if (pOCL->cmdQHdl)			{ 
		ciErrNum = clReleaseCommandQueue(pOCL->cmdQHdl);
		CheckCLError (ciErrNum, "Could not release command queue.", "Command queue released successfully.");
		pOCL->cmdQHdl=NULL;		
	}
	if (pOCL->contextHdl)		{ 
		ciErrNum = clReleaseContext(pOCL->contextHdl);
		CheckCLError (ciErrNum, "Could not release context.", "Context released successfully.");
		pOCL->contextHdl= NULL;	
	}

	memset(pOCL, 0, sizeof (OCLResources));

	if (bExit)
		exit (iExitCode);
}

void parseArgs(OCLResources* pOCL, int argc, char** argv, unsigned int* test_iterations, char* pDeviceStr, char* pVendorStr, unsigned int* widthReSz, unsigned int* heightReSz, bool* pbShowCL)
{	
	char*			pDeviceWStr = NULL;
	char*			pVendorWStr = NULL;
	const char sUsageString[512] = "Usage: Quicksort [num test iterations] [cpu|gpu] [intel|amd|nvidia] [SurfWidth(^2 only)] [SurfHeight(^2 only)] [show_CL | no_show_CL]";
	
	if (argc != 7)
	{
		Cleanup (pOCL, -1, true, sUsageString);
	}
	else
	{
		*test_iterations	= atoi (argv[1]);
		pDeviceWStr			= argv[2];			// "cpu" or "gpu"	
		pVendorWStr			= argv[3];			// "intel" or "amd" or "nvidia"
		*widthReSz	= atoi (argv[4]);
		*heightReSz	= atoi (argv[5]);
		if (argv[6][0]=='s')
			*pbShowCL = true;
		else
			*pbShowCL = false;
	}
	sprintf (pDeviceStr, "%s", pDeviceWStr);
	sprintf (pVendorStr, "%s", pVendorWStr);
}

#define RUN_CPU_SORTS
#define HOST 1
#include "Quicksort.h"


template <class T>
void InstantiateOpenCLKernels(OCLResources *pOCL, size_t size, const size_t MAXSEQ, const size_t MAX_SIZE, work_record* pdone, work_record* pnews, T* d)
{	
	pOCL->relauncher_kernel = clCreateKernel(pOCL->programHdl, "relauncher_kernel", &ciErrNum);
	CheckCLError (ciErrNum, "Kernel creation failed.", "Kernel created.");

	// Create buffer objects for memory.
	// allocate buffers
	pOCL->db = clCreateBuffer(pOCL->contextHdl, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, ((sizeof(T)*size)/64 + 1)*64, d, &ciErrNum);
	CheckCLError (ciErrNum, "clCreateBuffer failed.", "clCreateBuffer.");
	pOCL->dnb = clCreateBuffer(pOCL->contextHdl, CL_MEM_READ_WRITE, sizeof(T)*size, 0, &ciErrNum);
	CheckCLError (ciErrNum, "clCreateBuffer failed.", "clCreateBuffer.");
	pOCL->blocksb = clCreateBuffer(pOCL->contextHdl, CL_MEM_READ_WRITE, sizeof(block_record)*MAX_SIZE, 0, &ciErrNum);
	CheckCLError (ciErrNum, "clCreateBuffer failed.", "clCreateBuffer.");
	pOCL->parentsb = clCreateBuffer(pOCL->contextHdl, CL_MEM_READ_WRITE, sizeof(parent_record)*MAX_SIZE, 0, &ciErrNum);
	CheckCLError (ciErrNum, "clCreateBuffer failed.", "clCreateBuffer.");
	//cl_mem newsb = clCreateBuffer(pOCL->contextHdl, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(work_record)*MAX_SIZE, &news[0], &ciErrNum);
	pOCL->newsb = clCreateBuffer(pOCL->contextHdl, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, ((sizeof(work_record)*MAX_SIZE)/64+1)*64, pnews, &ciErrNum);
	CheckCLError (ciErrNum, "clCreateBuffer failed.", "clCreateBuffer.");
	pOCL->workb = clCreateBuffer(pOCL->contextHdl, CL_MEM_READ_WRITE, sizeof(work_record)*MAX_SIZE, 0, &ciErrNum);
	CheckCLError (ciErrNum, "clCreateBuffer failed.", "clCreateBuffer.");
	//cl_mem doneb = clCreateBuffer(pOCL->contextHdl, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(work_record)*MAX_SIZE, &done[0], &ciErrNum);
	pOCL->doneb = clCreateBuffer(pOCL->contextHdl, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, ((sizeof(work_record)*MAX_SIZE)/64+1)*64, pdone, &ciErrNum);
	CheckCLError (ciErrNum, "clCreateBuffer failed.", "clCreateBuffer.");

	ciErrNum |= clSetKernelArg(pOCL->relauncher_kernel, 0, sizeof(cl_mem), (void*) &pOCL->db);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");
	ciErrNum |= clSetKernelArg(pOCL->relauncher_kernel, 1, sizeof(cl_mem), (void*) &pOCL->dnb);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");
	ciErrNum |= clSetKernelArg(pOCL->relauncher_kernel, 2, sizeof(cl_mem), (void*) &pOCL->blocksb);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");
	ciErrNum |= clSetKernelArg(pOCL->relauncher_kernel, 3, sizeof(cl_mem), (void*) &pOCL->parentsb);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");
	ciErrNum |= clSetKernelArg(pOCL->relauncher_kernel, 4, sizeof(cl_mem), (void*) &pOCL->newsb);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");
	ciErrNum |= clSetKernelArg(pOCL->relauncher_kernel, 5, sizeof(cl_mem), (void*) &pOCL->workb);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");
	ciErrNum |= clSetKernelArg(pOCL->relauncher_kernel, 6, sizeof(cl_mem), (void*) &pOCL->doneb);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");

	ciErrNum |= clSetKernelArg(pOCL->relauncher_kernel, 8, sizeof(uint), (void*) &MAXSEQ);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");
}
template <class T>
T* partition(T* left, T* right, T pivot) {
    // move pivot to the end
    T temp = *right;
    *right = pivot;
    *left = temp;

    T* store = left;

    for(T* p = left; p != right; p++) {
        if (*p < pivot) {
            temp = *store;
            *store = *p;
            *p = temp;
            store++;
        }
    }

    temp = *store;
    *store = pivot;
    *right = temp;

    return store;
}

template <class T>
void quicksort(T* data, int left, int right)
{
    T* store = partition(data + left, data + right, data[left]);
    int nright = store-data;
    int nleft = nright+1;

    if (left < nright) {
      if (nright - left > 32) {
        quicksort(data, left, nright);
      } else
        std::sort(data + left, data + nright + 1);
    }

    if (nleft < right) {
      if (right - nleft > 32)  {
		    quicksort(data, nleft, right); 
      } else {
        std::sort(data + nleft, data + right + 1);
      }
	}
}

size_t optp(size_t s, double k, size_t m) {
	return (size_t)pow(2, floor(log(s*k + m)/log(2.0) + 0.5));
}

template <class T>
void GPUQSort(OCLResources *pOCL, size_t size, T* d, work_record* pdone, work_record* pnews, const size_t MAX_SIZE)  {
	uint startpivot = median(d[0], d[size/2], d[size-1]);	
	
	for(uint i = 0; i < MAX_SIZE; i++) {
		new (pdone+i) work_record();
		if (i == 0)
			new (pnews) work_record(0, size, startpivot, 1);
		else
			new (pnews+i) work_record();
	}

	uint done_size = 0;
	ciErrNum |= clSetKernelArg(pOCL->relauncher_kernel, 7, sizeof(uint), (void*) &done_size);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");

	uint num_workgroups = 1;
	ciErrNum |= clSetKernelArg(pOCL->relauncher_kernel, 9, sizeof(uint), (void*) &num_workgroups);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");

	size_t		dimNDR = 1;	
	ciErrNum = clEnqueueNDRangeKernel (pOCL->cmdQHdl, pOCL->relauncher_kernel, 1, NULL, &dimNDR, 0, 0, NULL, 0);
	CheckCLError(ciErrNum, "clEnqueueNDRangeKernel failed.", "clEnqueueNDRangeKernel");

	T* foo = (T*)clEnqueueMapBuffer(pOCL->cmdQHdl, pOCL->db, CL_TRUE, CL_MAP_READ, 0, sizeof(T)*size, 0, 0, 0, &ciErrNum); 
	CheckCLError(ciErrNum, "clEnqueueMapBuffer failed.", "clEnqueueMapBuffer");

	ciErrNum = clEnqueueUnmapMemObject(pOCL->cmdQHdl, pOCL->db, foo, 0, 0, 0);
	CheckCLError(ciErrNum, "clEnqueueUnmapMemObject failed.", "clEnqueueUnmapMemObject");
}

int main(int argc, char** argv)
{
	OCLResources	myOCL;
	unsigned int	NUM_ITERATIONS;
	char			pDeviceStr[256];
	char			pVendorStr[256];
	const char*		pSourceFileStr	= "QuicksortKernels.cl";
	bool			bShowCL = false;

	uint			heightReSz, widthReSz;

	double totalTime, quickSortTime, stdSortTime;

	double beginClock, endClock;

	parseArgs (&myOCL, argc, argv, &NUM_ITERATIONS, pDeviceStr, pVendorStr, &widthReSz, &heightReSz, &bShowCL);

	printf("\n\n\n--------------------------------------------------------------------\n");
	
	uint arraySize = widthReSz*heightReSz;
	printf("Allocating array size of %d\n", arraySize);
#ifdef _MSC_VER 
	uint* pArray = (uint*)_aligned_malloc (((arraySize*sizeof(uint))/64 + 1)*64, 4096);
	uint* pArrayCopy = (uint*)_aligned_malloc (((arraySize*sizeof(uint))/64 + 1)*64, 4096);
#else // _MSC_VER
	uint* pArray = (uint*)aligned_alloc (4096, ((arraySize*sizeof(uint))/64 + 1)*64);
	uint* pArrayCopy = (uint*)aligned_alloc (4096, ((arraySize*sizeof(uint))/64 + 1)*64);
#endif // _MSC_VER
	// Maximum number of blocks formula adapted to Intel platforms
	// The heuristic is from Cederman/Tsigas paper
	// It leads to every block being QUICKSORT_BLOCK_SIZE for large input sizes, 
	// so MAXSEQ is approximately the number of workgroups for gqsort_kernel
	// For smaller input sizes (e.g. less than half a million elements), it bottoms out at 256 workgroups
	const size_t MAXSEQ = optp(arraySize, 0.00009516, 203);
	const size_t MAX_SIZE = 12*std::max(int(MAXSEQ), QUICKSORT_BLOCK_SIZE);

#ifdef _MSC_VER 
	work_record* pdone = (work_record*)_aligned_malloc (((MAX_SIZE*sizeof(work_record))/64 + 1)*64, 4096);
	work_record* pnews = (work_record*)_aligned_malloc (((MAX_SIZE*sizeof(work_record))/64 + 1)*64, 4096);
#else // _MSC_VER
	work_record* pdone = (work_record*)aligned_alloc (4096, ((MAX_SIZE*sizeof(work_record))/64 + 1)*64);
	work_record* pnews = (work_record*)aligned_alloc (4096, ((MAX_SIZE*sizeof(work_record))/64 + 1)*64);
#endif // _MSC_VER

	std::generate(pArray, pArray + arraySize, [](){static uint i = 0; return ++i; });
	std::random_shuffle(pArray, pArray + arraySize);
#ifdef RUN_CPU_SORTS
	std::cout << "Sorting the regular way..." << std::endl;
	std::copy(pArray, pArray + arraySize, pArrayCopy);

  beginClock = seconds();
	std::sort(pArrayCopy, pArrayCopy + arraySize);
  endClock = seconds();
	totalTime = endClock - beginClock;
	std::cout << "Time to sort: " << totalTime * 1000 << " ms" << std::endl;
	stdSortTime = totalTime;

	std::cout << "Sorting with parallel quicksort on the cpu: " << std::endl;
	std::copy(pArray, pArray + arraySize, pArrayCopy);

  beginClock = seconds();
	//quicksort(pArrayCopy, 0, arraySize-1);
  tbb::parallel_sort(pArrayCopy, pArrayCopy + arraySize);
  endClock = seconds();
	totalTime = endClock - beginClock;
	std::cout << "Time to sort: " << totalTime * 1000 << " ms" << std::endl;
	quickSortTime = totalTime;
#ifdef TRUST_BUT_VERIFY
	{
		std::vector<uint> verify(arraySize);
		std::copy(pArray, pArray + arraySize, verify.begin());

		std::cout << "verifying: ";
		std::sort(verify.begin(), verify.end());
		bool correct = std::equal(verify.begin(), verify.end(), pArrayCopy);
		unsigned int num_discrepancies = 0;
		if (!correct) {
			for(size_t i = 0; i < arraySize; i++) {
				if (verify[i] != pArrayCopy[i]) {
					//std:: cout << "discrepancy at " << i << " " << pArrayCopy[i] << " expected " << verify[i] << std::endl;
					num_discrepancies++;
				}
			}
		}
		std::cout << std::boolalpha << correct << std::endl;
		if (!correct) {
			char y;
			std::cout << "num_discrepancies: " << num_discrepancies << std::endl;
			std::cin >> y;
		}
	}
#endif
#endif // RUN_CPU_SORTS

	// Initialize OpenCL:
	bool bCPUDevice;
	InitializeOpenCL (pDeviceStr, pVendorStr, &myOCL.deviceID, &myOCL.contextHdl, &myOCL.cmdQHdl, bCPUDevice);
	if (bShowCL)
		QueryPrintOpenCLDeviceInfo (myOCL.deviceID, myOCL.contextHdl);	
  beginClock = seconds();
	CompileOpenCLProgram (bCPUDevice, myOCL.deviceID, myOCL.contextHdl, pSourceFileStr, &myOCL.programHdl);
  endClock = seconds();
	totalTime = endClock - beginClock;
	std::cout << "Time to build OpenCL Program: " << totalTime * 1000 << " ms" << std::endl;
	InstantiateOpenCLKernels (&myOCL, arraySize, MAXSEQ, MAX_SIZE, pdone, pnews, pArray);

	std::cout << "Sorting with GPUQSort on the " << pDeviceStr << ": " << std::endl;
	std::vector<uint> original(arraySize);
	std::copy(pArray, pArray + arraySize, original.begin());

	std::vector<double> times;
	times.resize(NUM_ITERATIONS);
	double AverageTime = 0.0;
	uint num_failures = 0;
	for(uint k = 0; k < NUM_ITERATIONS; k++) {
		std::copy(original.begin(), original.end(), pArray);
		std::vector<uint> seqs;
		std::vector<uint> verify(arraySize);
		std::copy(pArray, pArray + arraySize, verify.begin());

    beginClock = seconds();
		GPUQSort(&myOCL, arraySize, pArray, pdone, pnews, MAX_SIZE);
                endClock = seconds();
		totalTime = endClock - beginClock;
		printf("%4d.", k);
		std::cout << " Time to sort: " << totalTime * 1000 << " ms" << std::endl;
		times[k] = totalTime;
		AverageTime += totalTime;
#ifdef TRUST_BUT_VERIFY
		std::cout << "verifying: ";
		std::sort(verify.begin(), verify.end());
		bool correct = std::equal(verify.begin(), verify.end(), pArray);
		unsigned int num_discrepancies = 0;
		if (!correct) {
			for(size_t i = 0; i < arraySize; i++) {
				if (verify[i] != pArray[i]) {
					//std:: cout << "discrepancy at " << i << " " << pArray[i] << " expected " << verify[i] << std::endl;
					num_discrepancies++;
				}
			}
		}
		std::cout << std::boolalpha << correct << std::endl;
		if (!correct) {
			std::cout << "num_discrepancies: " << num_discrepancies << std::endl;
			num_failures ++;
		}
#endif
	}
	std::cout << " Number of failures: " << num_failures << " out of " << NUM_ITERATIONS << std::endl;
	AverageTime = AverageTime/NUM_ITERATIONS;
	std::cout << "Average Time: " << AverageTime * 1000 << " ms" << std::endl;
	double stdDev = 0.0, minTime = 1000000.0, maxTime = 0.0;
	for(uint k = 0; k < NUM_ITERATIONS; k++) 
	{
		stdDev += (AverageTime - times[k])*(AverageTime - times[k]);
		minTime = std::min(minTime, times[k]);
		maxTime = std::max(maxTime, times[k]);
	}

	if (NUM_ITERATIONS > 1) {
		stdDev = sqrt(stdDev/(NUM_ITERATIONS - 1));
		std::cout << "Standard Deviation: " << stdDev * 1000 << std::endl;
		std::cout << "%error (3*stdDev)/Average: " << 3*stdDev / AverageTime * 100 << "%" << std::endl;
		std::cout << "min time: " << minTime * 1000 << " ms" << std::endl;
		std::cout << "max time: " << maxTime * 1000 << " ms" << std::endl;
	}

#ifdef RUN_CPU_SORTS
	std::cout << "Average speedup over CPU quicksort: " << quickSortTime/AverageTime << std::endl;
	std::cout << "Average speedup over CPU std::sort: " << stdSortTime/AverageTime << std::endl;
#endif // RUN_CPU_SORTS

	Cleanup(&myOCL, 0, 0, "-------done--------------------------------------------------------\n");
#ifdef _MSC_VER 
	_aligned_free(pArray);
	_aligned_free(pArrayCopy);
	_aligned_free(pnews);
	_aligned_free(pdone);
#else // _MSC_VER
  free(pArray);
  free(pArrayCopy);
  free(pnews);
  free(pdone); 
#endif // _MSC_VER

	return 0;
}
