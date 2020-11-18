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
} OCLResources;

// Globals:
cl_int		ciErrNum;

static cl_kernel gqsort_kernel, lqsort_kernel;

void Cleanup(OCLResources* pOCL, int iExitCode, bool bExit, const char* optionalErrorMessage)
{
	if (optionalErrorMessage) 
		printf ("%s\n", optionalErrorMessage);

	memset(pOCL, 0, sizeof (OCLResources));

	if (pOCL->programHdl)		{ clReleaseProgram(pOCL->programHdl);		pOCL->programHdl=NULL;	}
	if (pOCL->cmdQHdl)			{ clReleaseCommandQueue(pOCL->cmdQHdl);		pOCL->cmdQHdl=NULL;		}
	if (pOCL->contextHdl)		{ clReleaseContext(pOCL->contextHdl);		pOCL->contextHdl= NULL;	}

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

void InstantiateOpenCLKernels(OCLResources *pOCL)
{	
	// Instantiate kernels:
	gqsort_kernel = clCreateKernel(pOCL->programHdl, "gqsort_kernel", &ciErrNum);
	CheckCLError (ciErrNum, "Kernel creation failed.", "Kernel created.");
	lqsort_kernel = clCreateKernel(pOCL->programHdl, "lqsort_kernel", &ciErrNum);
	CheckCLError (ciErrNum, "Kernel creation failed.", "Kernel created.");
}


//#define GET_DETAILED_PERFORMANCE 1
#define RUN_CPU_SORTS
#define HOST 1
#include "Quicksort.h"


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

template <class T>
void gqsort(OCLResources *pOCL, std::vector<block_record>& blocks, std::vector<parent_record>& parents, std::vector<work_record>& news, bool reset) {
	news.resize(blocks.size()*2);

	size_t		dimNDR[2] = { 0, 0};
	size_t		dimWG[2] = { 0, 0 };

	// Create buffer objects for memory.
	cl_mem blocksb = clCreateBuffer(pOCL->contextHdl, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(block_record)*blocks.size(), &blocks[0], &ciErrNum);
	CheckCLError (ciErrNum, "clCreateBuffer failed.", "clCreateBuffer.");
	cl_mem parentsb = clCreateBuffer(pOCL->contextHdl, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(parent_record)*parents.size(), &parents[0], &ciErrNum);
	CheckCLError (ciErrNum, "clCreateBuffer failed.", "clCreateBuffer.");
	cl_mem newsb = clCreateBuffer(pOCL->contextHdl, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(work_record)*news.size(), &news[0], &ciErrNum);
	CheckCLError (ciErrNum, "clCreateBuffer failed.", "clCreateBuffer.");
	

	ciErrNum |= clSetKernelArg(gqsort_kernel, 2, sizeof(cl_mem), (void*) &blocksb);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");
	ciErrNum |= clSetKernelArg(gqsort_kernel, 3, sizeof(cl_mem), (void*) &parentsb);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");
	ciErrNum |= clSetKernelArg(gqsort_kernel, 4, sizeof(cl_mem), (void*) &newsb);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");

	//std::cout << "blocks size is " << blocks.size() << std::endl;
#ifdef GET_DETAILED_PERFORMANCE
	static double absoluteTotal = 0.0;
	static uint count = 0;

	if (reset) {
		absoluteTotal = 0.0;
		count = 0;
	}

	double beginClock, endClock;
  beginClock = seconds();
#endif
	// Lets do phase 1 pass
	dimNDR[0] = GQSORT_LOCAL_WORKGROUP_SIZE * blocks.size();
	dimWG[0] = GQSORT_LOCAL_WORKGROUP_SIZE;

	ciErrNum = clEnqueueNDRangeKernel (pOCL->cmdQHdl, gqsort_kernel, 1, NULL, dimNDR, dimWG, 0, NULL, 0);
	CheckCLError(ciErrNum, "clEnqueueNDRangeKernel failed.", "clEnqueueNDRangeKernel");
	ciErrNum = clEnqueueReadBuffer(pOCL->cmdQHdl, newsb, CL_TRUE, 0, sizeof(work_record)*news.size(), &news[0], 0, NULL, NULL);
	CheckCLError(ciErrNum, "clEnqueueReadBuffer failed.", "clEnqueueReadBuffer");

#ifdef GET_DETAILED_PERFORMANCE
  endClock = seconds();
	double totalTime = endClock - beginClock;
	absoluteTotal += totalTime;
	std::cout << ++count << ": gqsort time " << absoluteTotal * 1000 << " ms" << std::endl;
#endif
	clReleaseMemObject(blocksb);
	clReleaseMemObject(parentsb);
	clReleaseMemObject(newsb);
}

template <class T>
void lqsort(OCLResources *pOCL, std::vector<work_record>& done, cl_mem tempb) {
	size_t		dimNDR[2] = { 0, 0};
	size_t		dimWG[2] = { 0, 0 };

	//std::cout << "done size is " << done.size() << std::endl; 
	cl_mem doneb = clCreateBuffer(pOCL->contextHdl, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(work_record)*done.size(), &done[0], &ciErrNum);
	CheckCLError (ciErrNum, "clCreateBuffer failed.", "clCreateBuffer.");
	
	ciErrNum |= clSetKernelArg(lqsort_kernel, 2, sizeof(cl_mem), (void*) &doneb);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");

#ifdef GET_DETAILED_PERFORMend
	double beginClock, endClock;
  beginClock = seconds();
#endif
	// Lets do phase 2 pass
	dimNDR[0] = LQSORT_LOCAL_WORKGROUP_SIZE * (done.size());
	dimWG[0] = LQSORT_LOCAL_WORKGROUP_SIZE;
	ciErrNum = clEnqueueNDRangeKernel (pOCL->cmdQHdl, lqsort_kernel, 1, NULL, dimNDR, dimWG, 0, NULL, 0);
	CheckCLError(ciErrNum, "clEnqueueNDRangeKernel failed.", "clEnqueueNDRangeKernel");

    ciErrNum = clFlush(pOCL->cmdQHdl);
	CheckCLError(ciErrNum, "clFlush failed.", "clFlush");
	ciErrNum = clFinish(pOCL->cmdQHdl);
    CheckCLError(ciErrNum, "clFinish failed.", "clFinish");

#ifdef GET_DETAILED_PERFORMANCE
	endClock = seconds();
	double totalTime = endClock - beginClock;
	std::cout << "lqsort time " << totalTime * 1000 << " ms" << std::endl;
#endif
	clReleaseMemObject(doneb);
}

size_t optp(size_t s, double k, size_t m) {
	return (size_t)pow(2, floor(log(s*k + m)/log(2.0) + 0.5));
}

template <class T>
void GPUQSort(OCLResources *pOCL, size_t size, T* d, T* dn)  {
	// allocate buffers
	cl_mem db = clCreateBuffer(pOCL->contextHdl, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, ((sizeof(T)*size)/64 + 1)*64, d, &ciErrNum);
	CheckCLError (ciErrNum, "clCreateBuffer failed.", "clCreateBuffer.");
	cl_mem dnb = clCreateBuffer(pOCL->contextHdl, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, ((sizeof(T)*size)/64 + 1)*64, dn, &ciErrNum);
	CheckCLError (ciErrNum, "clCreateBuffer failed.", "clCreateBuffer.");

	ciErrNum |= clSetKernelArg(gqsort_kernel, 0, sizeof(cl_mem), (void*) &db);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");
	ciErrNum |= clSetKernelArg(gqsort_kernel,	1, sizeof(cl_mem), (void*) &dnb);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");

	ciErrNum |= clSetKernelArg(lqsort_kernel, 0, sizeof(cl_mem), (void*) &db);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");
	ciErrNum |= clSetKernelArg(lqsort_kernel,	1, sizeof(cl_mem), (void*) &dnb);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");

	const size_t MAXSEQ = optp(size, 0.00009516, 203);
	const size_t MAX_SIZE = 12*std::max(MAXSEQ, (size_t)QUICKSORT_BLOCK_SIZE);
	//std::cout << "MAXSEQ = " << MAXSEQ << std::endl;
	uint startpivot = median(d[0], d[size/2], d[size-1]);
	std::vector<work_record> work, done, news;
	work.reserve(MAX_SIZE);
	done.reserve(MAX_SIZE);
	news.reserve(MAX_SIZE);
	std::vector<parent_record> parent_records;
	parent_records.reserve(MAX_SIZE);
	std::vector<block_record> blocks;
	blocks.reserve(MAX_SIZE);
	
	work.push_back(work_record(0, size, startpivot, 1));

	bool reset = true;

	while(!work.empty() /*&& work.size() + done.size() < MAXSEQ*/) {
		size_t blocksize = 0;
		
		for(auto it = work.begin(); it != work.end(); ++it) {
			blocksize += std::max((it->end - it->start)/MAXSEQ, (size_t)1);
		}
		for(auto it = work.begin(); it != work.end(); ++it) {
			uint start = it->start;
			uint end   = it->end;
			uint pivot = it->pivot;
			uint direction = it->direction;
			uint blockcount = (end - start + blocksize - 1)/blocksize;
			parent_record prnt(start, end, start, end, blockcount-1);
			parent_records.push_back(prnt);

			for(uint i = 0; i < blockcount - 1; i++) {
				uint bstart = start + blocksize*i;
				block_record br(bstart, bstart+blocksize, pivot, direction, parent_records.size()-1);
				blocks.push_back(br);
			}
			block_record br(start + blocksize*(blockcount - 1), end, pivot, direction, parent_records.size()-1);
			blocks.push_back(br);
		}

		gqsort<T>(pOCL, blocks, parent_records, news, reset);
		reset = false;
		//std::cout << " blocks = " << blocks.size() << " parent records = " << parent_records.size() << " news = " << news.size() << std::endl;
		work.clear();
		parent_records.clear();
		blocks.clear();
		for(auto it = news.begin(); it != news.end(); ++it) {
			if (it->direction != EMPTY_RECORD) {
				if (it->end - it->start <= QUICKSORT_BLOCK_SIZE /*size/MAXSEQ*/) {
					if (it->end - it->start > 0)
						done.push_back(*it);
				} else {
					work.push_back(*it);
				}
			}
		}
		news.clear();
	}
	for(auto it = work.begin(); it != work.end(); ++it) {
		if (it->end - it->start > 0)
			done.push_back(*it);
	}

	lqsort<T>(pOCL, done, db);

	// release buffers: we are done
	clReleaseMemObject(db);
	clReleaseMemObject(dnb);
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
	InstantiateOpenCLKernels (&myOCL);

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
		GPUQSort(&myOCL, arraySize, pArray, pArrayCopy);
		endClock = seconds();
		totalTime = endClock - beginClock;
		std::cout << "Time to sort: " << totalTime * 1000 << " ms" << std::endl;
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

	printf("-------done--------------------------------------------------------\n");
#ifdef _MSC_VER
	_aligned_free(pArray);
	_aligned_free(pArrayCopy);
#else // _MSC_VER
  free(pArray);
  free(pArrayCopy);
#endif // _MSC_VER

	return 0;
}
