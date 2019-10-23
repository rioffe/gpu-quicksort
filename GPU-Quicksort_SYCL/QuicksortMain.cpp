/* ************************************************************************* *\
               INTEL CORPORATION PROPRIETARY INFORMATION
     This software is supplied under the terms of a license agreement or 
     nondisclosure agreement with Intel Corporation and may not be copied 
     or disclosed except in accordance with the terms of that agreement. 
        Copyright (C) 2014-2019 Intel Corporation. All Rights Reserved.
\* ************************************************************************* */

// QuicksortMain.cpp : Defines the entry point for the console application.
//
#include <CL/sycl.hpp>

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

#ifndef _MSC_VER
// Linux
#include "tbb/parallel_sort.h"
using namespace tbb;
#endif
using namespace cl::sycl;

/* Classes can inherit from the device_selector class to allow users
 * to dictate the criteria for choosing a device from those that might be
 * present on a system. This example looks for a device with SPIR support
 * and prefers GPUs over CPUs. */
class intel_gpu_selector : public device_selector {
 public:
  intel_gpu_selector() : device_selector() {}

  /* The selection is performed via the () operator in the base
   * selector class.This method will be called once per device in each
   * platform. Note that all platforms are evaluated whenever there is
   * a device selection. */
  int operator()(const device& device) const override {
    /* We only give a valid score to devices that support SPIR. */
    //if (device.has_extension(cl::sycl::string_class("cl_khr_spir"))) {
    if (device.get_info<info::device::name>().find("Intel") != std::string::npos) {
      if (device.get_info<info::device::device_type>() ==
          info::device_type::gpu) {
        return 50;
      }
    }
    /* Devices with a negative score will never be chosen. */
    return -1;
  }
};

// Types:
typedef unsigned int uint;

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
	cl::sycl::queue     queue;
} OCLResources;

// Globals:
/* Create variable to store OpenCL errors. */
::cl_int		ciErrNum = 0;

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
  std::string pDeviceWStr;
  std::string pVendorWStr;
  const char sUsageString[512] = "Usage: Quicksort [num test iterations] [cpu|gpu] [intel|amd|nvidia] [SurfWidth(^2 only)] [SurfHeight(^2 only)] [show_CL | no_show_CL]";
  
  if (argc != 7)
  {
  	Cleanup (pOCL, -1, true, sUsageString);
  }
  else
  {
  	*test_iterations	= atoi (argv[1]);
  	pDeviceWStr			= std::string(argv[2]);			// "cpu" or "gpu"	
  	pVendorWStr			= std::string(argv[3]);			// "intel" or "amd" or "nvidia"
  	*widthReSz	= atoi (argv[4]);
  	*heightReSz	= atoi (argv[5]);
  	if (argv[6][0]=='s')
  		*pbShowCL = true;
  	else
  		*pbShowCL = false;
  }
  sprintf (pDeviceStr, "%s", pDeviceWStr.c_str());
  sprintf (pVendorStr, "%s", pVendorWStr.c_str());

  auto get_queue = [&pDeviceStr, &pVendorStr]() {  
    device_selector* pds;
    if (pVendorStr == std::string("intel")) {
      if (pDeviceStr == std::string("gpu")) {
          static intel_gpu_selector selector;
		  pds = &selector;
	  } else if (pDeviceStr == std::string("cpu")) {
		  static cpu_selector selector;
		  pds = &selector;
	  }
	} else {
		static default_selector selector;
		pds = &selector;
	}

    device d(*pds);

    queue queue(*pds, [](cl::sycl::exception_list l) {
      for (auto ep : l) {
        try {
          std::rethrow_exception(ep);
        } catch (cl::sycl::exception e) {
          std::cout << e.what() << std::endl;
        }
      }
    });
    return queue;
  };
  
  auto queue = get_queue();
  pOCL->queue = queue;
  /* Retrieve the underlying cl_context of the context associated with the
   * queue. */
  pOCL->contextHdl = queue.get_context().get();

  /* Retrieve the underlying cl_device_id of the device asscociated with the
   * queue. */
  pOCL->deviceID = queue.get_device().get();

  /* Retrieve the underlying cl_command_queue of the queue. */
  pOCL->cmdQHdl = queue.get();
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
void gqsort(OCLResources *pOCL, buffer<T, 1>& d_buffer, buffer<T, 1>& dn_buffer, std::vector<block_record>& blocks, std::vector<parent_record>& parents, std::vector<work_record>& news, bool reset) {
	news.resize(blocks.size()*2);

	// Create buffer objects for memory.
	buffer<block_record, 1>  blocks_buffer(blocks.data(), range<1>(blocks.size()));
	buffer<parent_record, 1>  parents_buffer(parents.data(), range<1>(parents.size()));
	buffer<work_record, 1>  news_buffer(news.data(), range<1>(news.size()));

    kernel sycl_gqsort_kernel(gqsort_kernel, pOCL->queue.get_context());

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

    pOCL->queue.submit([&](handler& cgh) {
	  auto db = d_buffer.template get_access<access::mode::discard_read_write>(cgh);
	  auto dnb = dn_buffer.template get_access<access::mode::discard_read_write>(cgh);
	  auto blocksb = blocks_buffer.template get_access<access::mode::discard_read_write>(cgh);
	  auto parentsb = parents_buffer.template get_access<access::mode::discard_read_write>(cgh);
	  auto newsb = news_buffer.template get_access<access::mode::read_write>(cgh);
      /* Normally, SYCL sets kernel arguments for the user. However, when
       * using the interoperability features, it is unable to do this and
       * the user must set the arguments manually. */
      cgh.set_arg(0, db);
      cgh.set_arg(1, dnb);
      cgh.set_arg(2, blocksb);
      cgh.set_arg(3, parentsb);
      cgh.set_arg(4, newsb);

      cgh.parallel_for(nd_range<1>(range<1>(GQSORT_LOCAL_WORKGROUP_SIZE * (blocks.size())), 
	                               range<1>(GQSORT_LOCAL_WORKGROUP_SIZE)), 
	    sycl_gqsort_kernel);
    });
    pOCL->queue.wait_and_throw();

#ifdef GET_DETAILED_PERFORMANCE
  endClock = seconds();
	double totalTime = endClock - beginClock;
	absoluteTotal += totalTime;
	std::cout << ++count << ": gqsort time " << absoluteTotal * 1000 << " ms" << std::endl;
#endif
}

template <class T>
void lqsort(OCLResources *pOCL, std::vector<work_record>& done, buffer<T, 1>& d_buffer, buffer<T, 1>& dn_buffer, T* d, size_t size) {
	buffer<work_record, 1>  done_buffer(done.data(), range<1>(done.size()));
	
#ifdef GET_DETAILED_PERFORMANCE
    double beginClock, endClock;
    beginClock = seconds();
#endif
    kernel sycl_lqsort_kernel(lqsort_kernel, pOCL->queue.get_context());

    pOCL->queue.submit([&](handler& cgh) {
      auto db = d_buffer.template get_access<access::mode::read_write>(cgh);
	  auto dnb = dn_buffer.template get_access<access::mode::discard_read_write>(cgh);
      auto doneb = done_buffer.get_access<access::mode::discard_read_write>(cgh);
      /* Normally, SYCL sets kernel arguments for the user. However, when
       * using the interoperability features, it is unable to do this and
       * the user must set the arguments manually. */
      cgh.set_arg(0, db);
      cgh.set_arg(1, dnb);
      cgh.set_arg(2, doneb);

      cgh.parallel_for(nd_range<1>(range<1>(LQSORT_LOCAL_WORKGROUP_SIZE * (done.size())), 
	                               range<1>(LQSORT_LOCAL_WORKGROUP_SIZE)), 
	    sycl_lqsort_kernel);
    });
    pOCL->queue.wait_and_throw();

#ifdef GET_DETAILED_PERFORMANCE
	endClock = seconds();
	double totalTime = endClock - beginClock;
	std::cout << "lqsort time " << totalTime * 1000 << " ms" << std::endl;
#endif
}

size_t optp(size_t s, double k, size_t m) {
	return (size_t)pow(2, floor(log(s*k + m)/log(2.0) + 0.5));
}

template <class T>
void GPUQSort(OCLResources *pOCL, size_t size, T* d, T* dn)  {
	// allocate buffers
	buffer<T, 1>  d_buffer(d, range<1>(size));
	buffer<T, 1>  dn_buffer(dn, range<1>(size));

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

		gqsort<T>(pOCL, d_buffer, dn_buffer, blocks, parent_records, news, reset);
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

	lqsort<T>(pOCL, done, d_buffer, dn_buffer, d, size);
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
  parallel_sort(pArrayCopy, pArrayCopy + arraySize);
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
	bool bCPUDevice = false;
	//InitializeOpenCL (pDeviceStr, pVendorStr, &myOCL.deviceID, &myOCL.contextHdl, &myOCL.cmdQHdl, bCPUDevice);
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
