/* ************************************************************************* *\
               INTEL CORPORATION PROPRIETARY INFORMATION
     This software is supplied under the terms of a license agreement or 
     nondisclosure agreement with Intel Corporation and may not be copied 
     or disclosed except in accordance with the terms of that agreement. 
        Copyright (C) 2014-2019 Intel Corporation. All Rights Reserved.
\* ************************************************************************* */

#ifndef QUICKSORT_H
#define QUICKSORT_H

#ifdef HOST
template <class T>
T median(T x1, T x2, T x3) {
	if (x1 < x2) {
		if (x2 < x3) {
			return x2;
		} else {
			if (x1 < x3) {
				return x3;
			} else {
				return x1;
			}
		}
	} else { // x1 >= x2
		if (x1 < x3) {
			return x1;
		} else { // x1 >= x3
			if (x2 < x3) {
				return x2;
			} else {
				return x3;
			}
		}
	}
}

template <class T> struct select_type_selector;

template <> struct select_type_selector<uint>
{
  typedef uint data_t;
};

template <> struct select_type_selector<float>
{
  typedef uint data_t;
};

template <> struct select_type_selector<double>
{
  typedef ulong data_t;
};

template <class T>
T median_select(T x1, T x2, T x3) {
	if (x1 < x2) {
		if (x2 < x3) {
			return x2;
		} else {
      return cl::sycl::select(x1, x3, typename select_type_selector<T>::data_t(x1 < x3));
		}
	} else { // x1 >= x2
		if (x1 < x3) {
			return x1;
		} else { // x1 >= x3
      return cl::sycl::select(x2, x3, typename select_type_selector<T>::data_t(x2 < x3));
		}
	}
}

#else // HOST
uint median(uint x1, uint x2, uint x3) {
	if (x1 < x2) {
		if (x2 < x3) {
			return x2;
		} else {
      return select(x1, x3, x1 < x3);
		}
	} else { // x1 >= x2
		if (x1 < x3) {
			return x1;
		} else { // x1 >= x3
      return select(x2, x3, x2 < x3);
		}
	}
}
#endif //HOST

#define TRUST_BUT_VERIFY 1
// Note that SORT_THRESHOLD should always be 2X LQSORT_LOCAL_WORKGROUP_SIZE due to the use of bitonic sort
// Always try LQSORT_LOCAL_WORKGROUP_SIZE to be 8X smaller than QUICKSORT_BLOCK_SIZE - then try everything else :)
#ifdef CPU_DEVICE
#define QUICKSORT_BLOCK_SIZE         1024 
#define GQSORT_LOCAL_WORKGROUP_SIZE   128 
#define LQSORT_LOCAL_WORKGROUP_SIZE   128 
#define SORT_THRESHOLD                256 
#else
#ifdef NVIDIA_GPU
// best for NVidia; 
#define QUICKSORT_BLOCK_SIZE         1024 
#define GQSORT_LOCAL_WORKGROUP_SIZE   128
#define LQSORT_LOCAL_WORKGROUP_SIZE   256 
#define SORT_THRESHOLD                512 
#else // NVIDIA_GPU
// best for Intel; 
#define QUICKSORT_BLOCK_SIZE         1728 
#define GQSORT_LOCAL_WORKGROUP_SIZE   256 
#define LQSORT_LOCAL_WORKGROUP_SIZE   128 
#define SORT_THRESHOLD                256 
#endif // NVIDIA_GPU
#endif

#define EMPTY_RECORD             42

// work record contains info about the part of array that is still longer than QUICKSORT_BLOCK_SIZE and 
// therefore cannot be processed by lqsort_kernel yet. It contins the start and the end indexes into 
// an array to be sorted, associated pivot and direction of the sort. 

template <class T>
struct work_record {
	uint start;
	uint end;
	T    pivot;
	uint direction;

	work_record() : 
		start(0), end(0), pivot(T(0)), direction(EMPTY_RECORD) {}
	work_record(uint s, uint e, T p, uint d) : 
		start(s), end(e), pivot(p), direction(d) {}
};


// parent record contains everything kernels need to know about the parent of a set of blocks:
// initially, the first two fields equal to the third and fourth fields respectively
// blockcount contains the total number of blocks associated with the parent.
// During processing, sstart and send get incremented. At the end of gqsort_kernel, all the 
// parent record fields are used to calculate new pivots and new work records.
typedef struct parent_record {
	uint sstart, send, oldstart, oldend, blockcount; 
    parent_record() :
	    sstart(0), send(0), oldstart(0), oldend(0), blockcount(0) {}
	parent_record(uint ss, uint se, uint os, uint oe, uint bc) : 
		sstart(ss), send(se), oldstart(os), oldend(oe), blockcount(bc) {}
} parent_record;

// block record contains everything kernels needs to know about the block:
// start and end indexes into input array, pivot, direction of sorting and the parent record index
template <class T>
struct block_record {
	uint start;
	uint end;
	T    pivot;
	uint direction;
	uint parent;
	block_record() : start(0), end(0), pivot(T(0)), direction(EMPTY_RECORD), parent(0) {}
	block_record(uint s, uint e, T p, uint d, uint prnt) : 
		start(s), end(e), pivot(p), direction(d), parent(prnt) {}
};
#endif // QUICKSORT_H
