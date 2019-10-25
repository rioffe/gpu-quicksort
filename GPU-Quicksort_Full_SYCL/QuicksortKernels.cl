/* ************************************************************************* *\
               INTEL CORPORATION PROPRIETARY INFORMATION
     This software is supplied under the terms of a license agreement or 
     nondisclosure agreement with Intel Corporation and may not be copied 
     or disclosed except in accordance with the terms of that agreement. 
        Copyright (C) 2014-2019 Intel Corporation. All Rights Reserved.
\* ************************************************************************* */

#include "Quicksort.h"

void plus_prescan(local uint *a, local uint *b) {
    uint av = *a;
	uint bv = *b;
    *a = bv;
    *b = bv + av;
}

/// bitonic_sort: sort 2*LOCAL_THREADCOUNT elements
void bitonic_sort(local uint* sh_data, const uint localid)
{
	for (uint ulevel = 1; ulevel < LQSORT_LOCAL_WORKGROUP_SIZE; ulevel <<= 1) {
        for (uint j = ulevel; j > 0; j >>= 1) {
            uint pos = 2*localid - (localid & (j - 1));

			uint direction = localid & ulevel;
			uint av = sh_data[pos], bv = sh_data[pos + j];
			const bool sortThem = av > bv;
			const uint greater = select(bv, av, sortThem);
			const uint lesser  = select(av, bv, sortThem);

			sh_data[pos]     = select(lesser, greater, direction);
			sh_data[pos + j] = select(greater, lesser, direction);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

	for (uint j = LQSORT_LOCAL_WORKGROUP_SIZE; j > 0; j >>= 1) {
        uint pos = 2*localid - (localid & (j - 1));

		uint av = sh_data[pos], bv = sh_data[pos + j];
		const bool sortThem = av > bv;
		sh_data[pos]      = select(av, bv, sortThem);
		sh_data[pos + j]  = select(bv, av, sortThem);

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

void sort_threshold(local uint* data_in, global uint* data_out,
					uint start, 
					uint end, local uint* temp, uint localid) 
{
	uint tsum = end - start;
	if (tsum == SORT_THRESHOLD) {
		bitonic_sort(data_in+start, localid);
		for (uint i = localid; i < SORT_THRESHOLD; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
			data_out[start + i] = data_in[start + i];
		}
	} else if (tsum > 1) {
		for (uint i = localid; i < SORT_THRESHOLD; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
			if (i < tsum) {
				temp[i] = data_in[start + i];
			} else {
				temp[i] = UINT_MAX;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		bitonic_sort(temp, localid);

		for (uint i = localid; i < tsum; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
			data_out[start + i] = temp[i];
		}
	} else if (tsum == 1 && localid == 0) {
		data_out[start] = data_in[start];
	} 
}


// record to push start of the sequence, end of the sequence and direction of sorting on internal stack
typedef struct workstack_record {
	uint start;
	uint end;
	uint direction;
} workstack_record;

#define PUSH(START, END) 			if (localid == 0) { \
										++workstack_pointer; \
                                        workstack_record wr = { (START), (END), direction ^ 1 }; \
										workstack[workstack_pointer] = wr; \
									} \
									barrier(CLK_LOCAL_MEM_FENCE);


//---------------------------------------------------------------------------------------
// Kernel implements the last stage of GPU-Quicksort, when all the subsequences are small
// enough to be processed in local memory. It uses similar algorithm to gqsort_kernel to 
// move items around the pivot and then switches to bitonic sort for sequences in
// the range [1, SORT_THRESHOLD] 
//
// d - input array
// dn - scratch array of the same size as the input array
// seqs - array of records to be sorted in a local memory, one sequence per work group.
//---------------------------------------------------------------------------------------
kernel void lqsort_kernel(global uint* d, global uint* dn, global work_record* seqs) 
{
	const uint blockid    = get_group_id(0);
	const uint localid    = get_local_id(0);

	// workstack: stores the start and end of the sequences, direction of sort
	// If the sequence is less that SORT_THRESHOLD, it gets sorted. 
	// It will only be pushed on the stack if it greater than the SORT_THRESHOLD. 
	// Note, that the sum of ltsum + gtsum is less than QUICKSORT_BLOCK_SIZE. 
	// The total sum of the length of records on the stack cannot exceed QUICKSORT_BLOCK_SIZE, 
	// but each individual record should be greater than SORT_THRESHOLD, so the maximum length 
	// of the stack is QUICKSORT_BLOCK_SIZE/SORT_THRESHOLD - in the case of BDW GT2 the length 
	// of the stack is 2 :)
	local workstack_record workstack[QUICKSORT_BLOCK_SIZE/SORT_THRESHOLD]; 
	local int workstack_pointer;

	local uint mys[QUICKSORT_BLOCK_SIZE], mysn[QUICKSORT_BLOCK_SIZE], temp[SORT_THRESHOLD];
	local uint *s, *sn;
  local uint ltsum, gtsum;
	local uint lt[LQSORT_LOCAL_WORKGROUP_SIZE+1], gt[LQSORT_LOCAL_WORKGROUP_SIZE+1];
	uint i, tmp, ltp, gtp;
	
	work_record block = seqs[blockid];
	const uint d_offset = block.start;
	uint start = 0; 
	uint end   = block.end - d_offset;

	uint direction = 1; // which direction to sort
	// initialize workstack and workstack_pointer: push the initial sequence on the stack
	if (localid == 0) {
		workstack_pointer = 0; // beginning of the stack
		workstack_record wr = { start, end, direction };
		workstack[0] = wr;
	}
	// copy block of data to be sorted by one workgroup into local memory
	// note that indeces of local data go from 0 to end-start-1
	if (block.direction == 1) {
		for (i = localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
			mys[i] = d[i+d_offset];
		}
	} else {
		for (i = localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
			mys[i] = dn[i+d_offset];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	while (workstack_pointer >= 0) { 
		// pop up the stack
		workstack_record wr = workstack[workstack_pointer];
		start = wr.start;
		end = wr.end;
		direction = wr.direction;
		barrier(CLK_LOCAL_MEM_FENCE);
		if (localid == 0) {
			--workstack_pointer;

			ltsum = gtsum = 0;	
		}
		if (direction == 1) {
			s = mys;
			sn = mysn;
		} else {
			s = mysn;
			sn = mys;
		}
		// Set thread local counters to zero
		lt[localid] = gt[localid] = 0;
		ltp = gtp = 0;
		barrier(CLK_LOCAL_MEM_FENCE);

		// Pick a pivot
		uint pivot = s[start];
		if (start < end) {
			pivot = median(pivot, s[(start+end) >> 1], s[end-1]);
		}
		// Align work item accesses for coalesced reads.
		// Go through data...
		for(i = start + localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
			tmp = s[i];
			// counting elements that are smaller ...
			if (tmp < pivot)
				ltp++;
			// or larger compared to the pivot.
			if (tmp > pivot) 
				gtp++;
		}
		lt[localid] = ltp;
		gt[localid] = gtp;
		barrier(CLK_LOCAL_MEM_FENCE);
		
		// calculate cumulative sums
		uint n;
		for(i = 1; i < LQSORT_LOCAL_WORKGROUP_SIZE; i <<= 1) {
			n = 2*i - 1;
			if ((localid & n) == n) {
				lt[localid] += lt[localid-i];
				gt[localid] += gt[localid-i];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		if ((localid & n) == n) {
			lt[LQSORT_LOCAL_WORKGROUP_SIZE] = ltsum = lt[localid];
			gt[LQSORT_LOCAL_WORKGROUP_SIZE] = gtsum = gt[localid];
			lt[localid] = 0;
			gt[localid] = 0;
		}
		
		for(i = LQSORT_LOCAL_WORKGROUP_SIZE/2; i >= 1; i >>= 1) {
			n = 2*i - 1;
			if ((localid & n) == n) {
				plus_prescan(&lt[localid - i], &lt[localid]);
				plus_prescan(&gt[localid - i], &gt[localid]);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		// Allocate locations for work items
		uint lfrom = start + lt[localid];
		uint gfrom = end - gt[localid+1];

		// go thru data again writing elements to their correct position
		for (i = start + localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
			tmp = s[i];
			// increment counts
			if (tmp < pivot) 
				sn[lfrom++] = tmp;
			
			if (tmp > pivot) 
				sn[gfrom++] = tmp;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// Store the pivot value between the new sequences
		for (i = start + ltsum + localid;i < end - gtsum; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
			d[i+d_offset] = pivot;
		}
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		// if the sequence is shorter than SORT_THRESHOLD
		// sort it using an alternative sort and place result in d
		if (ltsum <= SORT_THRESHOLD) {
			sort_threshold(sn, d+d_offset, start, start + ltsum, temp, localid);
		} else {
			PUSH(start, start + ltsum);
		}
		
		if (gtsum <= SORT_THRESHOLD) {
			sort_threshold(sn, d+d_offset, end - gtsum, end, temp, localid);
		} else {
			PUSH(end - gtsum, end);
		}
	}
}
