/*
Copyright (c) 2014-2024, Intel Corporation
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

#include "QuicksortTypes.h"

// ============================================================================
// Utility functions
// ============================================================================

// Median of three — used for pivot selection
template <typename T>
inline T median3(T x1, T x2, T x3) {
    if (x1 < x2) {
        if (x2 < x3) return x2;
        else return select(x1, x3, x1 < x3);
    } else {
        if (x1 < x3) return x1;
        else return select(x2, x3, x2 < x3);
    }
}

// ============================================================================
// Blelloch exclusive prefix scan over threadgroup
// Requires WG_SIZE to be a power of 2.
// Input: scan_data[localid] = per-thread value
// Output: scan_data[localid] = exclusive prefix sum, total stored in scan_data[WG_SIZE]
// ============================================================================
template <uint WG_SIZE>
inline void exclusive_scan(threadgroup uint* scan_data, uint localid) {
    // Up-sweep (reduce)
    uint n = 0;
    for (uint i = 1; i < WG_SIZE; i <<= 1) {
        n = 2 * i - 1;
        if ((localid & n) == n) {
            scan_data[localid] += scan_data[localid - i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store total and clear last element
    if ((localid & n) == n) {
        scan_data[WG_SIZE] = scan_data[localid];
        scan_data[localid] = 0;
    }

    // Down-sweep
    for (uint i = WG_SIZE / 2; i >= 1; i >>= 1) {
        n = 2 * i - 1;
        if ((localid & n) == n) {
            uint tmp = scan_data[localid - i];
            scan_data[localid - i] = scan_data[localid];
            scan_data[localid] += tmp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Inclusive scan: exclusive scan result + original element
// Returns the inclusive scan value for this thread; total is in scan_data[WG_SIZE]
template <uint WG_SIZE>
inline uint inclusive_scan(threadgroup uint* scan_data, uint localid, uint original_val) {
    exclusive_scan<WG_SIZE>(scan_data, localid);
    return scan_data[localid] + original_val;
}

// ============================================================================
// Bitonic sort for SORT_THRESHOLD elements in threadgroup memory
// Each thread handles 2 positions (SORT_THRESHOLD = 2 * LQSORT_LOCAL_WORKGROUP_SIZE)
// ============================================================================
template <typename T>
inline void bitonic_sort(threadgroup T* sh_data, uint localid) {
    for (uint ulevel = 1; ulevel < LQSORT_LOCAL_WORKGROUP_SIZE; ulevel <<= 1) {
        for (uint j = ulevel; j > 0; j >>= 1) {
            uint pos = 2 * localid - (localid & (j - 1));
            uint dir = localid & ulevel;
            T av = sh_data[pos], bv = sh_data[pos + j];
            bool sortThem = av > bv;
            T greater = select(bv, av, sortThem);
            T lesser  = select(av, bv, sortThem);
            sh_data[pos]     = select(lesser, greater, bool(dir));
            sh_data[pos + j] = select(greater, lesser, bool(dir));
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint j = LQSORT_LOCAL_WORKGROUP_SIZE; j > 0; j >>= 1) {
        uint pos = 2 * localid - (localid & (j - 1));
        T av = sh_data[pos], bv = sh_data[pos + j];
        bool sortThem = av > bv;
        sh_data[pos]     = select(av, bv, sortThem);
        sh_data[pos + j] = select(bv, av, sortThem);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Sort a small sequence using bitonic sort, writing results to global output
template <typename T>
inline void sort_threshold(threadgroup T* data_in, device T* data_out,
                           uint start, uint end,
                           threadgroup T* temp, uint localid) {
    uint tsum = end - start;
    if (tsum == SORT_THRESHOLD) {
        bitonic_sort(data_in + start, localid);
        for (uint i = localid; i < SORT_THRESHOLD; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
            data_out[start + i] = data_in[start + i];
        }
    } else if (tsum > 1) {
        for (uint i = localid; i < SORT_THRESHOLD; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
            if (i < tsum) {
                temp[i] = data_in[start + i];
            } else {
                // Pad with max value so bitonic sort puts them at the end
                temp[i] = T(UINT_MAX);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bitonic_sort(temp, localid);
        for (uint i = localid; i < tsum; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
            data_out[start + i] = temp[i];
        }
    } else if (tsum == 1 && localid == 0) {
        data_out[start] = data_in[start];
    }
}

// Float specialization for max value padding
template <>
inline void sort_threshold<float>(threadgroup float* data_in, device float* data_out,
                                  uint start, uint end,
                                  threadgroup float* temp, uint localid) {
    uint tsum = end - start;
    if (tsum == SORT_THRESHOLD) {
        bitonic_sort(data_in + start, localid);
        for (uint i = localid; i < SORT_THRESHOLD; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
            data_out[start + i] = data_in[start + i];
        }
    } else if (tsum > 1) {
        for (uint i = localid; i < SORT_THRESHOLD; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
            temp[i] = (i < tsum) ? data_in[start + i] : HUGE_VALF;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bitonic_sort(temp, localid);
        for (uint i = localid; i < tsum; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
            data_out[start + i] = temp[i];
        }
    } else if (tsum == 1 && localid == 0) {
        data_out[start] = data_in[start];
    }
}

// Half specialization for max value padding
template <>
inline void sort_threshold<half>(threadgroup half* data_in, device half* data_out,
                                 uint start, uint end,
                                 threadgroup half* temp, uint localid) {
    uint tsum = end - start;
    if (tsum == SORT_THRESHOLD) {
        bitonic_sort(data_in + start, localid);
        for (uint i = localid; i < SORT_THRESHOLD; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
            data_out[start + i] = data_in[start + i];
        }
    } else if (tsum > 1) {
        for (uint i = localid; i < SORT_THRESHOLD; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
            temp[i] = (i < tsum) ? data_in[start + i] : HALF_MAX;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bitonic_sort(temp, localid);
        for (uint i = localid; i < tsum; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
            data_out[start + i] = temp[i];
        }
    } else if (tsum == 1 && localid == 0) {
        data_out[start] = data_in[start];
    }
}

// ============================================================================
// GQSORT KERNEL — Global Quicksort
// Partitions large blocks around a pivot using work-group parallel scan.
// Multiple work groups process blocks belonging to the same parent partition.
// The last work group to finish (tracked via atomic blockcount) computes new pivots.
// ============================================================================

#define DEFINE_GQSORT_KERNEL(SUFFIX, T, WORK_REC, BLOCK_REC)                          \
kernel void gqsort_kernel_##SUFFIX(                                                     \
    device T* d                      [[buffer(0)]],                                     \
    device T* dn                     [[buffer(1)]],                                     \
    const device BLOCK_REC* blocks   [[buffer(2)]],                                     \
    device parent_record* parents    [[buffer(3)]],                                     \
    device WORK_REC* news            [[buffer(4)]],                                     \
    uint blockid                     [[threadgroup_position_in_grid]],                   \
    uint localid                     [[thread_position_in_threadgroup]])                 \
{                                                                                        \
    threadgroup uint lt_scan[GQSORT_LOCAL_WORKGROUP_SIZE + 1];                           \
    threadgroup uint gt_scan[GQSORT_LOCAL_WORKGROUP_SIZE + 1];                           \
    threadgroup uint ltsum_val, gtsum_val, lbeg_val = 0, gbeg_val = 0;                   \
                                                                                         \
    BLOCK_REC block = blocks[blockid];                                                   \
    uint start = block.start, end = block.end, direction = block.direction;              \
    T pivot = T(block.pivot);                                                            \
                                                                                         \
    device T* s  = (direction == 1) ? d : dn;                                           \
    device T* sn = (direction == 1) ? dn : d;                                           \
                                                                                         \
    /* Count elements < pivot and > pivot */                                              \
    uint lt = 0, gt = 0;                                                                 \
    for (uint i = start + localid; i < end; i += GQSORT_LOCAL_WORKGROUP_SIZE) {          \
        T tmp = s[i];                                                                    \
        if (tmp < pivot) lt++;                                                           \
        if (tmp > pivot) gt++;                                                           \
    }                                                                                    \
                                                                                         \
    /* Exclusive prefix scan to compute write positions */                               \
    lt_scan[localid] = lt;                                                               \
    gt_scan[localid] = gt;                                                               \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                      \
    exclusive_scan<GQSORT_LOCAL_WORKGROUP_SIZE>(lt_scan, localid);                       \
    exclusive_scan<GQSORT_LOCAL_WORKGROUP_SIZE>(gt_scan, localid);                       \
                                                                                         \
    if (localid == GQSORT_LOCAL_WORKGROUP_SIZE - 1) {                                    \
        ltsum_val = lt_scan[localid] + lt;                                               \
        gtsum_val = gt_scan[localid] + gt;                                               \
    }                                                                                    \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                      \
                                                                                         \
    /* Atomically allocate output positions from the parent record */                    \
    if (localid == 0) {                                                                  \
        device atomic_uint* psstart = (device atomic_uint*)&parents[block.parent].sstart;\
        device atomic_uint* psend   = (device atomic_uint*)&parents[block.parent].send;  \
        lbeg_val = atomic_fetch_add_explicit(psstart, ltsum_val, memory_order_relaxed);  \
        gbeg_val = atomic_fetch_sub_explicit(psend, gtsum_val, memory_order_relaxed)     \
                   - gtsum_val;                                                          \
    }                                                                                    \
    /* NOTE: the device barrier here only ensures THIS threadgroup's writes are */       \
    /* visible, not other threadgroups'. Cross-TG ordering is via atomics below. */                                                                                    \
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);                           \
                                                                                         \
    uint lfrom = lbeg_val + lt_scan[localid];                                            \
    uint gfrom = gbeg_val + gt_scan[localid];                                            \
                                                                                         \
    /* Write elements to their partitioned positions */                                  \
    for (uint i = start + localid; i < end; i += GQSORT_LOCAL_WORKGROUP_SIZE) {          \
        T tmp = s[i];                                                                    \
        if (tmp < pivot) sn[lfrom++] = tmp;                                              \
        if (tmp > pivot) sn[gfrom++] = tmp;                                              \
    }                                                                                    \
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);                           \
                                                                                         \
    /* Last block to finish: fill pivot region and compute new pivots */                 \
    if (localid == 0) {                                                                  \
        device atomic_uint* pbc =                                                        \
            (device atomic_uint*)&parents[block.parent].blockcount;                      \
        if (atomic_fetch_sub_explicit(pbc, 1u, memory_order_relaxed) == 0) {             \
            device atomic_uint* psstart2 =                                               \
                (device atomic_uint*)&parents[block.parent].sstart;                      \
            device atomic_uint* psend2 =                                                 \
                (device atomic_uint*)&parents[block.parent].send;                        \
            uint ss = atomic_load_explicit(psstart2, memory_order_relaxed);              \
            uint se = atomic_load_explicit(psend2, memory_order_relaxed);                \
            uint os = parents[block.parent].oldstart;                                    \
            uint oe = parents[block.parent].oldend;                                      \
                                                                                         \
            for (uint i = ss; i < se; i++) d[i] = pivot;                                 \
                                                                                         \
            T lpivot = sn[os];                                                           \
            T gpivot = sn[oe - 1];                                                      \
            if (os < ss)                                                                 \
                lpivot = median3(lpivot, sn[(os + ss) >> 1], sn[ss - 1]);                \
            if (se < oe)                                                                 \
                gpivot = median3(sn[se], sn[(oe + se) >> 1], gpivot);                    \
                                                                                         \
            uint new_dir = direction ^ 1;                                                \
            WORK_REC r1; r1.start = os; r1.end = ss;                                    \
            r1.pivot = lpivot; r1.direction = new_dir;                                   \
            news[2 * blockid] = r1;                                                      \
            WORK_REC r2; r2.start = se; r2.end = oe;                                    \
            r2.pivot = gpivot; r2.direction = new_dir;                                   \
            news[2 * blockid + 1] = r2;                                                  \
        }                                                                                \
    }                                                                                    \
}

DEFINE_GQSORT_KERNEL(uint,  uint,  work_record_uint,  block_record_uint)
DEFINE_GQSORT_KERNEL(float, float, work_record_float, block_record_float)
DEFINE_GQSORT_KERNEL(half,  half,  work_record_half,  block_record_half)


// ============================================================================
// LQSORT KERNEL — Local Quicksort
// Sorts blocks that are small enough to fit in threadgroup memory.
// Uses quicksort with an internal workstack, falling back to bitonic sort
// for sub-sequences <= SORT_THRESHOLD.
// ============================================================================

#define DEFINE_LQSORT_KERNEL(SUFFIX, T, WORK_REC)                                      \
kernel void lqsort_kernel_##SUFFIX(                                                      \
    device T* d                      [[buffer(0)]],                                      \
    device T* dn                     [[buffer(1)]],                                      \
    const device WORK_REC* seqs      [[buffer(2)]],                                      \
    uint blockid                     [[threadgroup_position_in_grid]],                    \
    uint localid                     [[thread_position_in_threadgroup]])                  \
{                                                                                         \
    threadgroup workstack_record workstack[QUICKSORT_BLOCK_SIZE / SORT_THRESHOLD];        \
    threadgroup int workstack_pointer;                                                    \
    threadgroup T mys[QUICKSORT_BLOCK_SIZE], mysn[QUICKSORT_BLOCK_SIZE];                  \
    threadgroup T temp_sort[SORT_THRESHOLD];                                               \
    threadgroup uint lt_scan[LQSORT_LOCAL_WORKGROUP_SIZE + 1];                            \
    threadgroup uint gt_scan[LQSORT_LOCAL_WORKGROUP_SIZE + 1];                            \
    threadgroup uint ltsum_val, gtsum_val;                                                 \
                                                                                          \
    WORK_REC block = seqs[blockid];                                                       \
    const uint d_offset = block.start;                                                    \
    uint start = 0;                                                                       \
    uint end = block.end - d_offset;                                                      \
                                                                                          \
    uint direction = 1;                                                                   \
    if (localid == 0) {                                                                   \
        workstack_pointer = 0;                                                            \
        workstack_record wr;                                                              \
        wr.start = start; wr.end = end; wr.direction = direction;                         \
        workstack[0] = wr;                                                                \
    }                                                                                     \
                                                                                          \
    /* Copy data into threadgroup memory */                                               \
    if (block.direction == 1) {                                                           \
        for (uint i = localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE)                 \
            mys[i] = d[i + d_offset];                                                    \
    } else {                                                                              \
        for (uint i = localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE)                 \
            mys[i] = dn[i + d_offset];                                                   \
    }                                                                                     \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                       \
                                                                                          \
    while (workstack_pointer >= 0) {                                                      \
        workstack_record wr = workstack[workstack_pointer];                               \
        start = wr.start;                                                                 \
        end = wr.end;                                                                     \
        direction = wr.direction;                                                         \
        if (localid == 0) {                                                               \
            workstack_pointer--;                                                          \
        }                                                                                 \
        threadgroup T* s  = (direction == 1) ? mys : mysn;                                \
        threadgroup T* sn = (direction == 1) ? mysn : mys;                                \
                                                                                          \
        uint lt = 0, gt = 0;                                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
                                                                                          \
        /* Pick a pivot */                                                                \
        T pivot = s[start];                                                               \
        if (start < end)                                                                  \
            pivot = median3(pivot, s[(start + end) >> 1], s[end - 1]);                    \
                                                                                          \
        /* Count elements < and > pivot */                                                \
        for (uint i = start + localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {       \
            T tmp = s[i];                                                                 \
            if (tmp < pivot) lt++;                                                        \
            if (tmp > pivot) gt++;                                                        \
        }                                                                                 \
                                                                                          \
        /* Prefix scans */                                                                \
        lt_scan[localid] = lt;                                                            \
        gt_scan[localid] = gt;                                                            \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
        exclusive_scan<LQSORT_LOCAL_WORKGROUP_SIZE>(lt_scan, localid);                    \
        exclusive_scan<LQSORT_LOCAL_WORKGROUP_SIZE>(gt_scan, localid);                    \
        /* For gt we need inclusive scan: exclusive + original */                          \
        uint gt_inclusive = gt_scan[localid] + gt;                                        \
                                                                                          \
        if (localid == LQSORT_LOCAL_WORKGROUP_SIZE - 1) {                                 \
            ltsum_val = lt_scan[localid] + lt;                                            \
            gtsum_val = gt_inclusive;                                                      \
        }                                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
                                                                                          \
        uint lfrom = start + lt_scan[localid];                                            \
        uint gfrom = end - gt_inclusive;                                                   \
                                                                                          \
        /* Write elements to partitioned positions */                                     \
        for (uint i = start + localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {       \
            T tmp = s[i];                                                                 \
            if (tmp < pivot) sn[lfrom++] = tmp;                                           \
            if (tmp > pivot) sn[gfrom++] = tmp;                                           \
        }                                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
                                                                                          \
        /* Fill pivot region in global output */                                          \
        for (uint i = start + ltsum_val + localid;                                        \
             i < end - gtsum_val;                                                         \
             i += LQSORT_LOCAL_WORKGROUP_SIZE) {                                          \
            d[i + d_offset] = pivot;                                                      \
        }                                                                                 \
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);                        \
                                                                                          \
        /* Sort or push left partition */                                                 \
        if (ltsum_val <= SORT_THRESHOLD) {                                                \
            sort_threshold(sn, d + d_offset, start, start + ltsum_val,                    \
                          temp_sort, localid);                                             \
        } else {                                                                          \
            if (localid == 0) {                                                           \
                workstack_pointer++;                                                      \
                workstack_record pwr;                                                     \
                pwr.start = start;                                                        \
                pwr.end = start + ltsum_val;                                              \
                pwr.direction = direction ^ 1;                                            \
                workstack[workstack_pointer] = pwr;                                       \
            }                                                                             \
            threadgroup_barrier(mem_flags::mem_threadgroup);                               \
        }                                                                                 \
                                                                                          \
        /* Sort or push right partition */                                                \
        if (gtsum_val <= SORT_THRESHOLD) {                                                \
            sort_threshold(sn, d + d_offset, end - gtsum_val, end,                        \
                          temp_sort, localid);                                             \
        } else {                                                                          \
            if (localid == 0) {                                                           \
                workstack_pointer++;                                                      \
                workstack_record pwr;                                                     \
                pwr.start = end - gtsum_val;                                              \
                pwr.end = end;                                                            \
                pwr.direction = direction ^ 1;                                            \
                workstack[workstack_pointer] = pwr;                                       \
            }                                                                             \
            threadgroup_barrier(mem_flags::mem_threadgroup);                               \
        }                                                                                 \
    }                                                                                     \
}

DEFINE_LQSORT_KERNEL(uint,  uint,  work_record_uint)
DEFINE_LQSORT_KERNEL(float, float, work_record_float)
DEFINE_LQSORT_KERNEL(half,  half,  work_record_half)
