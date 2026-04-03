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

#ifndef QUICKSORT_TYPES_H
#define QUICKSORT_TYPES_H

// This header is shared between Metal shaders and host Objective-C++ code.
// It defines data structures and constants that must have identical layout on both sides.

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#else
#include <cstdint>
typedef unsigned int uint;
#endif

// Tuning constants for Apple GPU (M-series).
// SORT_THRESHOLD must be 2x LQSORT_LOCAL_WORKGROUP_SIZE (bitonic sort requirement).
// LQSORT_LOCAL_WORKGROUP_SIZE should be ~8x smaller than QUICKSORT_BLOCK_SIZE.
#define QUICKSORT_BLOCK_SIZE         1728
#define GQSORT_LOCAL_WORKGROUP_SIZE   256
#define LQSORT_LOCAL_WORKGROUP_SIZE   128
#define SORT_THRESHOLD                256

#define EMPTY_RECORD  42

// work_record: describes a sub-array that still needs sorting.
// Contains start/end indices, pivot value, and sort direction (ping-pong between two buffers).
struct work_record_uint {
    uint start;
    uint end;
    uint pivot;
    uint direction;
};

struct work_record_float {
    uint  start;
    uint  end;
    float pivot;
    uint  direction;
};

struct work_record_half {
    uint start;
    uint end;
#ifdef __METAL_VERSION__
    half pivot;
    half _pad;  // pad to 4-byte alignment
#else
    uint16_t pivot;
    uint16_t _pad;
#endif
    uint direction;
};

// parent_record: tracks the parent partition for a set of blocks being processed by gqsort.
// sstart/send are atomically updated as blocks write their partitioned data.
// oldstart/oldend are the original boundaries.
// blockcount is atomically decremented; the last block to finish computes new pivots.
struct parent_record {
    uint sstart;
    uint send;
    uint oldstart;
    uint oldend;
    uint blockcount;
};

// block_record: describes one block assigned to a work group in gqsort.
// Contains start/end indices, pivot, direction, and index into the parent_record array.
struct block_record_uint {
    uint start;
    uint end;
    uint pivot;
    uint direction;
    uint parent;
};

struct block_record_float {
    uint  start;
    uint  end;
    float pivot;
    uint  direction;
    uint  parent;
};

struct block_record_half {
    uint start;
    uint end;
#ifdef __METAL_VERSION__
    half pivot;
    half _pad;
#else
    uint16_t pivot;
    uint16_t _pad;
#endif
    uint direction;
    uint parent;
};

// workstack_record: used by lqsort to manage an internal stack of sub-sequences
// that still need partitioning within threadgroup memory.
struct workstack_record {
    uint start;
    uint end;
    uint direction;
};

#endif // QUICKSORT_TYPES_H
