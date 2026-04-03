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

// QuicksortMain.mm — Metal port of GPU-Quicksort
// Ported from GPU-Quicksort_Full_SYCL_2.0

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <vector>

#ifdef USE_GCD_SORT
#include <dispatch/dispatch.h>
#endif

#include "QuicksortTypes.h"

// ============================================================================
// Host-side data structures (templated, matching the concrete Metal structs)
// ============================================================================

template <class T>
struct work_record {
    uint start;
    uint end;
    T    pivot;
    uint direction;

    work_record() : start(0), end(0), pivot(T(0)), direction(EMPTY_RECORD) {}
    work_record(uint s, uint e, T p, uint d) : start(s), end(e), pivot(p), direction(d) {}
};

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

// Verify layout compatibility with Metal-side structs
static_assert(sizeof(work_record<uint>) == sizeof(work_record_uint), "work_record<uint> layout mismatch");
static_assert(sizeof(work_record<float>) == sizeof(work_record_float), "work_record<float> layout mismatch");
static_assert(sizeof(block_record<uint>) == sizeof(block_record_uint), "block_record<uint> layout mismatch");
static_assert(sizeof(block_record<float>) == sizeof(block_record_float), "block_record<float> layout mismatch");
static_assert(sizeof(parent_record) == 20, "parent_record unexpected size");

// ============================================================================
// Timing utility
// ============================================================================

double seconds() {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec + now.tv_nsec / 1000000000.0;
}

// ============================================================================
// Metal resources
// ============================================================================

struct MetalResources {
    id<MTLDevice>              device;
    id<MTLCommandQueue>        commandQueue;
    id<MTLLibrary>             library;
    id<MTLComputePipelineState> gqsortPipeline_uint;
    id<MTLComputePipelineState> lqsortPipeline_uint;
    id<MTLComputePipelineState> gqsortPipeline_float;
    id<MTLComputePipelineState> lqsortPipeline_float;
    id<MTLComputePipelineState> gqsortPipeline_half;
    id<MTLComputePipelineState> lqsortPipeline_half;
};

id<MTLComputePipelineState> createPipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                            const char* functionName) {
    NSString* name = [NSString stringWithUTF8String:functionName];
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) {
        std::cerr << "Failed to find Metal function: " << functionName << std::endl;
        exit(1);
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        std::cerr << "Failed to create pipeline for " << functionName << ": "
                  << [[error localizedDescription] UTF8String] << std::endl;
        exit(1);
    }
    return pipeline;
}

void initMetal(MetalResources* metal, const char* metallibPath) {
    metal->device = MTLCreateSystemDefaultDevice();
    if (!metal->device) {
        std::cerr << "Metal is not supported on this device." << std::endl;
        exit(1);
    }
    metal->commandQueue = [metal->device newCommandQueue];

    // Load compiled Metal library
    NSError* error = nil;
    NSString* path = [NSString stringWithUTF8String:metallibPath];
    NSURL* url = [NSURL fileURLWithPath:path];
    metal->library = [metal->device newLibraryWithURL:url error:&error];
    if (!metal->library) {
        std::cerr << "Failed to load Metal library from " << metallibPath << ": "
                  << [[error localizedDescription] UTF8String] << std::endl;
        exit(1);
    }

    // Create pipeline states for all kernel variants
    metal->gqsortPipeline_uint  = createPipeline(metal->device, metal->library, "gqsort_kernel_uint");
    metal->lqsortPipeline_uint  = createPipeline(metal->device, metal->library, "lqsort_kernel_uint");
    metal->gqsortPipeline_float = createPipeline(metal->device, metal->library, "gqsort_kernel_float");
    metal->lqsortPipeline_float = createPipeline(metal->device, metal->library, "lqsort_kernel_float");
    metal->gqsortPipeline_half  = createPipeline(metal->device, metal->library, "gqsort_kernel_half");
    metal->lqsortPipeline_half  = createPipeline(metal->device, metal->library, "lqsort_kernel_half");
}

void queryPrintDeviceInfo(MetalResources* metal) {
    std::cout << "\nUsing Metal device: " << [[metal->device name] UTF8String] << std::endl;
    std::cout << "Metal Device info:" << std::endl;
    std::cout << "  Name                    : " << [[metal->device name] UTF8String] << std::endl;
    std::cout << "  Max Threadgroup Memory  : " << metal->device.maxThreadgroupMemoryLength << " bytes" << std::endl;
    std::cout << "  Max Threads/Threadgroup : " << metal->device.maxThreadsPerThreadgroup.width << std::endl;
    std::cout << "  Unified Memory          : " << (metal->device.hasUnifiedMemory ? "Yes" : "No") << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// Median (host-side pivot selection)
// ============================================================================

template <class T>
T median(T x1, T x2, T x3) {
    if (x1 < x2) {
        if (x2 < x3) return x2;
        else return (x1 < x3) ? x3 : x1;
    } else {
        if (x1 < x3) return x1;
        else return (x2 < x3) ? x3 : x2;
    }
}

// ============================================================================
// Pipeline selection traits
// ============================================================================

template <class T> struct PipelineSelector;

template <> struct PipelineSelector<uint> {
    static id<MTLComputePipelineState> gqsort(MetalResources* m) { return m->gqsortPipeline_uint; }
    static id<MTLComputePipelineState> lqsort(MetalResources* m) { return m->lqsortPipeline_uint; }
};

template <> struct PipelineSelector<float> {
    static id<MTLComputePipelineState> gqsort(MetalResources* m) { return m->gqsortPipeline_float; }
    static id<MTLComputePipelineState> lqsort(MetalResources* m) { return m->lqsortPipeline_float; }
};

// Note: host uses uint16_t for half; Metal uses half.
// We use __fp16 when available, otherwise uint16_t with conversion.

// ============================================================================
// gqsort — dispatch global quicksort kernel
// ============================================================================

template <class T>
void gqsort(MetalResources* metal,
            id<MTLBuffer> d_buffer,
            id<MTLBuffer> dn_buffer,
            std::vector<block_record<T>>& blocks,
            std::vector<parent_record>& parents,
            std::vector<work_record<T>>& news) {

    news.resize(blocks.size() * 2);

    id<MTLBuffer> blocks_buffer = [metal->device
        newBufferWithBytes:blocks.data()
        length:blocks.size() * sizeof(block_record<T>)
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> parents_buffer = [metal->device
        newBufferWithBytes:parents.data()
        length:parents.size() * sizeof(parent_record)
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> news_buffer = [metal->device
        newBufferWithLength:news.size() * sizeof(work_record<T>)
        options:MTLResourceStorageModeShared];

    // Initialize news to EMPTY_RECORD
    work_record<T>* news_ptr = (work_record<T>*)[news_buffer contents];
    for (size_t i = 0; i < news.size(); i++)
        news_ptr[i].direction = EMPTY_RECORD;

    id<MTLCommandBuffer> commandBuffer = [metal->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:PipelineSelector<T>::gqsort(metal)];
    [encoder setBuffer:d_buffer       offset:0 atIndex:0];
    [encoder setBuffer:dn_buffer      offset:0 atIndex:1];
    [encoder setBuffer:blocks_buffer  offset:0 atIndex:2];
    [encoder setBuffer:parents_buffer offset:0 atIndex:3];
    [encoder setBuffer:news_buffer    offset:0 atIndex:4];

    [encoder dispatchThreadgroups:MTLSizeMake(blocks.size(), 1, 1)
             threadsPerThreadgroup:MTLSizeMake(GQSORT_LOCAL_WORKGROUP_SIZE, 1, 1)];

    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    memcpy(news.data(), [news_buffer contents], news.size() * sizeof(work_record<T>));
}

// ============================================================================
// lqsort — dispatch local quicksort kernel
// ============================================================================

template <class T>
void lqsort(MetalResources* metal,
            id<MTLBuffer> d_buffer,
            id<MTLBuffer> dn_buffer,
            std::vector<work_record<T>>& done) {

    id<MTLBuffer> done_buffer = [metal->device
        newBufferWithBytes:done.data()
        length:done.size() * sizeof(work_record<T>)
        options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> commandBuffer = [metal->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:PipelineSelector<T>::lqsort(metal)];
    [encoder setBuffer:d_buffer    offset:0 atIndex:0];
    [encoder setBuffer:dn_buffer   offset:0 atIndex:1];
    [encoder setBuffer:done_buffer offset:0 atIndex:2];

    [encoder dispatchThreadgroups:MTLSizeMake(done.size(), 1, 1)
             threadsPerThreadgroup:MTLSizeMake(LQSORT_LOCAL_WORKGROUP_SIZE, 1, 1)];

    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

// ============================================================================
// GPUQSort — host-side orchestration loop
// ============================================================================

size_t optp(size_t s, double k, size_t m) {
    return (size_t)pow(2, floor(log(s * k + m) / log(2.0) + 0.5));
}

template <class T>
void GPUQSort(MetalResources* metal, size_t size, T* d, T* dn) {
    size_t bufSize = size * sizeof(T);
    id<MTLBuffer> d_buffer  = [metal->device newBufferWithBytes:d length:bufSize
                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> dn_buffer = [metal->device newBufferWithLength:bufSize
                               options:MTLResourceStorageModeShared];

    const size_t MAXSEQ = optp(size, 0.00009516, 203);
    const size_t MAX_SIZE = 12 * std::max(MAXSEQ, (size_t)QUICKSORT_BLOCK_SIZE);

    T startpivot = median(d[0], d[size / 2], d[size - 1]);
    std::vector<work_record<T>> work, done, news;
    work.reserve(MAX_SIZE);
    done.reserve(MAX_SIZE);
    news.reserve(MAX_SIZE);
    std::vector<parent_record> parent_records;
    parent_records.reserve(MAX_SIZE);
    std::vector<block_record<T>> blocks;
    blocks.reserve(MAX_SIZE);

    work.push_back(work_record<T>(0, size, startpivot, 1));

    while (!work.empty()) {
        size_t blocksize = 0;
        for (auto it = work.begin(); it != work.end(); ++it)
            blocksize += std::max((size_t)(it->end - it->start) / MAXSEQ, (size_t)1);

        for (auto it = work.begin(); it != work.end(); ++it) {
            uint start = it->start;
            uint end   = it->end;
            T pivot = it->pivot;
            uint direction = it->direction;
            uint blockcount = (end - start + blocksize - 1) / blocksize;
            parent_record prnt;
            prnt.sstart = start; prnt.send = end;
            prnt.oldstart = start; prnt.oldend = end;
            prnt.blockcount = blockcount - 1;
            parent_records.push_back(prnt);

            for (uint i = 0; i < blockcount - 1; i++) {
                uint bstart = start + blocksize * i;
                blocks.push_back(block_record<T>(bstart, bstart + blocksize, pivot, direction, parent_records.size() - 1));
            }
            blocks.push_back(block_record<T>(start + blocksize * (blockcount - 1), end, pivot, direction, parent_records.size() - 1));
        }

        gqsort(metal, d_buffer, dn_buffer, blocks, parent_records, news);

        work.clear();
        parent_records.clear();
        blocks.clear();
        for (auto it = news.begin(); it != news.end(); ++it) {
            if (it->direction != EMPTY_RECORD) {
                if (it->end - it->start <= QUICKSORT_BLOCK_SIZE) {
                    if (it->end - it->start > 0)
                        done.push_back(*it);
                } else {
                    work.push_back(*it);
                }
            }
        }
        news.clear();
    }
    for (auto it = work.begin(); it != work.end(); ++it) {
        if (it->end - it->start > 0)
            done.push_back(*it);
    }

    lqsort(metal, d_buffer, dn_buffer, done);

    memcpy(d, [d_buffer contents], bufSize);
}

// ============================================================================
// GCD parallel sort (optional CPU comparison)
// ============================================================================

#ifdef USE_GCD_SORT
template <class T>
void parallel_sort_gcd(T* data, size_t count) {
    if (count < 1024) {
        std::sort(data, data + count);
        return;
    }
    // Simple parallel merge sort using GCD
    size_t num_chunks = 8;  // Use 8 parallel chunks
    size_t chunk_size = (count + num_chunks - 1) / num_chunks;

    dispatch_apply(num_chunks, DISPATCH_APPLY_AUTO, ^(size_t idx) {
        size_t start = idx * chunk_size;
        size_t end = std::min(start + chunk_size, count);
        if (start < end) {
            std::sort(data + start, data + end);
        }
    });

    // Merge sorted chunks
    std::vector<T> temp(count);
    for (size_t width = chunk_size; width < count; width *= 2) {
        for (size_t i = 0; i < count; i += 2 * width) {
            size_t left = i;
            size_t mid = std::min(i + width, count);
            size_t right = std::min(i + 2 * width, count);
            std::merge(data + left, data + mid, data + mid, data + right, temp.begin() + left);
        }
        std::copy(temp.begin(), temp.end(), data);
    }
}
#endif

// ============================================================================
// big_test — benchmark and verification
// ============================================================================

#define TRUST_BUT_VERIFY 1

template <class T>
int big_test(MetalResources& metal, uint arraySize, unsigned int NUM_ITERATIONS,
             const std::string& type_name)
{
    double totalTime, stdSortTime;
    double beginClock, endClock;

    printf("\n\n\n--------------------------------------------------------------------\n");
    printf("Allocating array size of %d\n", arraySize);

    // Page-aligned allocation for Metal zero-copy buffers
    size_t allocSize = ((arraySize * sizeof(T)) / 4096 + 1) * 4096;
    T* pArray = (T*)aligned_alloc(4096, allocSize);
    T* pArrayCopy = (T*)aligned_alloc(4096, allocSize);

    uint gen_counter = 0;
    std::generate(pArray, pArray + arraySize, [&gen_counter]() { return T(++gen_counter); });
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(pArray, pArray + arraySize, g);

#ifdef RUN_CPU_SORTS
    std::cout << "Sorting with std::sort..." << std::endl;
    std::copy(pArray, pArray + arraySize, pArrayCopy);
    beginClock = seconds();
    std::sort(pArrayCopy, pArrayCopy + arraySize);
    endClock = seconds();
    totalTime = endClock - beginClock;
    std::cout << "Time to sort: " << totalTime * 1000 << " ms" << std::endl;
    stdSortTime = totalTime;

#ifdef TRUST_BUT_VERIFY
    {
        std::vector<T> verify(arraySize);
        std::copy(pArray, pArray + arraySize, verify.begin());
        std::cout << "verifying std::sort: ";
        std::sort(verify.begin(), verify.end());
        bool correct = std::equal(verify.begin(), verify.end(), pArrayCopy);
        std::cout << std::boolalpha << correct << std::endl;
    }
#endif

#ifdef USE_GCD_SORT
    std::cout << "Sorting with GCD parallel sort..." << std::endl;
    std::copy(pArray, pArray + arraySize, pArrayCopy);
    beginClock = seconds();
    parallel_sort_gcd(pArrayCopy, arraySize);
    endClock = seconds();
    totalTime = endClock - beginClock;
    std::cout << "Time to sort: " << totalTime * 1000 << " ms" << std::endl;
    double gcdSortTime = totalTime;
#endif
#endif // RUN_CPU_SORTS

    std::cout << "Sorting with GPUQSort on Metal with type " << type_name << ":" << std::endl;
    std::vector<T> original(arraySize);
    std::copy(pArray, pArray + arraySize, original.begin());

    std::vector<double> times(NUM_ITERATIONS);
    double AverageTime = 0.0;
    uint num_failures = 0;

    for (uint k = 0; k < NUM_ITERATIONS; k++) {
        std::copy(original.begin(), original.end(), pArray);
        std::vector<T> verify(arraySize);
        std::copy(pArray, pArray + arraySize, verify.begin());

        beginClock = seconds();
        GPUQSort(&metal, arraySize, pArray, pArrayCopy);
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
            for (size_t i = 0; i < arraySize; i++) {
                if (verify[i] != pArray[i]) num_discrepancies++;
            }
        }
        std::cout << std::boolalpha << correct << std::endl;
        if (!correct) {
            std::cout << "num_discrepancies: " << num_discrepancies << std::endl;
            num_failures++;
        }
#endif
    }

    std::cout << " Number of failures: " << num_failures << " out of " << NUM_ITERATIONS << std::endl;
    AverageTime = AverageTime / NUM_ITERATIONS;
    std::cout << "Average Time: " << AverageTime * 1000 << " ms" << std::endl;

    double stdDev = 0.0, minTime = 1e9, maxTime = 0.0;
    for (uint k = 0; k < NUM_ITERATIONS; k++) {
        stdDev += (AverageTime - times[k]) * (AverageTime - times[k]);
        minTime = std::min(minTime, times[k]);
        maxTime = std::max(maxTime, times[k]);
    }
    if (NUM_ITERATIONS > 1) {
        stdDev = sqrt(stdDev / (NUM_ITERATIONS - 1));
        std::cout << "Standard Deviation: " << stdDev * 1000 << std::endl;
        std::cout << "%error (3*stdDev)/Average: " << 3 * stdDev / AverageTime * 100 << "%" << std::endl;
        std::cout << "min time: " << minTime * 1000 << " ms" << std::endl;
        std::cout << "max time: " << maxTime * 1000 << " ms" << std::endl;
    }

#ifdef RUN_CPU_SORTS
    std::cout << "Average speedup over CPU std::sort: " << stdSortTime / AverageTime << std::endl;
#ifdef USE_GCD_SORT
    std::cout << "Average speedup over GCD parallel sort: " << gcdSortTime / AverageTime << std::endl;
#endif
#endif

    printf("-------done--------------------------------------------------------\n");
    free(pArray);
    free(pArrayCopy);
    return 0;
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) {
    const char* usage = "Usage: Quicksort <num_iterations> <array_width> <array_height> [show_device]";

    if (argc < 4) {
        std::cerr << usage << std::endl;
        return 1;
    }

    unsigned int NUM_ITERATIONS = atoi(argv[1]);
    uint widthReSz = atoi(argv[2]);
    uint heightReSz = atoi(argv[3]);
    bool showDevice = (argc > 4 && argv[4][0] == 's');

    // Find the metallib next to the executable
    // First try the build directory, then the executable directory
    std::string metallibPath = "QuicksortKernels.metallib";

    MetalResources metal;
    initMetal(&metal, metallibPath.c_str());

    if (showDevice)
        queryPrintDeviceInfo(&metal);

    uint arraySize = widthReSz * heightReSz;
    big_test<uint>(metal, arraySize, NUM_ITERATIONS, "uint");
    big_test<float>(metal, arraySize, NUM_ITERATIONS, "float");
    // Note: half (float16) test would require special handling for host-side
    // data generation and verification. Skipping for now since std::shuffle
    // and std::sort don't work directly with __fp16.
    // big_test<__fp16>(metal, arraySize, NUM_ITERATIONS, "half");

    return 0;
}
