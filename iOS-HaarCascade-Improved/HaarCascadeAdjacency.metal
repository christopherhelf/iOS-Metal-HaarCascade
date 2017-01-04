//
//  HaarCascadeGrouping.metal
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 10.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//  http://www.sciencedirect.com/science/article/pii/S1877050913003438
//  https://pdfs.semanticscholar.org/4c77/e5650e2328390995f3219ec44a4efd803b84.pdf
//  https://arxiv.org/pdf/1506.02226.pdf
//  https://pdfs.semanticscholar.org/4c77/e5650e2328390995f3219ec44a4efd803b84.pdf

#include <metal_stdlib>
#include "HaarStructsDefinitions.metal"
#include "StreamCompactionDefinitions.metal"
#include "IntegralDefinitions.metal"
using namespace metal;

kernel void haarCascadeBuildAdjacencyList(
                                          constant DetectionWindow* windows [[ buffer(0) ]],
                                          constant int &numWindows [[ buffer(1) ]],
                                          
                                          threadgroup int* subReduceMemory [[threadgroup(0)]],
                                          threadgroup int* sharedMemory [[threadgroup(1)]],
                                          threadgroup int& sharedStreamCompactNumPassed [[threadgroup(2)]],
                                          threadgroup int& sharedOutMaskOffset [[threadgroup(3)]],
                                          threadgroup int* sharedMemoryCompaction [[threadgroup(4)]],
                                          
                                          device atomic_int &adjacencyListCounter [[ buffer(2) ]],
                                          device int* adjacencyList [[ buffer(3) ]],
                                          device int* neighborCount [[ buffer(4) ]],
                                          device int* offset [[ buffer(5) ]],
                                          uint2 blockIdx [[threadgroup_position_in_grid]],
                                          uint2 threadIdx [[ thread_position_in_threadgroup ]]) {
    
    
    // if we are out of bounds, or the point has already been visited, return
    if ((int)blockIdx.x >= numWindows) {
        return;
    }
    
    // store the offset
    if (!threadIdx.x) {
        sharedStreamCompactNumPassed = 0;
        sharedOutMaskOffset = 0;
    }
    
    threadgroup_barrier( mem_flags::mem_threadgroup );
    
    // get our detection window
    DetectionWindow w1 = windows[blockIdx.x];
    
    // do the region query
    uint numChunks = (numWindows + numThreadsAnchorsParallel - 1) >> log2NumThreadsAnchorsParallel;
    uint chunkOffset = threadIdx.x;
    int currentNeighbors = 0;
    
    // iterate through all chunks
    for(uint iChunk = 0; iChunk<numChunks; iChunk++) {
        
        // whether we have a valid region
        bool isInRegion = false;
        
        // check whether we are within bounds
        if (iChunk * numThreadsAnchorsParallel + threadIdx.x < uint(numWindows)) {
            
            // get the current window
            DetectionWindow w2 = windows[chunkOffset];
            
            // get the delta
            //float delta = eps * (float) (min(w1.width, w2.width) + min(w1.height, w2.height)) * 0.5f;
            
            // check whether we are within the region
            /*isInRegion = abs(w1.x - w2.x) <= delta &&
            abs(w1.y - w2.y) <= delta &&
            abs(w1.x + w1.width - w2.x - w2.width) <= delta &&
            abs(w1.y + w1.height - w2.y - w2.height) <= delta;*/
            
            // do a simpler test here
            isInRegion = w1.x < w2.width+w2.x && w1.width+w1.x > w2.x &&
            w1.y < w2.height+w2.y && w1.height+w1.y > w2.y;
            
            // store neighbor count
            if (isInRegion) {
                currentNeighbors += 1;
            }
        }
        
        threadgroup_barrier( mem_flags::mem_threadgroup );
        
        // use stream compaction to write out
        streamCompactWriteOutToSharedMemory(
                              sharedMemory,
                              sharedStreamCompactNumPassed,
                              sharedOutMaskOffset,
                              threadIdx.x,
                              isInRegion,
                              (int) chunkOffset,
                              sharedMemoryCompaction
                              );
        
        threadgroup_barrier( mem_flags::mem_threadgroup );
        
        chunkOffset += numThreadsAnchorsParallel;
    }
    
    threadgroup_barrier( mem_flags::mem_device );
    
    // sum up the neighbors
    int neighbors = subReduceAdd(threadIdx.x, currentNeighbors, subReduceMemory);
    
    // get the offset position, we reuse a threadgroup variable here
    // and write out the neighbor count
    if (!threadIdx.x) {
        int _pos = atomic_fetch_add_explicit( &adjacencyListCounter, sharedOutMaskOffset, memory_order_relaxed );
        sharedStreamCompactNumPassed = _pos;
        neighborCount[blockIdx.x] = neighbors;
        offset[blockIdx.x] = sharedStreamCompactNumPassed;
    }
    
    // wait for the first thread to write out
    threadgroup_barrier( mem_flags::mem_threadgroup );
    
    
    // write out the results
    for(int i=threadIdx.x; i<neighbors; i+=numThreadsAnchorsParallel) {
        int p = sharedMemoryCompaction[i];
        if (p >= 0 && p<numWindows) {
            adjacencyList[sharedStreamCompactNumPassed+i]=p;
        } else {
            adjacencyList[sharedStreamCompactNumPassed+i]=-1;
        }
        
    }
}

