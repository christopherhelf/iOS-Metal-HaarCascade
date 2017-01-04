//
//  HaarCascadeRectangles.metal
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 16.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

#include <metal_stdlib>
#include "HaarStructsDefinitions.metal"
#include "StreamCompactionDefinitions.metal"
#include "IntegralDefinitions.metal"
using namespace metal;


kernel void haarCascadeGatherRectangles(
                                        constant int &numWindows                        [[ buffer(0) ]],
                                        constant DetectionWindow* windows               [[ buffer(1) ]],
                                        constant int* clusters                          [[ buffer(2) ]],
                                        device float4* rectangles                       [[ buffer(3) ]],
                                        constant float &scale                           [[ buffer(4) ]],
                                        threadgroup float4* subReduceMemoryRecs         [[threadgroup(0)]],
                                        threadgroup int* subReduceMemoryNum             [[threadgroup(1)]],
                                        uint numberOfThreads                            [[threads_per_threadgroup]],
                                        uint threadIdx                                  [[thread_index_in_threadgroup]],
                                        uint cluster                                    [[threadgroup_position_in_grid]]
                                        ) {
    
    // the sums, we need to split this up as there's no structure supporting 5 values
    float4 sumPerThread = 0.0;
    int totalNodesPerThread = 0;
    
    // iterate through all the nodes
    for(int i=threadIdx; i<numWindows; i+=numberOfThreads) {
        if (clusters[i] == (int) cluster + 1) {
            DetectionWindow w = windows[i];
            sumPerThread += float4( (float) w.x / scale, (float) w.y / scale, (float) w.width / scale, (float) w.height / scale);
            totalNodesPerThread += 1;
        }
    }
    
    // wait until all threads have finished
    threadgroup_barrier( mem_flags::mem_none );
    
    // sum up the rectangles
    float4 totalSum = subReduceAdd(threadIdx, sumPerThread, subReduceMemoryRecs);
    
    // sum up the number of nodes
    int totalNodes = subReduceAdd(threadIdx, totalNodesPerThread, subReduceMemoryNum);
    
    // write out the result
    if (!threadIdx) {
        rectangles[cluster] = totalSum/(float) totalNodes;
    }
    
    
}


