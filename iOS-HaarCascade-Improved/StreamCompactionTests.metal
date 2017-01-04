//
//  StreamCompactionTests.metal
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 05.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

#include "StreamCompactionDefinitions.metal"
using namespace metal;

kernel void subReduceAddTest(
                                  device float* input [[ buffer(0) ]],
                                  device float &output [[ buffer(1) ]],
                                  constant int &dataCount [[buffer(2)]],
                                  threadgroup float* shmem [[ threadgroup(0) ]],
                                  uint2 blockIdx  [[ threadgroup_position_in_grid ]],
                                  uint2 threadIdx [[ thread_position_in_threadgroup ]],
                                  uint2 globalIdx [[ thread_position_in_grid ]]
                                  ) {
    
    float value = int(globalIdx.x) < dataCount ? input[globalIdx.x] : 0.0;
    threadgroup_barrier( mem_flags::mem_threadgroup );
    float res = subReduceAdd(threadIdx.x, value, shmem);
        
    if (globalIdx.x == 0) {
        output = res;
    }
    
}

kernel void streamCompactionTest(
                                 device int* input [[ buffer(0) ]],
                                 device int* output [[ buffer(1) ]],
                                 device atomic_int &outMaskPosition [[ buffer(2) ]],
                                 constant int &dataCount [[buffer(3)]],
                                 threadgroup int* shmem [[threadgroup(0)]],
                                 threadgroup int& numPassed [[threadgroup(1)]],
                                 threadgroup int& outMaskOffset [[threadgroup(2)]],
                                 uint2 blockIdx  [[ threadgroup_position_in_grid ]],
                                 uint2 threadIdx [[ thread_position_in_threadgroup ]],
                                 uint2 globalIdx [[ thread_position_in_grid ]]
) {
    
    
    bool isValidData = int(globalIdx.x) < dataCount && input[globalIdx.x];
    
    int threadElem = 0;
    int threadPassFlag = 0;
    
    if (isValidData) {
        threadElem = globalIdx.x;
        threadPassFlag = 1;
    }
    
    streamCompactWriteOut(
                                       shmem,
                                       numPassed,
                                       outMaskOffset,
                                       threadIdx.x,
                                       threadPassFlag,
                                       threadElem,
                                       output,
                                       outMaskPosition
                                       );
    
    
    
    
}
