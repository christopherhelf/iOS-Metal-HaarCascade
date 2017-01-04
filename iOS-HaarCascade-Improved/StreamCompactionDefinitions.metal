//
//  StreamCompactionDefinitions.metal
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 07.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

#include <metal_stdlib>
#include <metal_compute>
using namespace metal;

constant uint kWarpSize [[ function_constant(0) ]]; // 32
constant uint kLog2WarpSize [[ function_constant(1) ]]; // 5
constant uint numThreadsAnchorsParallel [[ function_constant(2) ]]; // 64
constant uint log2NumThreadsAnchorsParallel [[ function_constant(3) ]]; // 6

template<typename T>
device T warpScanInclusive(uint threadIdx, T iData, volatile threadgroup T *sData);

template<typename T>
device T warpScanExclusive(uint threadIdx, T iData, volatile threadgroup T *sData);

template<typename T>
device T scan1Inclusive(uint threadIdx, T iData, volatile threadgroup T *sData);

device void streamCompactWriteOut(
                                  volatile threadgroup int *shmem,
                                  threadgroup int &numPassed,
                                  threadgroup int &outMaskOffset,
                                  uint threadIdx,
                                  int threadPassFlag,
                                  int threadElem,
                                  device int *vectorOut,
                                  device atomic_int &d_outMaskPosition
                                  );

device float subReduceAdd(uint threadIdx, float current, volatile threadgroup float* sharedMemory);
device int subReduceAdd(uint threadIdx, int current, volatile threadgroup int* sharedMemory);
device float4 subReduceAdd(uint threadIdx, float4 current, volatile threadgroup float4* sharedMemory);

device void streamCompactWriteOutToSharedMemory(
                                                volatile threadgroup int *shmem,
                                                threadgroup int &numPassed,
                                                threadgroup int &outMaskOffset,
                                                uint threadIdx,
                                                int threadPassFlag,
                                                int threadElem,
                                                volatile threadgroup int *vectorOut
                                                );

