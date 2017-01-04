//
//  New.metal
//  iOS-Metal-HaarCascade
//
//  Created by Christopher Helf on 04.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

#include "StreamCompactionDefinitions.metal"
using namespace metal;

/**
    Inclusive warp scan, requires warp size 32 due to unrolled look
    basically splits operations up into two arrays (for n=64), one for each
    warp and sets/sums up values
 */
template<typename T>
device T warpScanInclusive(uint threadIdx, T iData, volatile threadgroup T *sData)
{
    int pos = 2 * threadIdx - (threadIdx & (kWarpSize - 1));
    sData[pos] = 0;
    pos += kWarpSize;
    sData[pos] = iData;

    // loop here unrolled for n=32
    sData[pos] += sData[pos - 1];
    sData[pos] += sData[pos - 2];
    sData[pos] += sData[pos - 4];
    sData[pos] += sData[pos - 8];
    sData[pos] += sData[pos - 16];
    
    /*for(uint offset = 1; offset < kWarpSize; offset <<= 1){
        simdgroup_barrier( mem_flags::mem_threadgroup );
        T t = sData[pos] + sData[pos - offset];
        simdgroup_barrier( mem_flags::mem_threadgroup );
        sData[pos] = t;
    }*/
    
    return sData[pos];
}

/**
    Same as warpScanInclusive, but deducts the initial value
*/
template<typename T>
device T warpScanExclusive(uint threadIdx, T iData, volatile threadgroup T *sData)
{
    return warpScanInclusive<T>(threadIdx, iData, sData) - iData;
}

/**
    Parallel scan, with iData being the input data for the threadgroup shared memory sData
    at the position threadIdx
*/
template<typename T>
device T scan1Inclusive(uint threadIdx, T iData, volatile threadgroup T *sData) {
    
    // multiple warps
    if (numThreadsAnchorsParallel > kWarpSize)
    {
        // each warp scans individualy
        T warpResult = warpScanInclusive<T>(threadIdx, iData, sData);
        
        // wait for all warp scans
        simdgroup_barrier( mem_flags::mem_threadgroup );
        
        // assign warp results
        if( (threadIdx & (kWarpSize - 1)) == (kWarpSize - 1) )
        {
            // positions 0 and 1 for 31 and 63
            sData[threadIdx >> kLog2WarpSize] = warpResult;
        }
        
        //wait for individual warp scans to complete
        threadgroup_barrier( mem_flags::mem_threadgroup );
        
        // first two threads scan again
        if( threadIdx < (numThreadsAnchorsParallel / kWarpSize) )
        {
            // 0 and 1
            //grab top warp elements
            T val = sData[threadIdx];
            //calculate exclusive scan and write back to shared memory
            sData[threadIdx] = warpScanExclusive<T>(threadIdx, val, sData);
        }
        
        //return updated warp scans with exclusive scan results
        threadgroup_barrier( mem_flags::mem_threadgroup );
        
        return warpResult + sData[threadIdx >> kLog2WarpSize];
    }
    // single warp
    else
    {
        return warpScanInclusive<T>(threadIdx, iData, sData);
    }
    
    
}

device void streamCompactWriteOut(
                                               volatile threadgroup int *shmem,
                                               threadgroup int &numPassed,
                                               threadgroup int &outMaskOffset,
                                               uint threadIdx,
                                               int threadPassFlag,
                                               int threadElem,
                                               device int *vectorOut,
                                               device atomic_int &d_outMaskPosition
                                               )
{
    
    int incScan = scan1Inclusive<int>(threadIdx, threadPassFlag, shmem);
    
    threadgroup_barrier( mem_flags::mem_threadgroup );
    
    if (threadIdx == numThreadsAnchorsParallel-1)
    {
        numPassed = incScan;
        outMaskOffset = atomic_fetch_add_explicit( &d_outMaskPosition, incScan, memory_order_relaxed );
    }
    
    if (threadPassFlag)
    {
        int excScan = incScan - threadPassFlag;
        shmem[excScan] = threadElem;
    }
    
    threadgroup_barrier( mem_flags::mem_threadgroup );
    
    if (int(threadIdx) < numPassed)
    {
        vectorOut[outMaskOffset + threadIdx] = shmem[threadIdx];
    }

}

device void streamCompactWriteOutToSharedMemory(
                                  volatile threadgroup int *shmem,
                                  threadgroup int &numPassed,
                                  threadgroup int &outMaskOffset,
                                  uint threadIdx,
                                  int threadPassFlag,
                                  int threadElem,
                                  volatile threadgroup int *vectorOut
                                  )
{
    
    int incScan = scan1Inclusive<int>(threadIdx, threadPassFlag, shmem);
    
    threadgroup_barrier( mem_flags::mem_threadgroup );
    
    if (threadIdx == numThreadsAnchorsParallel-1)
    {
        numPassed = incScan;
    }
    
    if (threadPassFlag)
    {
        int excScan = incScan - threadPassFlag;
        shmem[excScan] = threadElem;
    }
    
    threadgroup_barrier( mem_flags::mem_threadgroup );
    
    if (int(threadIdx) < numPassed)
    {
        vectorOut[outMaskOffset + threadIdx] = shmem[threadIdx];
    }
    
    if (threadIdx == numThreadsAnchorsParallel-1) {
        outMaskOffset = outMaskOffset + incScan;
    }
    
    threadgroup_barrier( mem_flags::mem_threadgroup );
    
}





/**
    Performs a parallel scan and adds the values for the current thread index
    specified via the current argument, this function is fixed for
    64 parallel threads
*/
device float subReduceAdd(uint threadIdx, float current, volatile threadgroup float* sharedMemory) {
    
    // each warp scans individually
    warpScanInclusive<float>(threadIdx,current,sharedMemory);
    
    // wait for all warp scans
    simdgroup_barrier( mem_flags::mem_threadgroup );
    
    // assign warp results
    if( (threadIdx & (kWarpSize - 1)) == (kWarpSize - 1) )
    {
        // positions 0 and 1 for 31 and 63
        uint warp = threadIdx/kWarpSize;
        sharedMemory[warp] = sharedMemory[(threadIdx+1)*2-1];
    }
    
    //wait for individual warp scans to complete
    threadgroup_barrier( mem_flags::mem_threadgroup );
    
    // only one thread assigns sum, this is FIXED for 64 threads
    if (threadIdx == 0) {
        float sum = sharedMemory[0] + sharedMemory[1];
        sharedMemory[0] = sum;
    }
    
    // wait for first thread to sum
    threadgroup_barrier( mem_flags::mem_threadgroup );
    
    // return the value
    return sharedMemory[0];
}

// float4 version
device float4 subReduceAdd(uint threadIdx, float4 current, volatile threadgroup float4* sharedMemory) {
    
    // each warp scans individually
    warpScanInclusive<float4>(threadIdx,current,sharedMemory);
    
    // wait for all warp scans
    simdgroup_barrier( mem_flags::mem_threadgroup );
    
    // assign warp results
    if( (threadIdx & (kWarpSize - 1)) == (kWarpSize - 1) )
    {
        // positions 0 and 1 for 31 and 63
        uint warp = threadIdx/kWarpSize;
        sharedMemory[warp] = sharedMemory[(threadIdx+1)*2-1];
    }
    
    //wait for individual warp scans to complete
    threadgroup_barrier( mem_flags::mem_threadgroup );
    
    // only one thread assigns sum, this is FIXED for 64 threads
    if (threadIdx == 0) {
        float4 sum = sharedMemory[0] + sharedMemory[1];
        sharedMemory[0] = sum;
    }
    
    // wait for first thread to sum
    threadgroup_barrier( mem_flags::mem_threadgroup );
    
    // return the value
    return sharedMemory[0];
}


// the integer version
device int subReduceAdd(uint threadIdx, int current, volatile threadgroup int* sharedMemory) {
    
    // each warp scans individually
    warpScanInclusive<int>(threadIdx,current,sharedMemory);
    
    // wait for all warp scans
    simdgroup_barrier( mem_flags::mem_threadgroup );
    
    // assign warp results
    if( (threadIdx & (kWarpSize - 1)) == (kWarpSize - 1) )
    {
        // positions 0 and 1 for 31 and 63
        uint warp = threadIdx/kWarpSize;
        sharedMemory[warp] = sharedMemory[(threadIdx+1)*2-1];
    }
    
    //wait for individual warp scans to complete
    threadgroup_barrier( mem_flags::mem_threadgroup );
    
    // only one thread assigns sum, this is FIXED for 64 threads
    if (threadIdx == 0) {
        int sum = sharedMemory[0] + sharedMemory[1];
        sharedMemory[0] = sum;
    }
    
    // wait for first thread to sum
    threadgroup_barrier( mem_flags::mem_threadgroup );
    
    // return the value
    return sharedMemory[0];
}

