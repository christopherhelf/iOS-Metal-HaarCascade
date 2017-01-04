//
//  HaarCascadeHelpers.metal
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 09.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

#include <metal_stdlib>
#include "HaarStructsDefinitions.metal"
#include "StreamCompactionDefinitions.metal"
#include "IntegralDefinitions.metal"
using namespace metal;

kernel void haarCascadeResetCounters(
                                     device uint3 &threadgroupsIndirectArguments [[ buffer(0) ]],
                                     device int &outMaskCounter [[ buffer(1) ]],
                                     device int &previousOutMaskCounter [[ buffer(2) ]]
                                     ) {
    
    previousOutMaskCounter = outMaskCounter;
    outMaskCounter = 0;
    threadgroupsIndirectArguments = uint3((previousOutMaskCounter + numThreadsAnchorsParallel - 1) / numThreadsAnchorsParallel,1,1);
}

kernel void haarCascadeSetCascadeCounter(
                                     device uint3 &threadgroupsIndirectArguments [[ buffer(0) ]],
                                     device int &outMaskCounter [[ buffer(1) ]]
                                     ) {
    
    threadgroupsIndirectArguments = uint3(outMaskCounter,1,1);
}

kernel void haarCascadeResetOutputCounter(device int &outMaskCounter [[ buffer(0) ]]) {
    outMaskCounter = 0;
}
