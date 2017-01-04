//
//  HaarStructsTests.metal
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 06.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

#include "HaarStructsDefinitions.metal"
using namespace metal;

kernel void testHaarFeatureRect(
                             HaarFeatureRectTexture tex [[ texture(0) ]],
                             device uint* x [[ buffer(0) ]],
                             device uint* y [[ buffer(1) ]],
                             device uint* width [[ buffer(2) ]],
                             device uint* height [[ buffer(3) ]],
                             device float* weight [[ buffer(4) ]],
                             uint2 threadIdx [[ thread_position_in_threadgroup ]]
                             ) {
    
    uint2 pos = uint2(threadIdx.x, 1);
    HaarFeatureRect rect = getHaarFeatureRect(pos.x, tex, 1.0);
    
    x[threadIdx.x] = rect.x;
    y[threadIdx.x] = rect.y;
    width[threadIdx.x] = rect.width;
    height[threadIdx.x] = rect.height;
    weight[threadIdx.x] = rect.weight;
    
}

kernel void testHaarClassifier(
                                HaarClassifierTexture tex [[ texture(0) ]],
                                device uint* numRects [[ buffer(0) ]],
                                device uint* firstRectOffset [[ buffer(1) ]],
                                device float* threshold [[ buffer(2) ]],
                                device float* left [[ buffer(3) ]],
                                device float* right [[ buffer(4) ]],
                                uint2 threadIdx [[ thread_position_in_threadgroup ]]
                                ) {
    
    uint2 pos = uint2(threadIdx.x, 1);
    HaarClassifier rect = getHaarClassifier(pos.x, tex);
    
    numRects[threadIdx.x] = rect.numRects;
    firstRectOffset[threadIdx.x] = rect.firstRectOffset;
    threshold[threadIdx.x] = rect.threshold;
    left[threadIdx.x] = rect.left;
    right[threadIdx.x] = rect.right;
    
}

kernel void testHaarStage(
                          constant HaarStageStruct* tex [[ buffer(0) ]],
                          device uint* firstNodeOffset [[ buffer(1) ]],
                          device uint* numberOfTrees [[ buffer(2) ]],
                          device float* stageThreshold [[ buffer(3) ]],
                          uint2 threadIdx [[ thread_position_in_threadgroup ]]
) {
    
    uint2 pos = uint2(threadIdx.x, 1);
    HaarStage rect = getHaarStage(pos.x, tex);
    
    firstNodeOffset[threadIdx.x] = rect.firstNodeOffset;
    numberOfTrees[threadIdx.x] = rect.numberOfTrees;
    stageThreshold[threadIdx.x] = rect.stageThreshold;

    
}
