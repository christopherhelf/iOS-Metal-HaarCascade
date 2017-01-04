//
//  HaarCascadePixelParallelFromBuffer.metal
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

kernel void haarCascadePixelParallelFromBuffer(
                                                constant HaarCascade &haarCascade [[buffer(0)]],
                                                constant uint &startStage [[buffer(1)]],
                                                constant uint &endStage [[buffer(2)]],
                                               
                                                constant HaarStageStruct *haarStageBuffer [[ buffer(3) ]],
                                                HaarClassifierTexture haarClassifierTexture [[texture(0)]],
                                                HaarFeatureRectTexture haarFeatureRectTexture [[texture(1)]],
                                                texture2d<float, access::read> integralImage [[texture(2)]],
                                               
                                                threadgroup int* sharedMemory [[threadgroup(0)]],
                                                threadgroup int& sharedStreamCompactNumPassed [[threadgroup(1)]],
                                                threadgroup int& sharedOutMaskOffset [[threadgroup(2)]],
                                               
                                                device int &previousOutMaskCounter [[ buffer(4) ]],
                                                device atomic_int &outMaskCounter [[ buffer(5) ]],
                                                
                                                device int* outMask [[ buffer(6) ]],
                                                texture2d<float, access::read> varianceMap [[texture(3)]],
                                               
                                                uint2 blockIdx [[threadgroup_position_in_grid]],
                                                uint2 threadIdx [[ thread_position_in_threadgroup ]]
                                                
                                                ) {
 
    // get the current position within the buffer
    uint pos = blockIdx.x * numThreadsAnchorsParallel + threadIdx.x;
    
    // check whether this thread is inactive
    bool isInactiveThread = pos >= (uint) previousOutMaskCounter;
    
    // read from the buffer and gather variance and x/y
    uint outMaskVal = (uint) outMask[pos];
    uint yOffset = outMaskVal >> 16;
    uint xOffset = outMaskVal & 0xFFFF;
    float normVariance = varianceMap.read(uint2(xOffset,yOffset)).r;
    
    // the current detection window
    DetectionWindow window;
    window.x = xOffset;
    window.y = yOffset;
    window.width = haarCascade.width;
    window.height = haarCascade.height;
    
    // whether we passed all stages
    bool bPass = true;
    
    // iterate through the stages
    for(uint iStage = startStage; iStage < endStage; iStage++) {
        
        // break inactive threads immediately
        if(isInactiveThread) {
            break;
        }
        
        // the current sum of the stage
        float currentStageSum = 0.0f;
        
        // the specifics for the current stage
        HaarStage currentStage = getHaarStage(iStage, haarStageBuffer);
        
        // stage threshold, and offset etc
        float stageThreshold = currentStage.stageThreshold;
        uint currentRootNodeOffset = currentStage.firstNodeOffset;
        uint numberOfNodes = currentStage.numberOfTrees;
        
        // iterate through all nodes in the stage
        for(uint iNode = 0; iNode < numberOfNodes; iNode++) {
            
            // the current offset
            uint currentNodeOffset = currentRootNodeOffset + iNode;
            
            // get the current node
            HaarClassifier currentNode = getHaarClassifier(currentNodeOffset, haarClassifierTexture);
            
            // current feature offset
            uint currentFeatureOffset = currentNode.firstRectOffset;
            
            // the sum of the features
            float currentNodeSum = 0.0f;
            
            // iterate through all rectangles of the feature
            for(uint i=0; i<currentNode.numRects; i++) {
                
                // get the feature
                HaarFeatureRect rect = getHaarFeatureRect(currentFeatureOffset, haarFeatureRectTexture, haarCascade.scale);
                
                // get the response
                uint rleft = rect.x + window.x;
                uint rtop = rect.y + window.y;
                uint rwidth = rect.width;
                uint rheight = rect.height;
                
                float rectResponse = calculateSum(integralImage, rleft, rtop, rwidth, rheight);
                currentNodeSum += rectResponse * rect.weight;
                
                // increase the feature offset
                currentFeatureOffset += 1;
            }
            
            // left leaf value
            if (currentNodeSum < haarCascade.scaledArea * normVariance * currentNode.threshold) {
                currentStageSum += currentNode.left;
            }
            // right leaf value
            else {
                currentStageSum += currentNode.right;
            }
        }
        
        // stage didnt pass, break
        if (currentStageSum < stageThreshold) {
            bPass = false;
            break;
        }
        
        
    }
    
    // wait for all threads before writing out
    threadgroup_barrier( mem_flags::mem_threadgroup );
    
    // and write out
    streamCompactWriteOut(
                          sharedMemory,
                          sharedStreamCompactNumPassed,
                          sharedOutMaskOffset,
                          threadIdx.x,
                          bPass && !isInactiveThread,
                          outMaskVal,
                          outMask,
                          outMaskCounter
                          );
    
    
}


