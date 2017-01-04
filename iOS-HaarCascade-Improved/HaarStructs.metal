//
//  HaarStructs.metal
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 06.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

#include <metal_stdlib>
#include <metal_compute>
#include "HaarStructsDefinitions.metal"
using namespace metal;

HaarFeatureRect getHaarFeatureRect(uint pos, HaarFeatureRectTexture texture, float scale) {
    
    uint4 values = texture.read(uint2(pos,0));
    
    HaarFeatureRect rect;
    rect.height = (uint) (values[0] >> 24) * scale;
    rect.width  = (uint) ((values[0] & 0xFFFFFF) >> 16) * scale ;
    rect.y =      (uint) ((values[0] & 0xFFFF) >> 8) * scale ;
    rect.x =      (uint) ((values[0] & 0xF)) * scale ;
    rect.weight =  as_type<float>(values[1]);
    
    return rect;
}

HaarClassifier getHaarClassifier(uint pos, HaarClassifierTexture texture) {
    
    float4 values = texture.read(uint2(pos,0));
    
    HaarClassifier classifier;
    classifier.numRects = as_type<uint>(values[0]) >> 24;
    classifier.firstRectOffset = as_type<uint>(values[0]) & 0xFFFFF;
    classifier.threshold = values[1];
    classifier.left = values[2];
    classifier.right = values[3];
    
    return classifier;
}

HaarStage getHaarStage(uint pos, constant HaarStageStruct* buffer) {
    
    packed_float2 values = buffer[pos];

    HaarStage stage;
    stage.firstNodeOffset = as_type<uint>(values[0]) >> 16;
    stage.numberOfTrees = as_type<uint>(values[0]) & 0xFFFF;
    stage.stageThreshold = values[1];
    
    return stage;
}







