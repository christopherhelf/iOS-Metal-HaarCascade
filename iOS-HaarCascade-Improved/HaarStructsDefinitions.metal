//
//  Definitions.metal
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 07.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

constant float eps [[ function_constant(4) ]];
constant int minNeighbors [[ function_constant(5) ]];

// the current detection window, using only within the shader
struct DetectionWindow {
    uint x;
    uint y;
    uint width;
    uint height;
};

// struct for storing information about the current haar cascade
struct HaarCascade {
    float scale;
    float scaledArea;
    uint width;
    uint height;
};

// store feature rectangles
// consisting of x,y,width,height and weight into three float16s

// uint32, uint32
#define HaarFeatureRectTexture texture2d<uint, access::read>

struct HaarFeatureRect {
    uint x; // uint8
    uint y; // uint8
    uint width; // uint8
    uint height; // uint8
    float weight; // float32
};


// float32, float32, float32, float32
#define HaarClassifierTexture texture2d<float, access::read>

struct HaarClassifier {
    uint numRects; // uint8
    uint firstRectOffset; // uint24
    float threshold; // float32
    float left; // float32
    float right; // float32
};

// float32, float32
#define HaarStageStruct packed_float2

struct HaarStage {
    uint firstNodeOffset; // uint16
    uint numberOfTrees; // uint16
    float stageThreshold; // float32
};

HaarFeatureRect getHaarFeatureRect(uint pos, HaarFeatureRectTexture texture, float scale);
HaarClassifier getHaarClassifier(uint pos, HaarClassifierTexture texture);
HaarStage getHaarStage(uint pos, constant HaarStageStruct* buffer);
//HaarStage getHaarStage(constant HaarStageStruct &values);


