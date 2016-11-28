//
//  FaceDetectionKernels.metal
//
//  Created by Christopher Helf on 12.11.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

struct Rect {
    float x;
    float y;
    float width;
    float height;
};

struct FeatureRect {
    Rect r;
    float weight;
};

struct HaarFeature {
    FeatureRect r1;
    FeatureRect r2;
    FeatureRect r3;
};

struct HaarClassifier {
    HaarFeature haarFeature;
    float threshold;
    float alpha0;
    float alpha1;
};

struct HaarStageClassifier {
    int numOfClassifiers;
    int classifierOffset;
    float threshold;
};

struct HaarCascade {
    int numOfStages;
    int totalNumOfClassifiers;
    int originalWindowSizeWidth;
    int originalWindowSizeHeight;
    int windowSizeWidth;
    int windowSizeHeight;
    int detectionSizeWidth;
    int detectionSizeHeight;
    int realWindowSizeWidth;
    int realWindowSizeHeight;
    float step;
};

float calculateSum(texture2d<float, access::read> integralImage, thread Rect &rect, int2 start);

float calculateSum(texture2d<float, access::read> integralImage, thread Rect &rect, int2 start) {
    
    int height = integralImage.get_height();
    int width = integralImage.get_width();
    
    int r1 = min((int)rect.y+start.y, height)-1;
    int c1 = min((int)rect.x+start.x, width)-1;
    int r2 = min((int)(rect.y+rect.height+start.y), height)-1;
    int c2 = min((int)(rect.x+rect.width+start.x), width)-1;
    
    float A(0.0f), B(0.0f), C(0.0f), D(0.0f);
    
    if (r1 >= 0 && c1 >= 0) A = integralImage.read(uint2(c1,r1)).r;
    if (r1 >= 0 && c2 >= 0) B = integralImage.read(uint2(c2,r1)).r;
    if (r2 >= 0 && c1 >= 0) C = integralImage.read(uint2(c1,r2)).r;
    if (r2 >= 0 && c2 >= 0) D = integralImage.read(uint2(c2,r2)).r;
    
    return max(0.f, A-B-C+D);
}


float calculateMean(texture2d<float, access::read> integralImage, thread Rect &rect);

float calculateMean(texture2d<float, access::read> integralImage, thread Rect &rect)
{
    int height = integralImage.get_height();
    int width = integralImage.get_width();
    
    int r1 = min((int)rect.y, height)-1;
    int c1 = min((int)rect.x, width)-1;
    int r2 = min((int)(rect.y+rect.height), height)-1;
    int c2 = min((int)(rect.x+rect.width), width)-1;
    
    float A(0.0f), B(0.0f), C(0.0f), D(0.0f);
    if (r1 >= 0 && c1 >= 0) A = integralImage.read(uint2(c1,r1)).r;
    if (r1 >= 0 && c2 >= 0) B = integralImage.read(uint2(c2,r1)).r;
    if (r2 >= 0 && c2 >= 0) C = integralImage.read(uint2(c2,r2)).r;
    if (r2 >= 0 && c1 >= 0) D = integralImage.read(uint2(c1,r2)).r;
    
    return max(0.f, A-B+C-D);
}

float calculateVarianceNormFactor(texture2d<float, access::read> sqIntegralImage, thread Rect &rect);

float calculateVarianceNormFactor(texture2d<float, access::read> sqIntegralImage, thread Rect &rect) {
    
    int height = sqIntegralImage.get_height();
    int width = sqIntegralImage.get_width();
    
    int r1 = min((int)rect.y, height)-1;
    int c1 = min((int)rect.x, width)-1;
    int r2 = min((int)(rect.y+rect.height), height)-1;
    int c2 = min((int)(rect.x+rect.width), width)-1;
    
    float A(0.0f), B(0.0f), C(0.0f), D(0.0f);
    if (r1 >= 0 && c1 >= 0) A = sqIntegralImage.read(uint2(c1,r1)).r;
    if (r1 >= 0 && c2 >= 0) B = sqIntegralImage.read(uint2(c2,r1)).r;
    if (r2 >= 0 && c1 >= 0) C = sqIntegralImage.read(uint2(c1,r2)).r;
    if (r2 >= 0 && c2 >= 0) D = sqIntegralImage.read(uint2(c2,r2)).r;
    
    return max(0.f, A-B-C+D);
}


float calculateVariance(texture2d<float, access::read> integralImage, texture2d<float, access::read> sqIntegralImage, thread Rect &detectionWindow);

float calculateVariance(texture2d<float, access::read> integralImage, texture2d<float, access::read> sqIntegralImage, thread Rect &detectionWindow) {
   
    float inverseWindowArea = 1.0f / (detectionWindow.width * detectionWindow.height);
    float mean = calculateMean(integralImage, detectionWindow) * inverseWindowArea;
    float varianceNormFactor = calculateVarianceNormFactor(sqIntegralImage, detectionWindow);
    varianceNormFactor = varianceNormFactor * inverseWindowArea - mean * mean;
    
    if (varianceNormFactor >= 0.f) {
        return sqrt(varianceNormFactor);
    } else {
        return 1.0f;
    }
}

float runHaarFeature(texture2d<float, access::read> integralImage, thread HaarClassifier &classifier, thread Rect &detectionWindow, float varianceNormFactor, float weightScale);

float runHaarFeature(texture2d<float, access::read> integralImage, thread HaarClassifier &classifier, thread Rect &detectionWindow, float varianceNormFactor, float weightScale) {
    
    float t = classifier.threshold * varianceNormFactor;
    int2 offset = int2(detectionWindow.x, detectionWindow.y);
    
    float sum = calculateSum(integralImage, classifier.haarFeature.r1.r, offset) * classifier.haarFeature.r1.weight * weightScale;
    sum += calculateSum(integralImage, classifier.haarFeature.r2.r, offset) * classifier.haarFeature.r2.weight * weightScale;
    
    if (classifier.haarFeature.r3.weight > 0) {
        sum += calculateSum(integralImage, classifier.haarFeature.r3.r, offset) * classifier.haarFeature.r3.weight * weightScale;
    }
    
    if (sum >= t) {
        return classifier.alpha1;
    } else {
        return classifier.alpha0;
    }
    
}



void drawLine(texture2d<float, access::write> targetTexture, uint2 start, uint2 end, float4 color);
void drawLine(texture2d<float, access::write> targetTexture, uint2 start, uint2 end, float4 color)
{
    int iterations = 0;
    
    int x = int(start.x);
    int y = int(start.y);
    
    int dx = abs(x - int(end.x));
    int dy = abs(y - int(end.y));
    
    int sx = start.x < end.x ? 1 : -1;
    int sy = start.y < end.y ? 1 : -1;
    
    int err = (dx > dy ? dx : -dy) / 2;
    
    int width = int(targetTexture.get_width());
    int height = int(targetTexture.get_height());
    
    while (true)
    {
        if (x > 0 && y > 0 && x < width && y < height)
        {
            targetTexture.write(color, uint2(x,y));
        }
        
        iterations++;
        
        if (iterations > 600 || (x == int(end.x) && y == int(end.y)))
        {
            break;
        }
        
        int e2 = err;
        
        if (e2 > -dx)
        {
            err -= dy;
            x += sx;
        }
        
        if (e2 < dy)
        {
            err += dx;
            y += sy;
        }
    }
}

void drawRectangle(thread Rect &r, texture2d<float, access::write> output, float4 color);
void drawRectangle(thread Rect &r, texture2d<float, access::write> output, float4 color) {
    drawLine(output, uint2(r.x,r.y), uint2(r.x+r.width,r.y),color);
    drawLine(output, uint2(r.x+r.width,r.y), uint2(r.x+r.width,r.y+r.height),color);
    drawLine(output, uint2(r.x+r.width,r.y+r.height), uint2(r.x,r.y+r.height),color);
    drawLine(output, uint2(r.x,r.y+r.height), uint2(r.x,r.y),color);
}

kernel void haarCopyClassifiers(
    constant int &num [[buffer(0)]],
    constant HaarClassifier *cpuBuffer [[buffer(1)]],
    device HaarClassifier *gpuBuffer [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int pos = gid.x;
    if (pos < num) {
         gpuBuffer[pos] = cpuBuffer[pos];
    }
}

kernel void haarDetection(
                          constant HaarCascade &haarCascade [[buffer(0)]],
                          constant HaarStageClassifier *haarStageClassifiers [[buffer(1)]],
                          constant HaarClassifier *scaledHaarClassifiers [[buffer(2)]],
                          device atomic_int *faceCounter [[buffer(3)]],
                          device Rect* detectedFaces [[buffer(4)]],
                          texture2d<float, access::read> integralImage [[texture(0)]],
                          texture2d<float, access::read> sqIntegralImage [[texture(1)]],
                          texture2d<float, access::write> output [[texture(2)]],
                          uint2 blockIdx [[threadgroup_position_in_grid]],
                          uint2 threadIdx [[ thread_position_in_threadgroup ]],
                          uint2 gid [[thread_position_in_grid]]
                          ) {
    
    int x = gid.x * haarCascade.step;
    int y = gid.y * haarCascade.step;
    
    if (x < haarCascade.detectionSizeWidth || y < haarCascade.detectionSizeHeight) {
        
        Rect detectionWindow;
        detectionWindow.x = x;
        detectionWindow.y = y;
        detectionWindow.width = haarCascade.realWindowSizeWidth;
        detectionWindow.height = haarCascade.realWindowSizeHeight;
        
        float varianceNormFactor = calculateVariance(integralImage, sqIntegralImage, detectionWindow);
        float inverseWindowArea = 1.0f / (detectionWindow.width * detectionWindow.height);
        
        for(int i=0; i<haarCascade.numOfStages; i++) {
            
            float stageSum = 0.0;
            
            for(int j=0; j<haarStageClassifiers[i].numOfClassifiers; j++) {
                int index = j + haarStageClassifiers[i].classifierOffset;
                HaarClassifier classifier = scaledHaarClassifiers[index];
                stageSum += runHaarFeature(integralImage, classifier, detectionWindow, varianceNormFactor, inverseWindowArea);
            }
            
            if (stageSum < haarStageClassifiers[i].threshold) {
                return;
            }
        }
        
        int currentFace = atomic_fetch_add_explicit( faceCounter, 1, memory_order_relaxed );
        detectedFaces[currentFace]=detectionWindow;
        //drawRectangle(detectionWindow, output, float4(1,0,0,1));
        
    }
    
}

kernel void haarDrawRects(
                          constant Rect* detectedFaces [[buffer(0)]],
                          texture2d<float, access::write> output [[texture(0)]],
                          uint2 blockIdx [[threadgroup_position_in_grid]],
                          uint2 threadIdx [[ thread_position_in_threadgroup ]],
                          uint2 gid [[thread_position_in_grid]]
                          ) {
    
    int x = gid.x;
    Rect face = detectedFaces[x];
    drawRectangle(face, output, float4(0,1,0,1));
}






