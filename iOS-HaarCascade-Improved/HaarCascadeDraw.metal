//
//  HaarCascadeDraw.metal
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 10.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

#include <metal_stdlib>
#include "HaarStructsDefinitions.metal"
#include "StreamCompactionDefinitions.metal"
#include "IntegralDefinitions.metal"
using namespace metal;

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
        
        if (iterations > 720 || (x == int(end.x) && y == int(end.y)))
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

void drawRectangle(thread DetectionWindow &r, texture2d<float, access::write> output, float4 color);
void drawRectangle(thread DetectionWindow &r, texture2d<float, access::write> output, float4 color) {
    drawLine(output, uint2(r.x,r.y), uint2(r.x+r.width,r.y),color);
    drawLine(output, uint2(r.x+r.width,r.y), uint2(r.x+r.width,r.y+r.height),color);
    drawLine(output, uint2(r.x+r.width,r.y+r.height), uint2(r.x,r.y+r.height),color);
    drawLine(output, uint2(r.x,r.y+r.height), uint2(r.x,r.y),color);
}

kernel void haarCascadeDraw(
    constant float4* input [[ buffer(0) ]],
    constant float &scale [[ buffer(1) ]],
    constant int &numWindows [[ buffer(2) ]],
    texture2d<float, access::write> outputTex [[ texture(0)]],
    uint2 globalIdx [[thread_position_in_grid]]
) {
    
    if ((int)globalIdx.x >= numWindows) {
        return;
    }
    
    float4 in = input[globalIdx.x];
    DetectionWindow w;
    w.x = in[0];
    w.y = in[1];
    w.width = in[2];
    w.height = in[3];
    
    drawRectangle(w, outputTex, float4(1,0,0,1));
    
}

kernel void haarCascadeSetDrawDispatchArguments(
                                         device uint3 &threadgroupsIndirectArguments [[ buffer(0) ]],
                                         device int &outMaskCounter [[ buffer(1) ]]
                                         ) {
    
    threadgroupsIndirectArguments = uint3((outMaskCounter + numThreadsAnchorsParallel - 1) / numThreadsAnchorsParallel,1,1);
}


