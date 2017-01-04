//
//  Integral.metal
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 07.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

#include "IntegralDefinitions.metal"
using namespace metal;

float calculateSum(texture2d<float, access::read> image, uint x, uint y, uint rwidth, uint rheight) {
    
    uint height = image.get_height();
    uint width = image.get_width();
    
    int r1 = min(y, height)-1;
    int c1 = min(x, width)-1;
    int r2 = min(y+rheight, height)-1;
    int c2 = min(x+rwidth, width)-1;
    
    float A(0.0f), B(0.0f), C(0.0f), D(0.0f);
    
    if (r1 >= 0 && c1 >= 0) A = image.read(uint2(c1,r1)).r;
    if (r1 >= 0 && c2 >= 0) B = image.read(uint2(c2,r1)).r;
    if (r2 >= 0 && c1 >= 0) C = image.read(uint2(c1,r2)).r;
    if (r2 >= 0 && c2 >= 0) D = image.read(uint2(c2,r2)).r;
    
    return max(0.f, A-B-C+D);
}
