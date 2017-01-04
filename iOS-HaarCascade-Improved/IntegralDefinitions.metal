//
//  IntegralDefinitions.metal
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 07.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

float calculateSum(texture2d<float, access::read> image, uint x, uint y, uint rwidth, uint rheight);

