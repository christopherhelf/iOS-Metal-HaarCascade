//
//  ViewFunctions.metal
//
//  Created by Christopher Helf on 10.11.16.
//  Copyright © 2016 Christopher Helf. All rights reserved..
//

#include <metal_stdlib>
using namespace metal;


struct VertexIn {
    packed_float3 position;
    packed_float4 color;
    packed_float2 textureCoordinate;
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
    float2 textureCoordinate;
};

vertex VertexOut basic_vertex(const device VertexIn *vertex_array [[ buffer(0) ]],
                              unsigned int vid [[ vertex_id ]])
{
    VertexOut out;
    out.position = float4(vertex_array[vid].position * float3(-1.0, 1.0, 1.0), 1.0);
    out.color = vertex_array[vid].color;
    out.textureCoordinate = vertex_array[vid].textureCoordinate;
    
    return out;
}

fragment float4 copy(VertexOut interpolated [[stage_in]],
                     texture2d<float> tex2D [[ texture(0) ]],
                     sampler sampler2D [[ sampler(0) ]])
{
    return tex2D.sample(sampler2D, interpolated.textureCoordinate);
}
