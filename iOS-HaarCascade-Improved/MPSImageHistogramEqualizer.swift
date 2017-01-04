//
//  MPSImageHistogramEqualizer.swift
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 04.01.17.
//  Copyright © 2017 Christopher Helf. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

// convenience class for histogram equalization
class MPSImageHistogramEqualizer : MetalPerformanceShaderKernel {
    
    var bufferReference : MTLBuffer
    var histogramInfo = MPSImageHistogramInfo(
        numberOfHistogramEntries: 256,
        histogramForAlpha: false,
        minPixelValue: vector_float4(0,0,0,0),
        maxPixelValue: vector_float4(1,1,1,1))
    
    let histogram : MPSImageHistogram
    let eqHistograms : MPSImageHistogramEqualization
    
    init(device: MTLDevice) {
        self.bufferReference = Context.device().makeBuffer(bytes: &histogramInfo, length: MemoryLayout.size(ofValue: histogramInfo), options: MTLResourceOptions.optionCPUCacheModeWriteCombined)
        self.bufferReference.label = "histogramInfoBuffer"
        histogram = MPSImageHistogram(device: Context.device(), histogramInfo: &histogramInfo)
        histogram.label = "imageHistogram"
        eqHistograms = MPSImageHistogramEqualization(device: Context.device(), histogramInfo: &histogramInfo)
        eqHistograms.label = "imageHistogramEqualization"
    }
    
    func encode(commandBuffer: MTLCommandBuffer, inPlaceTexture texture: UnsafeMutablePointer<MTLTexture>, fallbackCopyAllocator copyAllocator: MPSCopyAllocator?) -> Bool {
        
        // histogram
        self.histogram.encode(to: commandBuffer, sourceTexture: texture.pointee, histogram: self.bufferReference, histogramOffset: 0)
        
        // equalize
        self.eqHistograms.encodeTransform(to: commandBuffer, sourceTexture: texture.pointee, histogram: self.bufferReference, histogramOffset: 0)
        return self.eqHistograms.encode(commandBuffer: commandBuffer, inPlaceTexture: texture, fallbackCopyAllocator: copyAllocator)
    }
    
    func encode(commandBuffer: MTLCommandBuffer, sourceTexture: MTLTexture, destinationTexture: MTLTexture) {
        
        // histogram
        self.histogram.encode(to: commandBuffer, sourceTexture: sourceTexture, histogram: self.bufferReference, histogramOffset: 0)
        
        // equalize
        self.eqHistograms.encodeTransform(to: commandBuffer, sourceTexture: sourceTexture, histogram: self.bufferReference, histogramOffset: 0)
        self.eqHistograms.encode(commandBuffer: commandBuffer, sourceTexture: sourceTexture, destinationTexture: destinationTexture)
        
    }

}
