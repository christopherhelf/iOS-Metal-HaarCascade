//
//  MetalPerformanceShader.swift
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 29.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

// this protocol is being used to store other mps-based classes
// as there the common MPSKernel class doesnt have any encoding functions
protocol MetalPerformanceShaderKernel {
    func encode(commandBuffer: MTLCommandBuffer, sourceTexture: MTLTexture, destinationTexture: MTLTexture)
    func encode(commandBuffer: MTLCommandBuffer, inPlaceTexture texture: UnsafeMutablePointer<MTLTexture>, fallbackCopyAllocator copyAllocator: MetalPerformanceShaders.MPSCopyAllocator?) -> Bool
}

// these classes implement the protocol already‚
extension MPSImageConversion : MetalPerformanceShaderKernel {}
extension MPSImageLanczosScale : MetalPerformanceShaderKernel {}
extension MPSImageHistogramEqualization : MetalPerformanceShaderKernel {}
extension MPSImageIntegral : MetalPerformanceShaderKernel {}
extension MPSImageIntegralOfSquares : MetalPerformanceShaderKernel {}

// class to store mpsshaders within the metalcomputepipeline
class MetalPerformanceShader : MetalShader {
    
    var encodeViaCommandBuffer: Bool = true
    let name : String
    let kernel : MetalPerformanceShaderKernel
    var source : Texture?
    var target : Texture?
    var inputTextures = Array<Texture>()
    
    // shaders can reference textures which are based on inputs during runtime  
    var hasSourceReference : Bool = false
    var hasTargetReference : Bool = false
    var sourceReference: Int = -1
    var targetReference: Int = -1
    
    init(name: String, kernel: MetalPerformanceShaderKernel) {
        self.name = name
        self.kernel = kernel
    }
    
    func setInputTextures(_ textures: [Texture]) {
        self.inputTextures = textures
    }
    
    func addSource(_ source: Texture) {
        self.source = source
        hasSourceReference = false
    }
    
    func addSourceReference(_ sourceReference: Int) {
        self.sourceReference = sourceReference
        hasSourceReference = true
    }
    
    func addTarget(_ target: Texture) {
        self.target = target
        hasTargetReference = false
    }
    
    func addTargetReference(_ targetReference: Int) {
        self.targetReference = targetReference
        self.hasTargetReference = true
    }
    
    func encode(_ commandBuffer: MTLCommandBuffer) {
        
        var _source : Texture? = nil
        var _target : Texture? = nil
        
        if self.hasSourceReference {
            _source = self.inputTextures[self.sourceReference]
        } else {
            _source = self.source
        }
        
        if self.hasTargetReference {
            _target = self.inputTextures[self.targetReference]
        } else {
            _target = self.target
        }
        
        guard let source = _source else {
            fatalError()
        }
        
        if let target = _target {
            self.kernel.encode(commandBuffer: commandBuffer, sourceTexture: source.texture, destinationTexture: target.texture)
        } else {
            let _ = self.kernel.encode(commandBuffer: commandBuffer, inPlaceTexture: &source.texture, fallbackCopyAllocator: nil)
        }
    }
    
    func encode(_ encoder: MTLComputeCommandEncoder, previous: MetalComputeShader?) {
        fatalError()
    }
    
    
}
