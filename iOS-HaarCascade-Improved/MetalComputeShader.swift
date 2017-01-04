//
//  MetalComputeShaderHelper.swift
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 18.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation
import Metal

// A helper class encoding information about buffers, textures and threadgroups
// that will go into a computecommandencoder, this can be hooked up in advance
class MetalComputeShader : MetalShader {
    
    var encodeViaCommandBuffer: Bool = false
    var threadsPerThreadGroup = MTLSize(width: Int(Context.sharedInstance.numThreadsAnchorsParallel), height: 1, depth: 1)
    var threadGroupsPerGrid = MTLSize(width: 1, height: 1, depth: 1)
    
    let name : String
    let pipeline : MTLComputePipelineState
    var buffers : [ComputeCommandEncoderArgument]
    var textures : [ComputeCommandEncoderArgument]
    var threadGroups : [Int]
    var inputTextures = Array<Texture>()
    
    var dispatchArguments : Buffer<UInt32>?
    
    init(name: String, pipeline: String) {
        self.pipeline = Context.makeComputePipeline(name: pipeline)
        self.buffers = Array<ComputeCommandEncoderArgument>()
        self.textures = Array<ComputeCommandEncoderArgument>()
        self.threadGroups = [Int]()
        self.name = name
    }
    
    init(name: String, pipeline: MTLComputePipelineState) {
        self.name = name
        self.pipeline = pipeline
        self.buffers = Array<ComputeCommandEncoderArgument>()
        self.textures = Array<ComputeCommandEncoderArgument>()
        self.threadGroups = [Int]()
    }
    
    func setInputTextures(_ textures: [Texture]) {
        self.inputTextures = textures
    }
    
    func setThreadsPerThreadGroup(_ width: Int, _ height: Int, _ depth: Int) {
        self.threadsPerThreadGroup = MTLSize(width: width, height: height, depth: depth)
    }
    
    func setThreadgroupsPerGrid(_ width: Int, _ height: Int, _ depth: Int) {
        self.threadGroupsPerGrid = MTLSize(width: width, height: height, depth: depth)
    }
    
    func setDispatchArgumentsBuffer(_ buffer: Buffer<UInt32>) {
        self.dispatchArguments  = buffer
    }
    
    func addBuffer(_ buffer: ComputeCommandEncoderArgument) {
        self.buffers.append(buffer)
    }
    
    func addBuffer(_ buffer: MTLBuffer) {
        let _buffer = Buffer<Int32>(buffer: buffer)
        self.buffers.append(_buffer)
    }
    
    func addTexture(_ tex: MTLTexture) {
        self.textures.append(Texture(tex))
    }
    
    func addTextureAtInput(_ index: Int) {
        self.textures.append(TextureInputReference(index))
    }
    
    func addTexture(_ tex: ComputeCommandEncoderArgument) {
        self.textures.append(tex)
    }
    
    func addThreadgroupMemory(_ length: Int) {
        // make sure the memory is 16 bytes aligned
        if length % 16 == 0 {
            self.threadGroups.append(length)
        } else {
            let length = (16 - length % 16) + length
            self.threadGroups.append(length)
        }
    }
    
    func addBuffers(_ buffers: [ComputeCommandEncoderArgument]) {
        for b in buffers {
            self.addBuffer(b)
        }
    }
    
    func addBuffers(_ buffers: [MTLBuffer]) {
        for b in buffers {
            self.addBuffer(b)
        }
    }
    
    func addTextures(_ tex: [MTLTexture]) {
        for t in tex {
            self.addTexture(t)
        }
    }
    
    func addTextures(_ tex: [ComputeCommandEncoderArgument]) {
        for t in tex {
            self.addTexture(t)
        }
    }
    
    func addThreadGroupMemories(_ lengths: [Int]) {
        for i in lengths {
            self.addThreadgroupMemory(i)
        }
    }
    
    func isSameBuffer(_ buffer: ComputeCommandEncoderArgument, at: Int) -> Bool {
        
        guard self.buffers.count > at else {
            return false
        }
        
        return self.buffers[at].uuid == buffer.uuid
    }
    
    func isSameThreadgroupLength(_ length: Int, at: Int) -> Bool {
        
        guard self.threadGroups.count > at else {
            return false
        }
        
        return self.threadGroups[at] == length
    }
    
    func encode(_ encoder: MTLComputeCommandEncoder, previous: MetalComputeShader?) {
        
        let hasPrevious = previous != nil
        
        encoder.pushDebugGroup(name)
        encoder.setComputePipelineState(self.pipeline)
        
        for i in 0..<textures.count {
            textures[i].set(encoder, at: i, inputs: self.inputTextures)
        }
        
        for i in 0..<buffers.count {
            if hasPrevious {
                if !previous!.isSameBuffer(buffers[i], at: i) {
                    buffers[i].set(encoder, at: i, inputs: self.inputTextures)
                }
            } else {
                buffers[i].set(encoder, at: i, inputs: self.inputTextures)
            }
        }
        
        for i in 0..<threadGroups.count {
            if hasPrevious {
                if !previous!.isSameThreadgroupLength(threadGroups[i], at: i) {
                    encoder.setThreadgroupMemoryLength(threadGroups[i], at: i)
                }
            } else {
                encoder.setThreadgroupMemoryLength(threadGroups[i], at: i)
            }
        }
        
        if let dispatchArguments = self.dispatchArguments {
            encoder.dispatchThreadgroups(indirectBuffer: dispatchArguments.buffer, indirectBufferOffset: 0, threadsPerThreadgroup: self.threadsPerThreadGroup)
        } else {
            encoder.dispatchThreadgroups(threadGroupsPerGrid, threadsPerThreadgroup: self.threadsPerThreadGroup)
        }
        
        encoder.popDebugGroup()
    }
    
    func encode(_ commandBuffer: MTLCommandBuffer) {
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.label = self.name
        self.encode(encoder, previous: nil)
        encoder.endEncoding()
    }
    
}
