//
//  MetalComputePipeline.swift
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 19.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation
import Metal
import MetalPerformanceShaders

class MetalComputePipeline {
    
    var shaders = Array<MetalShader>()
    var imagePrefetchList = Array<MPSImageDescriptor>()
    var inputTextures = Array<Texture>()
    
    let name : String
    
    init(name: String) {
        self.name = name
    }
    
    func addShader(_ shader: MetalShader) {
        shaders.append(shader)
    }
    
    func addImageToPrefetch(_ descriptor: MPSImageDescriptor) {
        self.imagePrefetchList.append(descriptor)
    }
    
    func setInputTextures(_ textures: [Texture]) {
        self.inputTextures = textures
    }
    
    func encode(_ commandBuffer: MTLCommandBuffer) {
        
        if shaders.count == 0 {
            return
        }
        
        if imagePrefetchList.count > 0 {
            MPSTemporaryImage.prefetchStorage(with: commandBuffer, imageDescriptorList: imagePrefetchList)
        }
        
        var encoderList = [MTLComputeCommandEncoder]()
        var hasEncoder = false
        
        for i in 0..<shaders.count {
            
            // add the inputs to the respective shaders
            shaders[i].setInputTextures(self.inputTextures)
            
            
            if shaders[i].encodeViaCommandBuffer {
                
                if hasEncoder {
                    encoderList.last!.popDebugGroup()
                    encoderList.last!.endEncoding()
                    hasEncoder = false
                }
                
                shaders[i].encode(commandBuffer)
                
            } else {
                
                if !hasEncoder {
                    encoderList.append(commandBuffer.makeComputeCommandEncoder())
                    encoderList.last!.label = name
                    encoderList.last!.pushDebugGroup(name)
                    hasEncoder = true
                }
                
                if i > 0, let shader = shaders[i-1] as? MetalComputeShader {
                    shaders[i].encode(encoderList.last!, previous: shader)
                } else {
                    shaders[i].encode(encoderList.last!, previous: nil)
                }
                
                if i == shaders.count - 1 {
                    encoderList.last!.popDebugGroup()
                    encoderList.last!.endEncoding()
                }
                
            }
        }
    }
}
