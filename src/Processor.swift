//
//  Processor.swift
//
//  Created by Christopher Helf on 15.11.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation
import MetalKit

class Processor {
    
    let view: View
    let faceDetection : FaceDetection
    let id : Int
    let haarDrawPipeline : MTLComputePipelineState
    
    init(view: View, id: Int) {
        self.id = id
        self.view = view
        let url = Bundle.main.url(forResource: "haarcascade_frontalface_default", withExtension: "json", subdirectory: nil, localization: nil)
        self.faceDetection = FaceDetection(path: url!, width: 720, height: 1280)
        let function = Context.library().makeFunction(name: "haarDrawRects")!
        function.label = "haarDrawRects"
        haarDrawPipeline = try! Context.device().makeComputePipelineState(function: function)
    }
    
    func process(drawable: CAMetalDrawable, descriptor: MTLRenderPassDescriptor, texture: MTLTexture, time: Double, semaphore: DispatchSemaphore) {
        
        // create the commandbuffer
        let commandBuffer = Context.commandQueue().makeCommandBuffer()
        commandBuffer.label = "mainCommandBuffer \(self.id)"
        
        // get the rectangles
        faceDetection.getFaces(commandBuffer: commandBuffer, input: texture, output: texture)
        
        // commit
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // get the faces
        let faces = faceDetection.groupRectangles(eps: 0.4).map { (r) -> HaarCascade.sRect in
            return r.getHaarRect()
        }
        
        // second commandbuffer
        // if grouping would be done on the gpu, we could use the same commandbuffer using
        // https://developer.apple.com/reference/metal/mtlcomputecommandencoder/1443157-dispatchthreadgroups
        // an indirect buffer as dispatch input
        let commandBuffer2 = Context.commandQueue().makeCommandBuffer()
        commandBuffer2.label = "drawCommandBuffer \(self.id)"
        
        // only necessary if we found a face
        if faces.count > 0 && faces.count < Context.device().maxThreadsPerThreadgroup.width {

            // create the buffer
            let faceBuffer = Context.device().makeBuffer(bytes: faces, length: faces.count * MemoryLayout<HaarCascade.sRect>.size, options: MTLResourceOptions.cpuCacheModeWriteCombined)
        
            // encode
            let encoder = commandBuffer2.makeComputeCommandEncoder()
            encoder.setComputePipelineState(haarDrawPipeline)
            encoder.setTexture(texture, at: 0)
            encoder.setBuffer(faceBuffer, offset: 0, at: 0)
            encoder.dispatchThreadgroups(MTLSizeMake(faces.count, 1, 1), threadsPerThreadgroup: MTLSizeMake(1, 1, 1))
            encoder.endEncoding()
        }
        
        // present the video feed
        view.encode(drawable: drawable, descriptor: descriptor, commandBuffer: commandBuffer2, input: texture)
        commandBuffer2.present(drawable)
        
        // signal the semaphore
        commandBuffer2.addCompletedHandler { (_) in
            semaphore.signal()
        }
        
        // done
        commandBuffer2.commit()
        
    }
    
    func setDimensions(width: Int, height: Int) {
        
    }
    
}
