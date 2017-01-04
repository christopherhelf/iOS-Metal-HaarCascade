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
    let haarDetection : HaarDetection
    let id : Int
    
    init(view: View, id: Int) {
        self.id = id
        self.view = view
        
        let path = Bundle.main.path(forResource: "haarcascade_frontalface_default", ofType: "json")!
        let url = URL(fileURLWithPath: path)
        
        self.haarDetection = HaarDetection(path: url, width: 720, height: 1280)
    }
    
    func process(drawable: CAMetalDrawable, descriptor: MTLRenderPassDescriptor, texture: MTLTexture, time: Double, semaphore: DispatchSemaphore) {
        
        // create the commandbuffer
        let commandBuffer = Context.commandQueue().makeCommandBuffer()
        commandBuffer.label = "mainCommandBuffer \(self.id)"
        
        let tex = Texture(texture)
        
        // preprocess
        self.haarDetection.preprocess(texture: tex, commandBuffer: commandBuffer)
        
        // process
        self.haarDetection.process(commandBuffer: commandBuffer)
        
        // group rectangles
        self.haarDetection.groupRectangles(commandBuffer: commandBuffer)
        
        // draw
        self.haarDetection.draw(texture: tex, commandBuffer: commandBuffer)
        
        // present the video feed
        view.encode(drawable: drawable, descriptor: descriptor, commandBuffer: commandBuffer, input: texture)
        commandBuffer.present(drawable)
        
        // commit
        commandBuffer.addCompletedHandler { (_) in
            //print(self.haarDetection.groupingParameters.clusterCount.pointer[0])
            semaphore.signal()
        }
        commandBuffer.commit()
        
    }
    
    func setDimensions(width: Int, height: Int) {
        // not really dynamic yet, as resolution is fixed atm
    }
    
}
