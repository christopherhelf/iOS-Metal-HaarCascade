//
//  HaarDetection+Draw.swift
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 10.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation
import Metal

extension HaarDetection {
    
    func createDrawPipeline() -> MetalComputePipeline {
        
        let pSetArguments = MetalComputeShader(name: "DrawingSetDrawDispatchArguments", pipeline: "haarCascadeSetDrawDispatchArguments")
        pSetArguments.addBuffer(self.processingMaskData.dispatchArguments)
        pSetArguments.addBuffer(self.groupingParameters.clusterCount.buffer)
        pSetArguments.setThreadsPerThreadGroup(1, 1, 1)
        pSetArguments.setThreadgroupsPerGrid(1, 1, 1)
        
        let threads = Int(Context.sharedInstance.numThreadsAnchorsParallel)
        
        let pDraw = MetalComputeShader(name: "DrawingDraw", pipeline: "haarCascadeDraw")
        pDraw.addBuffer(self.groupingParameters.rectanglesBuffer.buffer)
        pDraw.addBuffer(Float32(1.0))
        pDraw.addBuffer(self.groupingParameters.clusterCount.buffer)
        pDraw.addTextureAtInput(0) // this basically references a texture thats being inputed into the pipeline
        pDraw.setDispatchArgumentsBuffer(self.processingMaskData.dispatchArguments)
        pDraw.setThreadsPerThreadGroup(threads, 1, 1)
        
        let drawPipeline = MetalComputePipeline(name: "DrawingPipeline")
        drawPipeline.addShader(pSetArguments)
        drawPipeline.addShader(pDraw)
        
        return drawPipeline
    }

    func draw(texture: Texture, commandBuffer: MTLCommandBuffer) {
        self.drawingPipeline.setInputTextures([texture])
        self.drawingPipeline.encode(commandBuffer)
    }
    
    
}
