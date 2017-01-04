//
//  HaarDetection+Process.swift
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 07.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation
import Metal
import MetalPerformanceShaders

struct DetectionWindow {
    var x : UInt32
    var y : UInt32
    var width : UInt32
    var height : UInt32
}

extension DetectionWindow : ValidMetalBufferContent {
    static func getSizeInBytes() -> Int {
        return MemoryLayout<DetectionWindow>.size
    }
}

extension HaarDetection {
    
    struct ProcessingDataPerScale {
        var scale : Float32
        var cascadeBuffer : Buffer<HaarCascade.HaarCascadeDescriptor>
        var gridWidth : Int
        var gridHeight : Int
    }
    
    struct ProcessingData {
        var stageBuffer : Buffer<Float32>
        var classifiersBuffer : Texture
        var rectsTexture : Texture
    }
    
    struct ProcessingMaskData {
        
        var outputCounter : BufferWithPointer<Int32>
        var previousOutputCounter : Buffer<Int32>
        var dispatchArguments : Buffer<UInt32>
        
        var outputMask : Buffer<Int32>
        var varianceMap : Texture
        
        init(outputMaskLength: Int, width: Int, height: Int) {
            outputCounter = MetalHelpers.createBufferWithPointer(0, "outputCounter")
            outputMask = MetalHelpers.createBuffer(outputMaskLength, "outputMask")
            let varianceMapDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: width, height: height, mipmapped: false)
            varianceMap = Context.device().makeTexture(varianceMapDescriptor, name: "varianceMap")
            previousOutputCounter = MetalHelpers.createBuffer("previousOutputCounter")
            dispatchArguments = MetalHelpers.createBuffer(4, "MaskDispatchArguments")
        }
        
        func reset() {
            self.outputCounter.set(0)
        }
        
    }
    
    struct OutputData {
        
        var output : Buffer<DetectionWindow>
        var outputCounter : BufferWithPointer<Int32>
        
        init(scales: Int, width: Int, height: Int) {
            outputCounter = MetalHelpers.createBufferWithPointer(0, "finalOutputCounter")
            output = MetalHelpers.createBuffer(scales * width * height)
        }
        
        func reset() {
            self.outputCounter.set(0)
        }
        
    }
    
    func createProcessingPipeline() -> MetalComputePipeline {
        
        let nParallelThreads = Int(Context.sharedInstance.numThreadsAnchorsParallel)
        let nBytesInt32 = MemoryLayout<Int32>.size
        let nBytesFloat32 = MemoryLayout<Float32>.size
        
        let middle = getStageNumWithNotLessThanNclassifiers(nParallelThreads)
        let stops = [0, middle, 2 * middle]
        let pipeline = MetalComputePipeline(name: "HaarCascadeProcessing")
        
        for (index, scale) in self.processingScaleData.enumerated() {
         
            // 1. pixel parallel from texture
            
            let pPixelParallelFromTexture = MetalComputeShader(name: "PixelParallelFromTexture Scale \(index)", pipeline: self.pixelParallelFromTexture)
            pPixelParallelFromTexture.addTexture(self.processingData.classifiersBuffer)
            pPixelParallelFromTexture.addTexture(self.processingData.rectsTexture)
            pPixelParallelFromTexture.addTexture(self.preprocessingData.integralImage)
            pPixelParallelFromTexture.addTexture(self.preprocessingData.squareIntegralImage)
            pPixelParallelFromTexture.addTexture(self.processingMaskData.varianceMap)
            
            pPixelParallelFromTexture.addBuffer(scale.cascadeBuffer) //0
            pPixelParallelFromTexture.addBuffer(stops[0]) //1
            pPixelParallelFromTexture.addBuffer(stops[1]) //2
            pPixelParallelFromTexture.addBuffer(self.processingData.stageBuffer) //3
            pPixelParallelFromTexture.addBuffer(self.processingMaskData.outputCounter) //4
            pPixelParallelFromTexture.addBuffer(self.processingMaskData.outputMask) //5
            pPixelParallelFromTexture.addBuffer(self.step) //6
            
            pPixelParallelFromTexture.addThreadgroupMemory(nParallelThreads * nBytesInt32 * 2)
            pPixelParallelFromTexture.addThreadgroupMemory(nBytesInt32)
            pPixelParallelFromTexture.addThreadgroupMemory(nBytesInt32)
            
            pPixelParallelFromTexture.setThreadsPerThreadGroup(nParallelThreads, 1, 1)
            pPixelParallelFromTexture.setThreadgroupsPerGrid((scale.gridWidth + nParallelThreads - 1) / nParallelThreads, scale.gridHeight, 1)
            
            pipeline.addShader(pPixelParallelFromTexture)
            
            // 2. encode the next dispatch size and swap counters
            
            let pSwapCounters = MetalComputeShader(name: "SwapCounters Scale \(index)", pipeline: self.haarCascadeResetCounters)
            pSwapCounters.addBuffer(self.processingMaskData.dispatchArguments)
            pSwapCounters.addBuffer(self.processingMaskData.outputCounter)
            pSwapCounters.addBuffer(self.processingMaskData.previousOutputCounter)
            pSwapCounters.setThreadsPerThreadGroup(1, 1, 1)
            pSwapCounters.setThreadgroupsPerGrid(1, 1, 1)
            
            pipeline.addShader(pSwapCounters)
            
            // 3. encode another batch pixel-parallel but from the buffer
            
            let pPixelParallelFromBuffer = MetalComputeShader(name: "PixelParallelFromBuffer Scale \(index)", pipeline: self.pixelParallelFromBuffer)
            
            pPixelParallelFromBuffer.addTexture(self.processingData.classifiersBuffer)
            pPixelParallelFromBuffer.addTexture(self.processingData.rectsTexture)
            pPixelParallelFromBuffer.addTexture(self.preprocessingData.integralImage)
            pPixelParallelFromBuffer.addTexture(self.processingMaskData.varianceMap)
            
            pPixelParallelFromBuffer.addBuffer(scale.cascadeBuffer) //0
            pPixelParallelFromBuffer.addBuffer(stops[1]) //1
            pPixelParallelFromBuffer.addBuffer(stops[2]) //2
            pPixelParallelFromBuffer.addBuffer(self.processingData.stageBuffer) //3
            pPixelParallelFromBuffer.addBuffer(self.processingMaskData.previousOutputCounter) //4
            pPixelParallelFromBuffer.addBuffer(self.processingMaskData.outputCounter) //5
            pPixelParallelFromBuffer.addBuffer(self.processingMaskData.outputMask) //6
            
            pPixelParallelFromBuffer.addThreadgroupMemory(nParallelThreads * nBytesInt32 * 2)
            pPixelParallelFromBuffer.addThreadgroupMemory(nBytesInt32)
            pPixelParallelFromBuffer.addThreadgroupMemory(nBytesInt32)

            pPixelParallelFromBuffer.setThreadsPerThreadGroup(nParallelThreads, 1, 1)
            pPixelParallelFromBuffer.setDispatchArgumentsBuffer(self.processingMaskData.dispatchArguments)
            
            pipeline.addShader(pPixelParallelFromBuffer)
            
            // 4. encode the next dispatch size for all left stages
            
            let pSwapCoutersLeftStages = MetalComputeShader(name: "SwapCounters Scale \(index)", pipeline: self.haarCascadeSetCascadeCounter)
            pSwapCoutersLeftStages.addBuffer(self.processingMaskData.dispatchArguments)
            pSwapCoutersLeftStages.addBuffer(self.processingMaskData.outputCounter)
            
            pSwapCoutersLeftStages.setThreadsPerThreadGroup(1, 1, 1)
            pSwapCoutersLeftStages.setThreadgroupsPerGrid(1, 1, 1)
            
            pipeline.addShader(pSwapCoutersLeftStages)
            
            // 5. encode the remaining cascade-parallel
            
            let pCascadeParallel = MetalComputeShader(name: "CascadeParallel Scale \(index)", pipeline: self.cascadeParallelFromBuffer)
            
            pCascadeParallel.addTexture(self.processingData.classifiersBuffer)
            pCascadeParallel.addTexture(self.processingData.rectsTexture)
            pCascadeParallel.addTexture(self.preprocessingData.integralImage)
            pCascadeParallel.addTexture(self.processingMaskData.varianceMap)
            
            pCascadeParallel.addBuffer(scale.cascadeBuffer) // 0
            pCascadeParallel.addBuffer(stops[2]) // 1
            pCascadeParallel.addBuffer(self.cascade.stages.count) // 2
            pCascadeParallel.addBuffer(self.processingData.stageBuffer) // 3
            pCascadeParallel.addBuffer(self.processingMaskData.outputCounter) // 4
            pCascadeParallel.addBuffer(self.outputData.outputCounter.buffer) // 5
            pCascadeParallel.addBuffer(self.processingMaskData.outputMask) // 6
            pCascadeParallel.addBuffer(self.outputData.output.buffer) // 7
            
            pCascadeParallel.addThreadgroupMemory(nParallelThreads * 2 * nBytesFloat32)
        
            pCascadeParallel.setThreadsPerThreadGroup(nParallelThreads, 1, 1)
            pCascadeParallel.setDispatchArgumentsBuffer(self.processingMaskData.dispatchArguments)
            
            pipeline.addShader(pCascadeParallel)

            // 6. reset the counter
            
            let pResetCounter = MetalComputeShader(name: "ResetCounter Scale \(index)", pipeline: self.haarCascadeResetOutputCounter)
            pResetCounter.addBuffer(self.processingMaskData.outputCounter)
            pResetCounter.setThreadsPerThreadGroup(1, 1, 1)
            pResetCounter.setThreadgroupsPerGrid(1, 1, 1)
            
            pipeline.addShader(pResetCounter)
            
        }
        
        return pipeline
    }
    
    private func getStageNumWithNotLessThanNclassifiers(_ num: Int) -> Int {
        for i in 0..<self.cascade.stages.count {
            if self.cascade.stages[i].classifiers.count >= num {
                return i
            }
        }
        return 0
    }
    
    private func getStageSplits() -> [Int] {
        
        var middle = getStageNumWithNotLessThanNclassifiers(Int(Context.sharedInstance.numThreadsAnchorsParallel))
        var current = 0
        if middle == 0 { middle = 1 }
        let perStage = 7
        
        var stops = Array<Int>()
        
        while current < middle {
            stops.append(current)
            current += perStage
        }
        
        if (current > perStage && current - middle > perStage / 2)
        {
            stops[stops.count-1] =
                (middle - (current - 2 * perStage)) / 2
        }
        
        stops.append(middle)
        return stops
    }
    
    func process(commandBuffer: MTLCommandBuffer) {
        
        // reset the counters
        self.processingMaskData.reset()
        self.outputData.reset()
        
        // process
        self.processingPipeline.encode(commandBuffer)
        
        /*// three inputs
        let integralImage = preprocessingData.integralImage
        let squaredIntegralImage = preprocessingData.squareIntegralImage

        let threads = Int(Context.sharedInstance.numThreadsAnchorsParallel)
        let block = MTLSizeMake(threads, 1, 1)
        
        // reset the counter
        self.processingMaskData.reset()
        self.outputData.reset()
        
        // iterate through all scales
        for (index, scale) in self.processingScaleData.enumerated() {
            
            // process the first few stages pixel-parallel
            var startStage : UInt32 = 0
            var endStage : UInt32 = 7
            
            // the encoder
            let encoder = commandBuffer.makeComputeCommandEncoder()
            encoder.label = "HaarCascadeProcessing Scale \(index)"
            encoder.insertDebugSignpost("HaarCascadeProcessing Scale \(index)")
            
            encoder.setComputePipelineState(self.pixelParallelFromTexture)
            encoder.setTexture(self.processingData.classifiersBuffer.texture, at: 0)
            encoder.setTexture(self.processingData.rectsTexture.texture, at: 1)
            encoder.setTexture(integralImage.texture, at: 2)
            encoder.setTexture(squaredIntegralImage.texture, at: 3)
            encoder.setBytes(&startStage, length: MemoryLayout<Int32>.size, at: 1)
            encoder.setBytes(&endStage, length: MemoryLayout<Int32>.size, at: 2)
            encoder.setBuffer(self.processingData.stageBuffer.buffer, offset: 0, at: 3)
            encoder.setBuffer(self.processingMaskData.outputCounter.buffer, offset: 0, at: 4)
            encoder.setBuffer(self.processingMaskData.outputMask.buffer, offset: 0, at: 5)
            encoder.setBytes(&self.step, length: MemoryLayout<Int32>.size, at: 6)
            encoder.setTexture(self.processingMaskData.varianceMap.texture, at: 4)
            
            encoder.setThreadgroupMemoryLength(threads*2*MemoryLayout<Int32>.size, at: 0)
            encoder.setThreadgroupMemoryLength(MemoryLayout<Int32>.size * 4, at: 1)
            encoder.setThreadgroupMemoryLength(MemoryLayout<Int32>.size * 4, at: 2)

            // encode the first few stages pixel parallel by reading from the texture
            let haarCascade = scale.cascadeBuffer
            let width = (scale.gridWidth + threads - 1) / threads
            let height = scale.gridHeight

            encoder.setBuffer(haarCascade, offset: 0, at: 0)
            encoder.dispatchThreadgroups(MTLSizeMake(width, height, 1), threadsPerThreadgroup: block)
            
            // encode the next dispatch size and swap counters
            encoder.setComputePipelineState(self.haarCascadeResetCounters)
            encoder.setBuffer(self.processingMaskData.dispatchArguments.buffer, offset: 0, at: 0)
            encoder.setBuffer(self.processingMaskData.outputCounter.buffer, offset: 0, at: 1)
            encoder.setBuffer(self.processingMaskData.previousOutputCounter.buffer, offset: 0, at: 2)
            encoder.dispatchThreadgroups(MTLSizeMake(1, 1, 1), threadsPerThreadgroup: MTLSizeMake(1, 1, 1))
            
            // encode the next stages pixel parallel by reading from the buffer
            startStage = endStage
            endStage = startStage + 7
            
            encoder.setComputePipelineState(self.pixelParallelFromBuffer)
            encoder.setBuffer(haarCascade, offset: 0, at: 0)
            encoder.setBytes(&startStage, length: MemoryLayout<Int32>.size, at: 1)
            encoder.setBytes(&endStage, length: MemoryLayout<Int32>.size, at: 2)
            encoder.setBuffer(self.processingMaskData.previousOutputCounter.buffer, offset: 0, at: 4)
            encoder.setBuffer(self.processingMaskData.outputCounter.buffer, offset: 0, at: 5)
            encoder.setBuffer(self.processingMaskData.outputMask.buffer, offset: 0, at: 6)
            encoder.setTexture(self.processingMaskData.varianceMap.texture, at: 3)
            
            encoder.dispatchThreadgroups(indirectBuffer: self.processingMaskData.dispatchArguments.buffer, indirectBufferOffset: 0, threadsPerThreadgroup: block)
            
            // encode the next dispatch size and swap counters
            encoder.setComputePipelineState(self.haarCascadeResetCounters)
            encoder.setBuffer(self.processingMaskData.dispatchArguments.buffer, offset: 0, at: 0)
            encoder.setBuffer(self.processingMaskData.outputCounter.buffer, offset: 0, at: 1)
            encoder.setBuffer(self.processingMaskData.previousOutputCounter.buffer, offset: 0, at: 2)
            encoder.dispatchThreadgroups(MTLSizeMake(1, 1, 1), threadsPerThreadgroup: MTLSizeMake(1, 1, 1))
            
            // encode the next stages pixel parallel by reading from the buffer
            startStage = endStage
            endStage = startStage + 7
            
            encoder.setComputePipelineState(self.pixelParallelFromBuffer)
            encoder.setBuffer(haarCascade, offset: 0, at: 0)
            encoder.setBytes(&startStage, length: MemoryLayout<Int32>.size, at: 1)
            encoder.setBytes(&endStage, length: MemoryLayout<Int32>.size, at: 2)
            encoder.setBuffer(self.processingMaskData.previousOutputCounter.buffer, offset: 0, at: 4)
            encoder.setBuffer(self.processingMaskData.outputCounter.buffer, offset: 0, at: 5)
            encoder.setBuffer(self.processingMaskData.outputMask.buffer, offset: 0, at: 6)
            encoder.setTexture(self.processingMaskData.varianceMap.texture, at: 3)
            
            encoder.dispatchThreadgroups(indirectBuffer: self.processingMaskData.dispatchArguments.buffer, indirectBufferOffset: 0, threadsPerThreadgroup: block)
            
            // encode the next dispatch size for all left stages
            encoder.setComputePipelineState(self.haarCascadeSetCascadeCounter)
            encoder.setBuffer(self.processingMaskData.dispatchArguments.buffer, offset: 0, at: 0)
            encoder.setBuffer(self.processingMaskData.outputCounter.buffer, offset: 0, at: 1)
            encoder.dispatchThreadgroups(MTLSizeMake(1, 1, 1), threadsPerThreadgroup: MTLSizeMake(1, 1, 1))
            
            // encode the next stages pixel parallel by reading from the buffer
            startStage = endStage
            endStage = UInt32(self.cascade.stages.count)

            encoder.setComputePipelineState(self.cascadeParallelFromBuffer)
            encoder.setBuffer(haarCascade, offset: 0, at: 0)
            encoder.setBytes(&startStage, length: MemoryLayout<Int32>.size, at: 1)
            encoder.setBytes(&endStage, length: MemoryLayout<Int32>.size, at: 2)
            encoder.setBuffer(self.processingMaskData.outputCounter.buffer, offset: 0, at: 4)
            encoder.setBuffer(self.outputData.outputCounter.buffer, offset: 0, at: 5)
            encoder.setBuffer(self.processingMaskData.outputMask.buffer, offset: 0, at: 6)
            encoder.setBuffer(self.outputData.output.buffer, offset: 0, at: 7)
            encoder.setThreadgroupMemoryLength(threads*2*MemoryLayout<Float32>.size, at: 0)
            
            encoder.dispatchThreadgroups(indirectBuffer: self.processingMaskData.dispatchArguments.buffer, indirectBufferOffset: 0, threadsPerThreadgroup: block)
            
            // reset the counter
            encoder.setComputePipelineState(self.haarCascadeResetOutputCounter)
            encoder.setBuffer(self.processingMaskData.outputCounter.buffer, offset: 0, at: 0)
            encoder.dispatchThreadgroups(MTLSizeMake(1, 1, 1), threadsPerThreadgroup: MTLSizeMake(1, 1, 1))
            
            encoder.endEncoding()
        }*/
        
        
    }
    
    
    
    
}
