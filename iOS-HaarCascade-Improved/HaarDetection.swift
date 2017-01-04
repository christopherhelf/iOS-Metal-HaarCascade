//
//  HaarDetection.swift
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 07.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation
import Metal

class HaarDetection {
    
    let scaleFactor : Float = 1.2
    var minNeighbors : Int32 = 3
    let histogramEqualization : Bool = false
    let initialScale : Float = 0.5
    var step : Int32 = 2
    
    let minSize : Float = 360
    let maxSize : Float = 720
    
    var preprocessingData : PreprocessingData
    let preprocessingShaders : PreprocessingShaders
    
    let scaledWidth : Int
    let scaledHeight : Int
    let width : Int
    let height : Int
    
    var processingScaleData = [ProcessingDataPerScale]()
    var processingData : ProcessingData
    var processingMaskData : ProcessingMaskData! = nil
    var outputData : OutputData! = nil
    
    let cascade : HaarCascade
    
    var pixelParallelFromTexture : MTLComputePipelineState
    var haarCascadeResetCounters : MTLComputePipelineState
    var pixelParallelFromBuffer : MTLComputePipelineState
    var cascadeParallelFromBuffer : MTLComputePipelineState
    var haarCascadeSetCascadeCounter : MTLComputePipelineState
    var haarCascadeResetOutputCounter : MTLComputePipelineState
    var haarCascadeSetDrawDispatchArguments : MTLComputePipelineState
    var haarCascadeDraw : MTLComputePipelineState
    var haarCascadeBuildAdjacencyList : MTLComputePipelineState
    var haarCascadeDoGrouping : MTLComputePipelineState
    var haarCascadeGatherRectangles : MTLComputePipelineState
    
    var processingPipeline : MetalComputePipeline! = nil
    var preprocessingPipeline : MetalComputePipeline! = nil
    var drawingPipeline : MetalComputePipeline! = nil
    
    var maxNumRects = 1500
    var groupingParameters: GroupingParameters
    var groupingPipeline : MetalComputePipeline! = nil
    
    init(path: URL, width: Int, height: Int) {
        
        // get the data for the cascade
        guard let data = try? Data(contentsOf: path) else {
            fatalError()
        }
        
        // get the cascade
        cascade = try! HaarCascade(data: data)
        
        // get the dimensions and scaled dimensions
        self.width = width
        self.height = height
        scaledWidth = Int(Float(width) * initialScale)
        scaledHeight = Int(Float(height) * initialScale)
        
        // make textures and buffers for preprocessing
        preprocessingData = HaarDetection.initPreprocessingData(scaledWidth: scaledWidth, scaledHeight: scaledHeight, width: width, height: height)
        preprocessingShaders = HaarDetection.initPreprocessingShaders()
        
        // get the buffers from the cascade
        let buffers = cascade.getBuffersAndTexture()
        self.processingData = ProcessingData(stageBuffer: buffers.0, classifiersBuffer: buffers.1, rectsTexture: buffers.2)
        
        // retreive the pipelines
        self.pixelParallelFromTexture = Context.makeComputePipeline(name: "haarCascadePixelParallelFromTexture")
        self.haarCascadeResetCounters = Context.makeComputePipeline(name: "haarCascadeResetCounters")
        self.pixelParallelFromBuffer = Context.makeComputePipeline(name: "haarCascadePixelParallelFromBuffer")
        self.cascadeParallelFromBuffer = Context.makeComputePipeline(name: "haarCascadeCascadeParallelFromBuffer")
        self.haarCascadeSetCascadeCounter = Context.makeComputePipeline(name: "haarCascadeSetCascadeCounter")
        self.haarCascadeResetOutputCounter = Context.makeComputePipeline(name: "haarCascadeResetOutputCounter")
        self.haarCascadeSetDrawDispatchArguments = Context.makeComputePipeline(name: "haarCascadeSetDrawDispatchArguments")
        self.haarCascadeDraw = Context.makeComputePipeline(name: "haarCascadeDraw")
        self.haarCascadeBuildAdjacencyList = Context.makeComputePipeline(name: "haarCascadeBuildAdjacencyList")
        self.haarCascadeDoGrouping = Context.makeComputePipeline(name: "haarCascadeDoGrouping")
        self.haarCascadeGatherRectangles = Context.makeComputePipeline(name: "haarCascadeGatherRectangles")
        
        // grouping parameters
        groupingParameters = GroupingParameters(maxPoints: self.maxNumRects, initialScale: Float32(self.initialScale))
        
        // calculate the scales
        self.calculateScales()
        
        self.processingMaskData = ProcessingMaskData(outputMaskLength: self.width * self.height * self.processingScaleData.count, width: scaledWidth, height: scaledHeight)
        
        // final output data
        self.outputData = OutputData(scales: processingScaleData.count, width: scaledWidth, height: scaledHeight)
        
        // the final pipelines
        self.groupingPipeline = self.createGroupingPipeline()
        self.drawingPipeline = self.createDrawPipeline()
        self.preprocessingPipeline = self.createPreprocessPipeline()
        self.processingPipeline = self.createProcessingPipeline()
    }
    
    func calculateScales() {
        
        // the size we start with
        let startSize = minSize * initialScale
        var currentSize : Float = 0
        var currentFactor : Float = 0
        
        // original detection window
        let originalDetectionWindowWidth = Float(self.cascade.windowWidth)
        let originalDetectionWindowHeight = Float(self.cascade.windowHeight)
        assert(originalDetectionWindowWidth == originalDetectionWindowHeight)
        
        var gridWidth = 1
        var gridHeight = 1
    
        // iterate through all scales
        while currentSize <= maxSize * initialScale {
    
            // get the current scale
            let scale = pow(scaleFactor,currentFactor)
            
            // set the current size
            currentSize = startSize * scale
            
            // calculate the detection window size
            let detectionWindowScale = Float32(currentSize / originalDetectionWindowWidth)
            
            // get cascade buffer for this scale
            let cascadeBuffer = self.cascade.getScaledHaarCascadeBuffer(scale: Float32(detectionWindowScale))
            
            // calculate grid width and height
            gridWidth = Int(scaledWidth) - Int(currentSize)
            gridHeight = Int(scaledHeight) - Int(currentSize)
            
            // monitor also for the grid
            if gridWidth <= 0 || gridHeight <= 0 {
                break
            }
            
            gridWidth = (gridWidth+1)/Int(step)+1
            gridHeight = (gridHeight+1)/Int(step)+1
            
            // store that information
            processingScaleData.append(ProcessingDataPerScale(scale: detectionWindowScale, cascadeBuffer: cascadeBuffer, gridWidth: gridWidth, gridHeight: gridHeight))
            
            // increase the factor
            currentFactor = currentFactor + 1
            
        }
    
    }
    
    
    
}
