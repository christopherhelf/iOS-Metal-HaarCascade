//
//  HaarDetection+Grouping.swift
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 10.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation
import Metal
import MetalPerformanceShaders

extension HaarDetection {
    
    struct GroupingParameters {
        
        var adjacencyListCounter : BufferWithPointer<Int32>
        var dispatchArguments : Buffer<UInt32>
        
        var adjacencyList : Buffer<Int32>
        var neighborCount : Buffer<Int32>
        var offset: Buffer<Int32>
        
        var clusterCount : BufferWithPointer<Int32>
        
        var clusterBuffer : Buffer<Int32>
        var rectanglesBuffer : Buffer<Float32>
        
        var scaleBuffer : Buffer<Float32>
        var maxPts : Int
        
        let threads = Int(Context.sharedInstance.numThreadsAnchorsParallel)
        let block : MTLSize
        
        init(maxPoints: Int, initialScale: Float32) {
            
            self.maxPts = maxPoints
            self.block = MTLSizeMake(threads, 1, 1)
            
            adjacencyListCounter = MetalHelpers.createBufferWithPointer(0, "GroupingAdjacencyListCounter")
            dispatchArguments = MetalHelpers.createBuffer(4, "GroupingDispatchArguments")
            adjacencyList = MetalHelpers.createBuffer(maxPoints * maxPoints, "GroupingAdjacencyList")
            neighborCount = MetalHelpers.createBuffer(maxPoints, "GroupingNeighborCount")
            offset = MetalHelpers.createBuffer(maxPoints * maxPoints, "GroupingOffset")
            clusterCount = MetalHelpers.createBufferWithPointer(0, "GroupingClusterCount")
            clusterBuffer = MetalHelpers.createBuffer(maxPoints, "GroupingClusterBuffer")
            rectanglesBuffer = MetalHelpers.createBuffer(4 * maxPoints, "GroupingRectangles")
            scaleBuffer = MetalHelpers.createBuffer(initialScale, "GroupingInitialScale")
            
        }
        
        func reset() {
            self.adjacencyListCounter.pointer[0] = 0
            self.clusterCount.pointer[0] = 0
        }
    }
    
    
    func createGroupingPipeline() -> MetalComputePipeline {
        
        let nParallelThreads = Int(Context.sharedInstance.numThreadsAnchorsParallel)
        let nBytesInt32 = MemoryLayout<Int32>.size
        let nBytesFloat32 = MemoryLayout<Float32>.size
        let nMaxPts = self.groupingParameters.maxPts
        let nMinNeighbors = self.minNeighbors
        
        let pSetCounter = MetalComputeShader(name: "GroupingSetDispatchArgumentsFromCounter", pipeline: "haarCascadeSetCascadeCounter")
        pSetCounter.addBuffer(self.groupingParameters.dispatchArguments)
        pSetCounter.addBuffer(self.outputData.outputCounter)
        pSetCounter.setThreadsPerThreadGroup(1, 1, 1)
        pSetCounter.setThreadgroupsPerGrid(1, 1, 1)
        
        let pBuildAdjacency = MetalComputeShader(name: "GroupingBuildAdjacencyList", pipeline: "haarCascadeBuildAdjacencyList")
        pBuildAdjacency.addBuffer(self.outputData.output)
        pBuildAdjacency.addBuffer(self.outputData.outputCounter)
        pBuildAdjacency.addBuffer(self.groupingParameters.adjacencyListCounter)
        pBuildAdjacency.addBuffer(self.groupingParameters.adjacencyList)
        pBuildAdjacency.addBuffer(self.groupingParameters.neighborCount)
        pBuildAdjacency.addBuffer(self.groupingParameters.offset)
        pBuildAdjacency.addThreadgroupMemory(2 * nParallelThreads * nBytesInt32)
        pBuildAdjacency.addThreadgroupMemory(2 * nParallelThreads * nBytesInt32)
        pBuildAdjacency.addThreadgroupMemory(4 * nBytesInt32)
        pBuildAdjacency.addThreadgroupMemory(4 * nBytesInt32)
        pBuildAdjacency.addThreadgroupMemory(nMaxPts * nBytesInt32)
        pBuildAdjacency.setDispatchArgumentsBuffer(self.groupingParameters.dispatchArguments)
        pBuildAdjacency.setThreadsPerThreadGroup(nParallelThreads, 1, 1)
        
        let pBuildClusters = MetalComputeShader(name: "GroupingBuildClusters", pipeline: "haarCascadeDoGrouping")
        pBuildClusters.addBuffer(self.outputData.outputCounter)
        pBuildClusters.addBuffer(nMinNeighbors)
        pBuildClusters.addBuffer(self.groupingParameters.neighborCount)
        pBuildClusters.addBuffer(self.groupingParameters.offset)
        pBuildClusters.addBuffer(self.groupingParameters.adjacencyList)
        pBuildClusters.addBuffer(self.groupingParameters.clusterCount)
        pBuildClusters.addBuffer(self.groupingParameters.clusterBuffer)
        pBuildClusters.addThreadgroupMemory(4 * nBytesInt32)
        pBuildClusters.addThreadgroupMemory(4 * nBytesInt32)
        pBuildClusters.addThreadgroupMemory(nMaxPts * nBytesInt32)
        pBuildClusters.addThreadgroupMemory(4 * nBytesInt32)
        pBuildClusters.addThreadgroupMemory(nMaxPts * nBytesInt32)
        pBuildClusters.addThreadgroupMemory(4 * nBytesInt32)
        pBuildClusters.setThreadsPerThreadGroup(128, 1, 1)
        pBuildClusters.setThreadgroupsPerGrid(1, 1, 1)
        
        let pSetSecondCounter = MetalComputeShader(name: "GroupingSetDispatchArgumentsFromCluster", pipeline: "haarCascadeSetCascadeCounter")
        pSetSecondCounter.addBuffer(self.groupingParameters.dispatchArguments)
        pSetSecondCounter.addBuffer(self.groupingParameters.clusterCount)
        pSetSecondCounter.setThreadsPerThreadGroup(1, 1, 1)
        pSetSecondCounter.setThreadgroupsPerGrid(1, 1, 1)
        
        let pGatherRectangles = MetalComputeShader(name: "GroupingGatherRectangles", pipeline: "haarCascadeGatherRectangles")
        pGatherRectangles.addBuffer(self.outputData.outputCounter)
        pGatherRectangles.addBuffer(self.outputData.output)
        pGatherRectangles.addBuffer(self.groupingParameters.clusterBuffer)
        pGatherRectangles.addBuffer(self.groupingParameters.rectanglesBuffer)
        pGatherRectangles.addBuffer(self.groupingParameters.scaleBuffer)
        pGatherRectangles.addThreadgroupMemory(4 * nBytesFloat32 * nParallelThreads * 2)
        pGatherRectangles.addThreadgroupMemory(nBytesInt32 * 64 * 2)
        pGatherRectangles.setDispatchArgumentsBuffer(self.groupingParameters.dispatchArguments)
        pGatherRectangles.setThreadsPerThreadGroup(nParallelThreads, 1, 1)
        
        let groupingPipeline = MetalComputePipeline(name: "GroupingPipeline")
        groupingPipeline.addShader(pSetCounter)
        groupingPipeline.addShader(pBuildAdjacency)
        groupingPipeline.addShader(pBuildClusters)
        groupingPipeline.addShader(pSetSecondCounter)
        groupingPipeline.addShader(pGatherRectangles)
        
        return groupingPipeline
    }
    
    func groupRectangles(commandBuffer: MTLCommandBuffer) {
        self.groupingParameters.reset()
        self.groupingPipeline.encode(commandBuffer)
    }
    
    
}
