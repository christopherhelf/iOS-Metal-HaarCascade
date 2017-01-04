//
//  iOS_HaarCascade_ImprovedTests.swift
//  iOS-HaarCascade-ImprovedTests
//
//  Created by Christopher Helf on 05.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import XCTest
import Metal

@testable import iOS_HaarCascade_Improved

class iOS_HaarCascade_ImprovedTests: XCTestCase {
    
    var context = Context.sharedInstance
    var pipeline1 : MTLComputePipelineState! = nil
    var pipeline2 : MTLComputePipelineState! = nil
    var pipeline3 : MTLComputePipelineState! = nil
    var pipeline4 : MTLComputePipelineState! = nil
    var pipeline5 : MTLComputePipelineState! = nil
    var pipeline6 : MTLComputePipelineState! = nil
    var pipeline8 : MTLComputePipelineState! = nil
    var pipeline9 : MTLComputePipelineState! = nil

    var detection : HaarDetection! = nil

    override func setUp() {
        super.setUp()
        context = Context.sharedInstance
        pipeline1 = Context.makeComputePipeline(name: "streamCompactionTest")
        pipeline2 = Context.makeComputePipeline(name: "subReduceAddTest")
        pipeline3 = Context.makeComputePipeline(name: "testHaarFeatureRect")
        pipeline4 = Context.makeComputePipeline(name: "testHaarClassifier")
        pipeline5 = Context.makeComputePipeline(name: "testHaarStage")
        pipeline6 = Context.makeComputePipeline(name: "haarCascadeBuildAdjacencyList")
        pipeline8 = Context.makeComputePipeline(name: "haarCascadeDoGrouping")
        pipeline9 = Context.makeComputePipeline(name: "haarCascadeGatherRectangles")

        let path = Bundle.main.path(forResource: "haarcascade_frontalface_default", ofType: "json")!
        let url = URL(fileURLWithPath: path)
        
        detection = HaarDetection(path: url, width: 720, height: 1280)
    }
    
    override func tearDown() {
        super.tearDown()
    }
    
    func testStreamCompaction() {
        
        var input = [Int32]()
        
        for _ in 0..<1280*3 {
            input.append(Float(arc4random()) / Float(UINT32_MAX) > 0.7 ? 1 : 0)
        }
        
        var count : Int32 = Int32(input.count)
        
        var indices = [Int]()
        for (index, i) in input.enumerated() {
            if i == 1 {
                indices.append(index)
            }
        }
        
        var output = [Int32](repeating: 0, count: input.count)
        
        let outputBuffer = Context.device().makeBuffer(bytes: &output, length: input.count * MemoryLayout<Int32>.size, options: .storageModeShared)
        let inputBuffer = Context.device().makeBuffer(bytes: &input, length: input.count * MemoryLayout<Int32>.size, options: .storageModeShared)
        
        var outMask : Int32 = 0
        
        let outMaskBuffer = Context.device().makeBuffer(bytes: &outMask, length: MemoryLayout<Int32>.size, options: .storageModeShared)
        
        assert(Int(Context.sharedInstance.kWarpSize) == pipeline1.threadExecutionWidth)
        
        let commandBuffer = Context.commandQueue().makeCommandBuffer()
        let encoder = commandBuffer.makeComputeCommandEncoder()
        let blocks = input.count/Int(Context.sharedInstance.numThreadsAnchorsParallel)
        let threads = Int(Context.sharedInstance.numThreadsAnchorsParallel)
        
        encoder.setComputePipelineState(pipeline1)
        encoder.setBuffer(inputBuffer, offset: 0, at: 0)
        encoder.setBuffer(outputBuffer, offset: 0, at: 1)
        encoder.setBuffer(outMaskBuffer, offset: 0, at: 2)
        encoder.setBytes(&count, length: MemoryLayout<Int32>.size, at: 3)
        encoder.setThreadgroupMemoryLength(threads*2*MemoryLayout<Int32>.size, at: 0)
        encoder.setThreadgroupMemoryLength(MemoryLayout<Int32>.size, at: 1)
        encoder.setThreadgroupMemoryLength(MemoryLayout<Int32>.size, at: 2)
        encoder.dispatchThreadgroups(MTLSizeMake(blocks, 1, 1), threadsPerThreadgroup: MTLSizeMake(threads, 1, 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let outmask : Int = Int(outMaskBuffer.contents().assumingMemoryBound(to: Int32.self).pointee)
        let ptr = outputBuffer.contents().assumingMemoryBound(to: Int32.self)
        
        XCTAssertTrue(outmask > 0)
        XCTAssertTrue(outmask == indices.count)
        
        for i in 0..<indices.count {
            let index = indices[i]
            var found = false
            for j in 0..<outmask {
                if ptr[j] == Int32(index) {
                    found = true
                    break
                }
            }
            XCTAssertTrue(found)
        }
    
    }
    
    func testSubReduce() {
        
        var input = [Float32]()
        var actualSum : Float = 0.0
        let useIncrease : Bool = false
        
        for i in 0..<64 {
            let rnd = useIncrease ? Float(i) : Float(arc4random()) / Float(UINT32_MAX)
            input.append(Float32(rnd))
            actualSum = actualSum + Float(rnd)
        }
        
        var count : Int32 = Int32(input.count)
        
        let outputBuffer = Context.device().makeBuffer(length: MemoryLayout<Float32>.size, options: .storageModeShared)
        let inputBuffer = Context.device().makeBuffer(bytes: &input, length: input.count * MemoryLayout<Float32>.size, options: .storageModeShared)

        assert(Int(Context.sharedInstance.kWarpSize) == pipeline1.threadExecutionWidth)
        
        let commandBuffer = Context.commandQueue().makeCommandBuffer()
        let encoder = commandBuffer.makeComputeCommandEncoder()
        let blocks = input.count/Int(Context.sharedInstance.numThreadsAnchorsParallel)
        let threads = Int(Context.sharedInstance.numThreadsAnchorsParallel)
        
        encoder.setComputePipelineState(pipeline2)
        encoder.setBuffer(inputBuffer, offset: 0, at: 0)
        encoder.setBuffer(outputBuffer, offset: 0, at: 1)
        encoder.setBytes(&count, length: MemoryLayout<Int32>.size, at: 2)
        encoder.setThreadgroupMemoryLength(threads*2*MemoryLayout<Float32>.size, at: 0)
        encoder.dispatchThreadgroups(MTLSizeMake(blocks, 1, 1), threadsPerThreadgroup: MTLSizeMake(threads, 1, 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let sum : Float32 = outputBuffer.contents().assumingMemoryBound(to: Float32.self).pointee
        
        XCTAssertEqualWithAccuracy(sum, Float32(actualSum), accuracy: Float32(1e-4))
        
    }
    
    
    func testHaarFeatureRects() {
        
        func getRect(x: UInt8, y: UInt8, w: UInt8, h: UInt8, weight: Float32) -> [UInt32] {
            var weight = weight
            let u1 = UInt32(x) | UInt32(y) << 8 | UInt32(w) << 16 | UInt32(h) << 24
            var u2 = UInt32(0)
            memcpy(&u2, &weight, 4)
            return [u1, u2]
        }
        
        var rects = getRect(x: 7, y: 6, w: 18, h: 17, weight: 1.05)
        rects = rects + getRect(x: 9, y: 3, w: 11, h: 19, weight: 3.07)
        
        let rectTexBuf = Context.device().makeBuffer(bytes: &rects, length: rects.count * MemoryLayout<UInt32>.size, options: .storageModeShared)
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Uint, width: 2, height: 1, mipmapped: false)
        let rectTex = rectTexBuf.makeTexture(descriptor: desc, offset: 0, bytesPerRow: 2 * 2 * MemoryLayout<UInt32>.size)
        
        let xBuf = Context.device().makeBuffer(length: 2 * MemoryLayout<UInt32>.size, options: .storageModeShared)
        let yBuf = Context.device().makeBuffer(length: 2 * MemoryLayout<UInt32>.size, options: .storageModeShared)
        let wBuf = Context.device().makeBuffer(length: 2 * MemoryLayout<UInt32>.size, options: .storageModeShared)
        let hBuf = Context.device().makeBuffer(length: 2 * MemoryLayout<UInt32>.size, options: .storageModeShared)
        let weightBuf = Context.device().makeBuffer(length: 2 * MemoryLayout<Float32>.size, options: .storageModeShared)
        
        let commandBuffer = Context.commandQueue().makeCommandBuffer()
        let encoder = commandBuffer.makeComputeCommandEncoder()
        let blocks = 1
        let threads = 2
        
        encoder.setComputePipelineState(pipeline3)
        encoder.setTexture(rectTex, at: 0)
        encoder.setBuffer(xBuf, offset: 0, at: 0)
        encoder.setBuffer(yBuf, offset: 0, at: 1)
        encoder.setBuffer(wBuf, offset: 0, at: 2)
        encoder.setBuffer(hBuf, offset: 0, at: 3)
        encoder.setBuffer(weightBuf, offset: 0, at: 4)
        encoder.dispatchThreadgroups(MTLSizeMake(blocks, 1, 1), threadsPerThreadgroup: MTLSizeMake(threads, 1, 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let xptr = xBuf.contents().assumingMemoryBound(to: UInt32.self)
        let yptr = yBuf.contents().assumingMemoryBound(to: UInt32.self)
        let wptr = wBuf.contents().assumingMemoryBound(to: UInt32.self)
        let hptr = hBuf.contents().assumingMemoryBound(to: UInt32.self)
        let weightptr = weightBuf.contents().assumingMemoryBound(to: Float32.self)
        
        XCTAssertTrue(xptr[0] == UInt32(7))
        XCTAssertTrue(yptr[0] == UInt32(6))
        XCTAssertTrue(wptr[0] == UInt32(18))
        XCTAssertTrue(hptr[0] == UInt32(17))
        XCTAssertTrue(weightptr[0] == Float32(1.05))
        
        XCTAssertTrue(xptr[1] == UInt32(9))
        XCTAssertTrue(yptr[1] == UInt32(3))
        XCTAssertTrue(wptr[1] == UInt32(11))
        XCTAssertTrue(hptr[1] == UInt32(19))
        XCTAssertTrue(weightptr[1] == Float32(3.07))
        
    }
    
    func testHaarClassifier() {
        
        func getClassifier(numRects: UInt8, firstRectOffset: UInt32, threshold: Float32, left: Float32, right: Float32) -> [Float32] {
            
            var a = UInt32(firstRectOffset) | UInt32(numRects) << 24;
            var b : Float32 = 0
            memcpy(&b, &a, 4)
            
            return [b,threshold,left,right]
        }
        
        var classifiers = getClassifier(numRects: 3, firstRectOffset: 1280, threshold: 0.12345, left: 5.789, right: 0.101112)
        classifiers = classifiers + getClassifier(numRects: 7, firstRectOffset: 720, threshold: 3.14, left: 0.1927, right: 4.2314)
        
        
        let rectTexBuf = Context.device().makeBuffer(bytes: &classifiers, length: classifiers.count * MemoryLayout<Float32>.size, options: .storageModeShared)
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: 2, height: 1, mipmapped: false)
        let rectTex = rectTexBuf.makeTexture(descriptor: desc, offset: 0, bytesPerRow: 2 * 4 * MemoryLayout<Float32>.size)
        
        let numRectsBuf = Context.device().makeBuffer(length: 2 * MemoryLayout<UInt32>.size, options: .storageModeShared)
        let firstRectOffsetBuf = Context.device().makeBuffer(length: 2 * MemoryLayout<UInt32>.size, options: .storageModeShared)
        let thresholdBuf = Context.device().makeBuffer(length: 2 * MemoryLayout<Float32>.size, options: .storageModeShared)
        let leftBuf = Context.device().makeBuffer(length: 2 * MemoryLayout<Float32>.size, options: .storageModeShared)
        let rightBuf = Context.device().makeBuffer(length: 2 * MemoryLayout<Float32>.size, options: .storageModeShared)
        
       
        let commandBuffer = Context.commandQueue().makeCommandBuffer()
        let encoder = commandBuffer.makeComputeCommandEncoder()
        let blocks = 1
        let threads = 2
        
        encoder.setComputePipelineState(pipeline4)
        encoder.setTexture(rectTex, at: 0)
        encoder.setBuffer(numRectsBuf, offset: 0, at: 0)
        encoder.setBuffer(firstRectOffsetBuf, offset: 0, at: 1)
        encoder.setBuffer(thresholdBuf, offset: 0, at: 2)
        encoder.setBuffer(leftBuf, offset: 0, at: 3)
        encoder.setBuffer(rightBuf, offset: 0, at: 4)
        encoder.dispatchThreadgroups(MTLSizeMake(blocks, 1, 1), threadsPerThreadgroup: MTLSizeMake(threads, 1, 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let aptr = numRectsBuf.contents().assumingMemoryBound(to: UInt32.self)
        let bptr = firstRectOffsetBuf.contents().assumingMemoryBound(to: UInt32.self)
        let cptr = thresholdBuf.contents().assumingMemoryBound(to: Float32.self)
        let dptr = leftBuf.contents().assumingMemoryBound(to: Float32.self)
        let eptr = rightBuf.contents().assumingMemoryBound(to: Float32.self)

        XCTAssertTrue(aptr[0] == UInt32(3))
        XCTAssertTrue(bptr[0] == UInt32(1280))
        XCTAssertTrue(cptr[0] == Float32(0.12345))
        XCTAssertTrue(dptr[0] == Float32(5.789))
        XCTAssertTrue(eptr[0] == Float32(0.101112))
        
        XCTAssertTrue(aptr[1] == UInt32(7))
        XCTAssertTrue(bptr[1] == UInt32(720))
        XCTAssertTrue(cptr[1] == Float32(3.14))
        XCTAssertTrue(dptr[1] == Float32(0.1927))
        XCTAssertTrue(eptr[1] == Float32(4.2314))
        
        
    }
    
    func testHaarStage() {
        
        func getStage(offset: UInt32, numberOfTrees: UInt32, threshold: Float32) -> [Float32] {
            
            var a = numberOfTrees | offset << 16
            var b : Float32 = 0
            memcpy(&b, &a, 4)
            
            return [b,threshold]
        }
        
        var stages = getStage(offset: 1923, numberOfTrees: 728, threshold: 0.5689)
        stages = stages + getStage(offset: 9334, numberOfTrees: 99, threshold: 7.5927)
        
        let stageBuffer = Context.device().makeBuffer(bytes: &stages, length: stages.count * MemoryLayout<Float32>.size, options: .storageModeShared)
        
        let aBuf = Context.device().makeBuffer(length: 2 * MemoryLayout<UInt32>.size, options: .storageModeShared)
        let bBuf = Context.device().makeBuffer(length: 2 * MemoryLayout<UInt32>.size, options: .storageModeShared)
        let cBuf = Context.device().makeBuffer(length: 2 * MemoryLayout<Float32>.size, options: .storageModeShared)

        let commandBuffer = Context.commandQueue().makeCommandBuffer()
        let encoder = commandBuffer.makeComputeCommandEncoder()
        let blocks = 1
        let threads = 2
        
        encoder.setComputePipelineState(pipeline5)
        encoder.setBuffer(stageBuffer, offset: 0, at: 0)
        encoder.setBuffer(aBuf, offset: 0, at: 1)
        encoder.setBuffer(bBuf, offset: 0, at: 2)
        encoder.setBuffer(cBuf, offset: 0, at: 3)
        encoder.dispatchThreadgroups(MTLSizeMake(blocks, 1, 1), threadsPerThreadgroup: MTLSizeMake(threads, 1, 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let aptr = aBuf.contents().assumingMemoryBound(to: UInt32.self)
        let bptr = bBuf.contents().assumingMemoryBound(to: UInt32.self)
        let cptr = cBuf.contents().assumingMemoryBound(to: Float32.self)
        
        XCTAssertTrue(aptr[0] == UInt32(1923))
        XCTAssertTrue(bptr[0] == UInt32(728))
        XCTAssertTrue(cptr[0] == Float32(0.5689))
        
        XCTAssertTrue(aptr[1] == UInt32(9334))
        XCTAssertTrue(bptr[1] == UInt32(99))
        XCTAssertTrue(cptr[1] == Float32(7.5927))
        
    }
    
    func testAdjacencyList() {
        
        let a = DetectionWindow(x: 10, y: 10, width: 10, height: 10)
        let b = DetectionWindow(x: 40, y: 40, width: 10, height: 10)
        let c = DetectionWindow(x: 60, y: 60, width: 10, height: 10)
        let d = DetectionWindow(x: 80, y: 80, width: 10, height: 10)
        let e = DetectionWindow(x: 100, y: 100, width: 10, height: 10)
        
        let an = 67
        let bn = 7
        let cn = 10
        let dn = 347
        let en = 1
        
        var array = [a]
        
        for _ in 0..<an - 1 { array.append(a) }
        for _ in 0..<bn { array.append(b) }
        for _ in 0..<cn { array.append(c) }
        for _ in 0..<dn { array.append(d) }
        for _ in 0..<en { array.append(e) }
        
        array = array.shuffled()
        
        let buffer = Context.device().makeBuffer(bytes: &array, length: array.count * MemoryLayout<DetectionWindow>.size, options: .storageModeShared)
        var count : Int32 = Int32(array.count)

        // the encoder
        let commandBuffer = Context.commandQueue().makeCommandBuffer()
        let encoder = commandBuffer.makeComputeCommandEncoder()

        // calculate the adjacency list
        encoder.setComputePipelineState(self.pipeline6)
        encoder.setBuffer(buffer, offset: 0, at: 0)
        encoder.setBytes(&count, length: 4, at: 1)
        encoder.setBuffer(detection.groupingParameters.adjacencyListCounter.buffer, offset: 0, at: 2)
        encoder.setBuffer(detection.groupingParameters.adjacencyList.buffer, offset: 0, at: 3)
        encoder.setBuffer(detection.groupingParameters.neighborCount.buffer, offset: 0, at: 4)
        encoder.setBuffer(detection.groupingParameters.offset.buffer, offset: 0, at: 5)
        encoder.setThreadgroupMemoryLength(64*2*MemoryLayout<Int32>.size, at: 0)
        encoder.setThreadgroupMemoryLength(64*2*MemoryLayout<Int32>.size, at: 1)
        encoder.setThreadgroupMemoryLength(MemoryLayout<Int32>.size * 4, at: 2)
        encoder.setThreadgroupMemoryLength(MemoryLayout<Int32>.size * 4, at: 3)
        encoder.setThreadgroupMemoryLength(3500*MemoryLayout<Int32>.size * 4, at: 4)
        encoder.dispatchThreadgroups(MTLSizeMake(array.count, 1, 1), threadsPerThreadgroup: MTLSizeMake(64, 1, 1))
        
        encoder.popDebugGroup()
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let n = Int(detection.groupingParameters.adjacencyListCounter.pointer[0])
        
        var adjacency = [Int]()
        var neighbors = [Int]()
        var offsets = [Int]()
        
        let list = detection.groupingParameters.adjacencyList.buffer.contents().assumingMemoryBound(to: Int32.self)
        let _neighbors = detection.groupingParameters.neighborCount.buffer.contents().assumingMemoryBound(to: Int32.self)
        let _offsets = detection.groupingParameters.offset.buffer.contents().assumingMemoryBound(to: Int32.self)
        
        for i in 0..<n {
            adjacency.append(Int(list[i]))
        }
        
        for i in 0..<Int(count) {
            neighbors.append(Int(_neighbors[i]))
            offsets.append(Int(_offsets[i]))
        }

        for i in 0..<Int(count) {
            let num = neighbors[i]
            let x = array[i].x
            if x == 10 {
                XCTAssert(num == an, "Invalid Neighbor Count")
            } else if x == 40 {
                XCTAssert(num == bn, "Invalid Neighbor Count")
            } else if x == 60 {
                XCTAssert(num == cn, "Invalid Neighbor Count")
            } else if x == 80 {
                XCTAssert(num == dn, "Invalid Neighbor Count")
            } else if x == 100 {
                XCTAssert(num == en, "Invalid Neighbor Count")
            }
        }
        
        for i in 0..<Int(count) {
            let offset = offsets[i]
            for j in 0..<Int(count) {
                let offset2 = offsets[j]
                if i != j {
                    XCTAssert(offset != offset2, "Offsets not unique")
                }
            }
        }
        
        var num : Int = 0
        var x : Int = 0
        var start : Int = 0
        var end : Int = 0
        var x2: Int = 0
        
        for i in 0..<Int(count) {
            num = neighbors[i]
            x = Int(array[i].x)
            start = offsets[i]
            end = start + num
            
            for j in start..<end {
                XCTAssert(j < adjacency.count)
                
                if adjacency[j] < 0 || adjacency[j] >= array.count {
                    XCTAssert(adjacency[j] > 0 && adjacency[j] < array.count)
                }
                
                x2 = Int(array[adjacency[j]].x)
                if (x2 != x) {
                    print(j)
                    XCTAssert(x2 == x, "Invalid Adjacency List, index \(i), offset \(j)")
                }
                
            }
            
        }
        
    }

    func testGrouping() {
        
        let maxPts = 1500;
        var minNeighbors : Int32 = 3;
        var numWindows : Int32 = 40
        var (_clusters, adjacencies, offsets, neighbors) = generateTestRectangles(minNeighbors: Int(minNeighbors), n: Int(numWindows))
        
        var clusterCount : Int32 = 0;
        var clusters = [Int32](repeating: -1, count: Int(numWindows))
        
        let neighborsBuf = Context.device().makeBuffer(bytes: &neighbors, length: neighbors.count * MemoryLayout<Int32>.size, options: .storageModeShared)
        let offsetBuf = Context.device().makeBuffer(bytes: &offsets, length: offsets.count * MemoryLayout<Int32>.size, options: .storageModeShared)
        let adjacenciesBuf = Context.device().makeBuffer(bytes: &adjacencies, length: adjacencies.count * MemoryLayout<Int32>.size, options: .storageModeShared)
        let clusterCountBuf = Context.device().makeBuffer(bytes: &clusterCount, length: MemoryLayout<Int32>.size, options: .storageModeShared)
        let clustersBuf = Context.device().makeBuffer(bytes: &clusters, length: clusters.count * MemoryLayout<Int32>.size, options: .storageModeShared)
        
        let commandBuffer = Context.commandQueue().makeCommandBuffer()
        let encoder = commandBuffer.makeComputeCommandEncoder()
        
        // calculate the adjacency list
        encoder.setComputePipelineState(self.pipeline8)
        
        encoder.setBytes(&numWindows, length: 4, at: 0)
        encoder.setBytes(&minNeighbors, length: 4, at: 1)
        encoder.setBuffer(neighborsBuf, offset: 0, at: 2)
        encoder.setBuffer(offsetBuf, offset: 0, at: 3)
        encoder.setBuffer(adjacenciesBuf, offset: 0, at: 4)
        encoder.setBuffer(clusterCountBuf, offset: 0, at: 5)
        encoder.setBuffer(clustersBuf, offset: 0, at: 6)
        
        encoder.setThreadgroupMemoryLength(MemoryLayout<Int32>.size, at: 0)
        encoder.setThreadgroupMemoryLength(MemoryLayout<Int32>.size, at: 1)
        encoder.setThreadgroupMemoryLength(maxPts * MemoryLayout<Int32>.size, at: 2)
        encoder.setThreadgroupMemoryLength(MemoryLayout<Int32>.size, at: 3)
        encoder.setThreadgroupMemoryLength(maxPts * MemoryLayout<Int32>.size, at: 4)
        encoder.setThreadgroupMemoryLength(MemoryLayout<Int32>.size, at: 5)
        
        assert(128 <= self.pipeline8.maxTotalThreadsPerThreadgroup)
        
        encoder.dispatchThreadgroups(MTLSizeMake(1, 1, 1), threadsPerThreadgroup: MTLSizeMake(128, 1, 1))
        
        encoder.popDebugGroup()
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)
        
        let resultClusterNumber = Int(clusterCountBuf.contents().assumingMemoryBound(to: Int32.self).pointee)
        let resultClusterPtr = clustersBuf.contents().assumingMemoryBound(to: Int32.self)
        var results = [[Int]]()
        
        for i in 0..<Int(numWindows) {
            results.append([i,Int(resultClusterPtr[i])])
        }
        
        // the noise
        let noise = results.filter { (v) -> Bool in
            return v[1] < 0
        }.map { (v) -> Int in
            return v[0]
        }
        
        var foundNodes = [Int : Int]()
        
        for (index,c) in _clusters.enumerated() {
            if c.count > 0 {
                for cc in c {
                    assert(foundNodes[cc] == nil)
                    foundNodes[cc] = index + 1
                }
            }
        }
        
        let actualNoise = Int(numWindows) - foundNodes.count
        
        // empty clusters are noise values
        assert(noise.count == actualNoise)
        
        // filter empty clusters
        _clusters = _clusters.filter({ (v) -> Bool in
            return v.count != 0
        })
        
        // must be the same
        XCTAssertTrue(resultClusterNumber == _clusters.count)
        
        results = results.filter { (v) -> Bool in
            return v[1] >= 0
        }

        let uniqueClusters = Array(Set(results.map { (v) -> Int in
            return v[1]
        }))
        
        var clusterMap = [Int: [Int]]()
        
        for c in uniqueClusters.sorted() {
            
            let nodes = (results.map { (v) -> [Int] in
                return [v[0],v[1]]
            }).filter({ (v) -> Bool in
                return v[1] == c
            }).map({ (v) -> Int in
                return v[0]
            }).sorted()
            
            clusterMap[c] = nodes
            //print("\(c) -> \((nodes.flatMap { String($0) }).joined(separator: ","))")
        }
        
        
        for c in _clusters.enumerated() {
            let actualNodes = c
            
            var correspIdx = -1
            for num in clusterMap.enumerated() {
    
                let k = num.element.key
                let vs = num.element.value
                
                if vs.contains(actualNodes.element[0]) {
                    correspIdx = k
                    break
                }
            }
            
            XCTAssert(correspIdx >= 0)
            XCTAssertNotNil(clusterMap[correspIdx])
            let m = clusterMap[correspIdx]!
            
            for n in actualNodes.element {
                XCTAssert(m.contains(n) == true)
            }
        }
        
        
        
        
    }
    
    
    
    func testAssembleRectangles() {
        
        var numWindows : Int32 = 10
        var scale : Float32 = 1.0
        var windows = Array<DetectionWindow>()

        for i in 1...Int(numWindows) {
            let v = UInt32(i)
            let w = DetectionWindow(x: v, y: v, width: v, height: v)
            windows.append(w)
        }

        var clusters = [Int32](repeating: 1, count: windows.count)
        
        let clustersBuf = Context.device().makeBuffer(bytes: &clusters, length: clusters.count * MemoryLayout<Int32>.size, options: .storageModeShared)
        let windowsBuffer = Context.device().makeBuffer(bytes: &windows, length: windows.count * MemoryLayout<DetectionWindow>.size, options: .storageModeShared)
        let rectanglesBuf = Context.device().makeBuffer(length: 4 * MemoryLayout<Float32>.size * 1500, options: .storageModeShared)

        let commandBuffer = Context.commandQueue().makeCommandBuffer()
        let encoder = commandBuffer.makeComputeCommandEncoder()
        
        // calculate the adjacency list
        encoder.setComputePipelineState(self.pipeline9)
        
        encoder.setBytes(&numWindows, length: 4, at: 0)
        encoder.setBuffer(windowsBuffer, offset: 0, at: 1)
        encoder.setBuffer(clustersBuf, offset: 0, at: 2)
        encoder.setBuffer(rectanglesBuf, offset: 0, at: 3)
        encoder.setBytes(&scale, length: 4, at: 4)
        
        encoder.setThreadgroupMemoryLength(MemoryLayout<Float32>.size * 4 * 64 * 2, at: 0)
        encoder.setThreadgroupMemoryLength(MemoryLayout<Int32>.size * 64 * 2, at: 1)
        encoder.dispatchThreadgroups(MTLSizeMake(1, 1, 1), threadsPerThreadgroup: MTLSizeMake(64, 1, 1))
        
        encoder.popDebugGroup()
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        XCTAssertNil(commandBuffer.error)
        
        let ptr = rectanglesBuf.contents().assumingMemoryBound(to: Float32.self)
        let x = ptr[0]
        let y = ptr[1]
        let width = ptr[2]
        let height = ptr[3]
        
        XCTAssertEqual((windows.map { return Float32($0.x) } ).reduce(0, +) / Float32(numWindows), x)
        XCTAssertEqual((windows.map { return Float32($0.y) } ).reduce(0, +) / Float32(numWindows), y)
        XCTAssertEqual((windows.map { return Float32($0.width) } ).reduce(0, +) / Float32(numWindows), width)
        XCTAssertEqual((windows.map { return Float32($0.height) } ).reduce(0, +) / Float32(numWindows), height)
        
    }
    
    
}

struct Rect {
    var x : Float
    var y : Float
    var width : Float
    var height : Float
}

func isInRegion(_ w1: Rect, _ w2: Rect, _ eps: Float) -> Bool {
    
    let delta = eps * (min(w1.width, w2.width) + min(w1.height, w2.height)) * 0.5
    
    // check whether we are within the region
    return abs(w1.x - w2.x) <= delta &&
        abs(w1.y - w2.y) <= delta &&
        abs(w1.x + w1.width - w2.x - w2.width) <= delta &&
        abs(w1.y + w1.height - w2.y - w2.height) <= delta;
}

func generateTestRectangles(minNeighbors: Int, n: Int) -> (clusters: [[Int]], adjacencies: [Int32], offsets: [Int32], neighbors: [Int32]) {
    
    var list = [Rect]()
    
    for _ in 0..<n {
        let x = Float(arc4random()) / Float(UINT32_MAX) * 720
        let y = Float(arc4random()) / Float(UINT32_MAX) * 1280
        list.append(Rect(x: x, y: y, width: 300, height: 300))
    }
    
    list.shuffle()
    
    var adjacencies = [Int32]()
    var offsets = [Int32]()
    var neighbors = [Int32]()
    var currentOffset : Int32 = 0
    
    for r1 in list {

        var rneighbors = [Int32]()
        
        for i in 0..<list.count {
            
            let r2 = list[i]
            
            if isInRegion(r1, r2, 0.5) {
                rneighbors.append(Int32(i))
            }
        }
        
        offsets.append(currentOffset)
        currentOffset = currentOffset + Int32(rneighbors.count)
        neighbors.append(Int32(rneighbors.count))
        adjacencies = adjacencies + rneighbors
    }
    
    var visited = [Bool](repeating: false, count: n)
    var clusters = [[Int]]()
    
    for i in 0..<n {
        
        if (visited[i] || neighbors[i] < Int32(minNeighbors)) {
            continue
        }
        
        var _clusters = [Int]()
        var marked = [Int](repeating: 0, count: n)
        marked[i] = 1
        
        while (true) {
            
            var stop = true
            
            for j in 0..<n {
                
                if marked[j] == 1 && !visited[j] {
                    
                    visited[j] = true
                    marked[j] = 0
                    _clusters.append(j)
                    
                    for z in Int(offsets[j])..<Int(offsets[j]+neighbors[j]) {
                        let idx = Int(adjacencies[z])
                        if (!visited[idx]) {
                            marked[idx] = 1
                            stop = false
                        }
                    }
                    
                }
                
            }
            
            if stop {
                break
            }
        }
        
        clusters.append(_clusters)
    }
    
    
    /*for c in clusters {
        
        let c1 = Float(arc4random()) / Float(UINT32_MAX)
        let c2 = Float(arc4random()) / Float(UINT32_MAX)
        let c3 = Float(arc4random()) / Float(UINT32_MAX)
        
        //let color = "[\(c1) \(c2) \(c3)]"
        
        for cc in c {
            let r = list[cc]
            //let position = "[\(r.x) \(r.y) \(r.width) \(r.height)]"
            /*if false {
                print("rectangle('Position',\(position),'EdgeColor',\(color))")
            }*/
        }
        
    }*/

    return (clusters, adjacencies, offsets, neighbors)
    //rectangle('Position',[3 0 2 4],'EdgeColor',[0 .5 .5])
    
    
    
}

















func testtest(count: Int, graph: [Node]) {
    
    var indices = [Int]()
    for i in 0..<count {
        indices.append(i)
    }

    var list = [Int]()
    
    for i in indices {
        //var u = i
        let node = findNode(id: i, graph: graph)!
        if node.parent > node.id {
            continue
        }
        var b = false
        for n in node.nodes {
            if n.id > node.id {
                b = true
            }
        }
        if (!b) {
            list.append(node.id)
        }
    }
    
    
}

func shuffleIndices(id: Int, graph: inout [Node]) {
    
    var indices = [Int]()
    for i in 0..<id {
        indices.append(i)
    }
    indices.shuffle()
    
    for (index,i) in indices.enumerated() {
        findNodeAndSet(id: index, graph: &graph, newid: i)
    }
    
}


func buildClusterInfos(id: Int, graph: [Node], windows: [Int]) -> (adjacencies: [Int32], neighbors: Int32){
    
    let node = findNode(id: id, graph: graph)!
    
    var neighborIndices = [id]
    
    if node.parent >= 0 {
        neighborIndices.append(node.parent)
    }
    
    for n in node.nodes {
        neighborIndices.append(n.id)
    }
    
    var adjacencies = [Int32]()
    
    for n in neighborIndices {
        adjacencies.append(Int32(windows.index(of: n)!))
        //adjacencies.append(Int32(n))
    }
    
    return (adjacencies, Int32(neighborIndices.count))
    
}

func findNodeAndSet(id: Int, graph: inout [Node], newid: Int) {
    
    for i in 0..<graph.count {
        
        if graph[i].id == id {
            graph[i].id = newid
            return
        }

        findNodeAndSet(id: id, graph: &graph[i].nodes, newid: newid)
    
    }
    
    return
    
}


func findNode(id: Int, graph: [Node]) -> Node? {
    
    for n in graph {
        
        if n.id == id {
            return n
        }
        
        if let node = findNode(id: id, graph: n.nodes) {
            return node
        }
        
    }
    
    return nil
    
}

func generateRandomCluster(maxDepth: Int, maxNeighbors: Int, id: inout Int) -> Node {
    return generateRandomNode(depth: 0, maxDepth: maxDepth, maxNeighbors: maxNeighbors, id: &id, parent: -1)!
}

func generateRandomNode(depth: Int, maxDepth: Int, maxNeighbors: Int, id: inout Int, parent: Int) -> Node? {
    
    if depth == maxDepth {
        return nil
    }
    
    var node = Node(id, parent)
    id = id + 1
    let neighbors = randomNumber(max: maxNeighbors)
    
    for _ in 0..<neighbors {
        guard let n = generateRandomNode(depth: depth+1, maxDepth: maxDepth, maxNeighbors: maxNeighbors, id: &id, parent: node.id) else {
            return node
        }
        node.nodes.append(n)
    }
    
    return node
    
}

func randomNumber(max: Int) -> Int {
    return Int(Float(arc4random()) / Float(UINT32_MAX) * Float(max)) + 1
}

struct Node {
    var nodes = [Node]()
    var id : Int
    var parent : Int
    init(_ id: Int, _ parent: Int) {
        self.id = id
        self.parent = parent
    }
}

func buildNeighborCount(arr: [Int]) -> [Int32] {
    let unique = Array(Set(arr))
    let neighbors = unique.map { (w) -> Int in
        return arr.filter({ (v) -> Bool in
            w == v
        }).count
    }
    return arr.map({ (v) -> Int32 in
        return Int32(neighbors[unique.index(of: Int(v))!])
    })
}

func buildAdjacencyList(arr: [Int], neighbors: [Int32]) -> (offset: [Int32], adjacencies: [Int32]){
    
    var adj = [Int32]()
    var offset = [Int32]()
    var offsetCount = 0;
    
    for (i,_v) in arr.enumerated() {
        let n = neighbors[i]
        let localadj = arr.enumerated().map({ (v) -> Int32 in
            if _v == v.element {
                return Int32(v.offset)
            } else {
                return -1
            }
        }).filter({ (v) -> Bool in
            return v >= 0
        })
        assert(n == Int32(localadj.count))
        adj = adj + localadj
        offset.append(Int32(offsetCount))
        offsetCount = offsetCount + localadj.count
    }
    
    return (offset, adj)
}



extension MutableCollection where Indices.Iterator.Element == Index {
    /// Shuffles the contents of this collection.
    mutating func shuffle() {
        let c = count
        guard c > 1 else { return }
        
        for (firstUnshuffled , unshuffledCount) in zip(indices, stride(from: c, to: 1, by: -1)) {
            let d: IndexDistance = numericCast(arc4random_uniform(numericCast(unshuffledCount)))
            guard d != 0 else { continue }
            let i = index(firstUnshuffled, offsetBy: d)
            swap(&self[firstUnshuffled], &self[i])
        }
    }
}

extension Sequence {
    /// Returns an array with the contents of this sequence, shuffled.
    func shuffled() -> [Iterator.Element] {
        var result = Array(self)
        result.shuffle()
        return result
    }
}
