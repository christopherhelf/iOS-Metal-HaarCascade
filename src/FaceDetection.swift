//
//  MetalBasedFaceDetection.swift
//
//  Created by Christopher Helf on 12.11.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.

import Foundation
import MetalKit
import MetalPerformanceShaders

class FaceDetection {
    
    let scaleFactor : Float = 1.2
    let minNeighbors = 3
    
    var buffers = Array<HaarCascade.Buffers>()
    var sizes = [(width: Int, height: Int)]()
    var scales = [Float]()
    
    let conversion : MPSImageConversion
    let integral : MPSImageIntegral
    let sqIntegral : MPSImageIntegralOfSquares
    let haarPipeline : MTLComputePipelineState
    
    var histogramInfo : MPSImageHistogramInfo
    let histogram : MPSImageHistogram
    let eqHistograms : MPSImageHistogramEqualization
    let histogramInfoBuffer : MTLBuffer
    
    var grayscaleTex : MTLTexture
    let integralTex : MTLTexture
    let sqIntegralTex : MTLTexture
    
    var detectedFaceBuffer : MTLBuffer
    var detectedFacePtr : UnsafeMutablePointer<HaarCascade.sRect>
    
    var counterBuffer : MTLBuffer
    var counterPtr : UnsafeMutableBufferPointer<Int32>
    
    let computeHistogram = false
    
    init(path: URL, width: Int, height: Int) {
        
        guard let data = try? Data(contentsOf: path) else {
            fatalError()
        }
        
        let cascade = try! HaarCascade(data: data, width: width, height: height)
        var factor : Float = 16
        
        repeat {

            scales.append(factor)
            cascade.setFeaturesForScale(scale: factor)
            sizes.append((cascade.getDetectionSizeWidth()/cascade.getStep(),cascade.getDetectionSizeHeight()/cascade.getStep()))
            buffers.append(cascade.getBuffers())
            factor *= scaleFactor

        } while (Int(factor * Float(cascade.getWindowSizeWidth())) < width - 10 && Int(factor * Float(cascade.getWindowSizeHeight())) < height - 10)
        
        let conversionInfo = CGColorConversionInfo(src: CGColorSpaceCreateDeviceRGB(), dst: CGColorSpaceCreateDeviceGray())
        conversion = MPSImageConversion(device: Context.device(), srcAlpha: .alphaIsOne, destAlpha: .alphaIsOne, backgroundColor: nil, conversionInfo: conversionInfo)
        conversion.label = "imageConversion"
        integral = MPSImageIntegral(device: Context.device())
        integral.label = "imageIntegral"
        sqIntegral = MPSImageIntegralOfSquares(device: Context.device())
        sqIntegral.label = "imageIntegralSquares"
        
        histogramInfo = MPSImageHistogramInfo(
            numberOfHistogramEntries: 256,
            histogramForAlpha: false,
            minPixelValue: vector_float4(0,0,0,0),
            maxPixelValue: vector_float4(1,1,1,1))
        
        histogram = MPSImageHistogram(device: Context.device(), histogramInfo: &histogramInfo)
        histogram.label = "imageHistogram"
        histogramInfoBuffer = Context.device().makeBuffer(bytes: &histogramInfo, length: MemoryLayout.size(ofValue: histogramInfo), options: MTLResourceOptions.optionCPUCacheModeWriteCombined)
        histogramInfoBuffer.label = "histogramInfoBuffer"
        eqHistograms = MPSImageHistogramEqualization(device: Context.device(), histogramInfo: &histogramInfo)
        eqHistograms.label = "imageHistogramEqualization"
        
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: MTLPixelFormat.r32Float,
            width: width,
            height: height,
            mipmapped: false)
        
        grayscaleTex = Context.device().makeTexture(descriptor: descriptor)
        grayscaleTex.label = "grayscale"
        integralTex = Context.device().makeTexture(descriptor: descriptor)
        integralTex.label = "imageIntegralTexture"
        sqIntegralTex = Context.device().makeTexture(descriptor: descriptor)
        sqIntegralTex.label = "imageIntegralSquaresTexture"
        
        let function = Context.library().makeFunction(name: "haarDetection")!
        function.label = "haarDetection"
        haarPipeline = try! Context.device().makeComputePipelineState(function: function)
        
        detectedFaceBuffer = Context.device().makeBuffer(length: width*height*MemoryLayout<HaarCascade.sRect>.size, options: .storageModeShared)
        detectedFaceBuffer.label = "detectedFacesBuffer"
        detectedFacePtr = UnsafeMutablePointer<HaarCascade.sRect>(OpaquePointer(detectedFaceBuffer.contents()))
        
        var counter = 0
        counterBuffer = Context.device().makeBuffer(bytes: &counter, length: MemoryLayout<Int32>.size, options: MTLResourceOptions.storageModeShared)
        counterBuffer.label = "counterBuffer"
        let ptr = UnsafeMutablePointer<Int32>(OpaquePointer(counterBuffer.contents()))
        counterPtr = UnsafeMutableBufferPointer<Int32>(start: ptr, count: 1)
    }
    
    
    func getFaces(commandBuffer: MTLCommandBuffer, input: MTLTexture, output: MTLTexture) {
        
        // reset counter
        counterPtr[0]=0
        
        // convert to gray
        conversion.encode(commandBuffer: commandBuffer, sourceTexture: input, destinationTexture: grayscaleTex)
        
        // histogram equalization, costs performance ...
        if computeHistogram {
            // histogram
            histogram.encode(to: commandBuffer, sourceTexture: grayscaleTex, histogram: histogramInfoBuffer, histogramOffset: 0)
        
            // equalize
            eqHistograms.encode(commandBuffer: commandBuffer, inPlaceTexture: &grayscaleTex, fallbackCopyAllocator: nil)
        }
        
        // integral image
        integral.encode(commandBuffer: commandBuffer, sourceTexture: grayscaleTex, destinationTexture: integralTex)
        
        // sq integral image
        sqIntegral.encode(commandBuffer: commandBuffer, sourceTexture: grayscaleTex, destinationTexture: sqIntegralTex)
        
        // haar kernel
        let block = MTLSizeMake(8, 32, 1)
        
        // encode
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()
        commandEncoder.pushDebugGroup("haarKernels")
        commandEncoder.setComputePipelineState(haarPipeline)
        commandEncoder.setTexture(integralTex, at: 0)
        commandEncoder.setTexture(sqIntegralTex, at: 1)
        commandEncoder.setTexture(output, at: 2)
        commandEncoder.setBuffer(counterBuffer, offset: 0, at: 3)
        commandEncoder.setBuffer(detectedFaceBuffer, offset: 0, at: 4)
        
        for (index,buffers) in self.buffers.enumerated() {
            commandEncoder.setBuffer(buffers.haarCascadeBuffer, offset: 0, at: 0)
            commandEncoder.setBuffer(buffers.haarStageClassifiersBuffer, offset: 0, at: 1)
            commandEncoder.setBuffer(buffers.scaledHaarClassifiersBuffer, offset: 0, at: 2)
            let threadgroups = MTLSizeMake(max(sizes[index].width/block.width,1)+1, sizes[index].height/block.height+1, 1)
            commandEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: block)
        }

        commandEncoder.popDebugGroup()
        commandEncoder.endEncoding()
        
        commandBuffer.addCompletedHandler { (_) in
            print("Found \(self.counterPtr[0]) Faces via Haar Cascade!")
        }
        
    }
    
    private func predicate(eps: Float, r1: HaarCascade.sRect, r2: HaarCascade.sRect) -> Bool {
        let delta = Float32(eps) * (min(r1.width, r2.width) + min(r1.height, r2.height))*0.5
        return abs(r1.x - r2.x) <= delta &&
            abs(r1.y - r2.y) <= delta &&
            abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
            abs(r1.y + r1.height - r2.y - r2.height) <= delta
    }
    
    private func partition(eps: Float) -> (Int, [Int]) {
        
        let N = Int(self.counterPtr[0])
        var labels = [Int](repeating: 0, count: N)
        let vec = detectedFacePtr
        
        let PARENT = 0
        let RANK = 1
        
        var nodes = [[Int]](repeating: [-1, 0], count: 2*N)
        
        
        for i in 0..<N {
            
            var root = i
            
            while nodes[root][PARENT] >= 0 {
                root = nodes[root][PARENT]
            }
            
            for j in 0..<N {
                
                if i==j || !predicate(eps: eps, r1: vec[i], r2: vec[j]) {
                    continue
                }
                
                var root2 = j
                
                while nodes[root2][PARENT] >= 0 {
                    root2 = nodes[root2][PARENT]
                }
                
                if root2 != root {
                    
                    let rank = nodes[root][RANK]
                    let rank2 = nodes[root2][RANK]
                    
                    if rank > rank2 {
                        nodes[root2][PARENT] = root
                    } else {
                        nodes[root][PARENT] = root2
                        nodes[root2][RANK] += rank == rank2 ? 1 : 0
                        root = root2;
                    }
                    
                    var k = j;
                    var parent = nodes[k][PARENT]
                    
                    while parent >= 0 {
                        nodes[k][PARENT] = root
                        k = parent
                        parent = nodes[k][PARENT]
                    }
                    
                    k = i
                    parent = nodes[k][PARENT]
                    
                    while parent >= 0 {
                        nodes[k][PARENT] = root
                        k = parent
                        parent = nodes[k][PARENT]
                    }
                    
                    
                    
                }
                
            }
            
        }
        
        var nClasses = 0
        for i in 0..<N {
            var root = i
            
            while nodes[root][PARENT] >= 0 {
                root = nodes[root][PARENT]
            }
            
            if nodes[root][RANK] >= 0 {
                nodes[root][RANK] = ~nClasses
                nClasses += 1
            }
            
            labels[i] = ~nodes[root][RANK]
            
        }
        
        return (nClasses, labels)
    }
    
    func groupRectangles(eps: Float) -> [CGRect] {
        
        if self.counterPtr[0] == 0 {
            return [CGRect]()
        }
        
        let (nClasses, labels) = partition(eps: eps)
        
        var rrects = [CGRect](repeating: CGRect(x: 0, y: 0, width: 0, height: 0), count: nClasses)
        var rweights = [Int](repeating: 0, count: nClasses)
        
        for i in 0..<labels.count {
            let cls : Int = labels[i]
            var rect = rrects[cls]
            rect = rect.offsetBy(dx: CGFloat(detectedFacePtr[i].x), dy: CGFloat(detectedFacePtr[i].y))
            rect.size.width += CGFloat(detectedFacePtr[i].width)
            rect.size.height += CGFloat(detectedFacePtr[i].height)
            rrects[cls] = rect
            rweights[cls] += 1
        }
        
        for i in 0..<nClasses {
            let s = 1.0/CGFloat(rweights[i])
            let r = rrects[i]
            let rect = CGRect(x: round(r.origin.x * s), y: round(r.origin.y * s), width: round(r.width * s), height: round(r.height * s))
            rrects[i] = rect
        }
        
        var rectList = [CGRect]()
        var j : Int = 0
        
        for i in 0..<nClasses {
            let r1 = rrects[i]
            let n1 = rweights[i]
            
            if n1 <= self.minNeighbors {
                continue
            }
            
            while j < nClasses {
                
                let n2 = rweights[j]
                
                if j==i || n2 <= self.minNeighbors {
                    j += 1
                    continue
                }
                
                let r2 = rrects[j]
                let dx = round(r2.width * CGFloat(eps))
                let dy = round(r2.height * CGFloat(eps))
                
                if( i != j &&
                    r1.origin.x >= r2.origin.x - dx &&
                    r1.origin.y >= r2.origin.y - dy &&
                    r1.origin.x + r1.width <= r2.origin.x + r2.width + dx &&
                    r1.origin.y + r1.height <= r2.origin.y + r2.height + dy &&
                    (n2 > max(3, n1) || n1 < 3) ) {
                    break
                }
                
                j += 1
        
            }
            
            if j == nClasses {
                rectList.append(r1)
            }

        }
        
        return rectList
    }
    
}

extension CGRect {
    func getHaarRect() -> HaarCascade.sRect {
        return HaarCascade.sRect(x: Float32(self.origin.x), y: Float32(self.origin.y), width: Float32(self.width), height: Float32(self.height))
    }
}
