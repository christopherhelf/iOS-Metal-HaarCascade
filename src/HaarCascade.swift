//
//  HaarCascade.swift
//
//  Created by Christopher Helf on 12.11.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation
import MetalKit

class HaarCascade {
    
    struct sHaarCascade {
        var numOfStages : Int32
        var totalNumOfClassifiers : Int32
        var originalWindowSizeWidth : Int32
        var originalWindowSizeHeight : Int32
        var windowSizeWidth : Int32
        var windowSizeHeight : Int32
        var detectionSizeWidth : Int32
        var detectionSizeHeight : Int32
        var realWindowSizeWidth : Int32
        var realWindowSizeHeight : Int32
        var step : Float32
    }
    
    struct sRect {
        var x : Float32
        var y : Float32
        var width : Float32
        var height : Float32
    }
    
    struct sFeatureRect {
        var r : sRect
        var weight : Float32
    }
    
    struct sHaarFeature {
        var r1 : sFeatureRect
        var r2 : sFeatureRect
        var r3 : sFeatureRect
    }
    
    struct sHaarClassifier {
        var haarFeature : sHaarFeature
        var threshold : Float32
        var alpha0 : Float32
        var alpha1 : Float32
    }
    
    struct sHaarStageClassifier {
        var numOfClassifiers : Int32
        var classifierOffset : Int32
        var threshold : Float32
    }
    
    struct Buffers {
        var haarCascadeBuffer : MTLBuffer
        var haarStageClassifiersBuffer : MTLBuffer
        var scaledHaarClassifiersBuffer: MTLBuffer
    }

    var haarCascade: sHaarCascade
    var haarStageClassifiers = [sHaarStageClassifier]()
    var haarClassifiers = [sHaarClassifier]()
    var scaledHaarClassifiers = [sHaarClassifier]()
    var copyPipeline : MTLComputePipelineState
    
    init(data: Data, width: Int, height: Int) throws {

        // get the root node
        let cascadeRoot = try HaarCascade.getRootNode(data: data)
        
        // initial parameters
        let step : Float32 = 1
        
        // get the stages
        let stages = try HaarCascade.getStages(root: cascadeRoot)
        
        // get the stage count
        let numberOfStages = stages.count
        
        // get the size
        let (originalWindowSizeWidth, originalWindowSizeHeight) = try HaarCascade.getWindowSize(root: cascadeRoot)
        
        // counter for offsets
        var gpuClassifierCounter : Int32 = 0;
        
        // get all the classifiers
        for i in 0..<numberOfStages {
            
            // get the stage classifier, and the tree
            let (sStage, classifiers) = try HaarCascade.getHaarStageClassifier(stages[i], count: gpuClassifierCounter)
        
            // iterate through all classifiers
            for j in 0..<Int(sStage.numOfClassifiers) {
                
                let classifier = classifiers[j]
                let feature = try HaarCascade.getHaarFeature(classifier)
                let sClassifier = try HaarCascade.getHaarClassifier(classifier, feature: feature)
                
                // also store a scaled version (to be scaled later during inititalization)
                haarClassifiers.append(sClassifier)
                scaledHaarClassifiers.append(sClassifier)
                
                gpuClassifierCounter += 1
            }
            
            haarStageClassifiers.append(sStage)

        }
        
        // initialize the haar cascade
        haarCascade = sHaarCascade(numOfStages: Int32(numberOfStages), totalNumOfClassifiers: gpuClassifierCounter, originalWindowSizeWidth: originalWindowSizeWidth, originalWindowSizeHeight: originalWindowSizeHeight, windowSizeWidth: Int32(width), windowSizeHeight: Int32(height), detectionSizeWidth: 0, detectionSizeHeight: 0, realWindowSizeWidth: 0, realWindowSizeHeight: 0, step: step)
        
        // initialize the compute pipeline
        let copyFunc = Context.library().makeFunction(name: "haarCopyClassifiers")!
        copyPipeline = try! Context.device().makeComputePipelineState(function: copyFunc)
        
    }
    
    func setFeaturesForScale(scale: Float) {
        
        // set the scale
        haarCascade.step = Float32(scale)
        
        // calculate the actual window size
        haarCascade.realWindowSizeWidth = Int32(round(Float(haarCascade.originalWindowSizeWidth) * scale))
        haarCascade.realWindowSizeHeight = Int32(round(Float(haarCascade.originalWindowSizeHeight) * scale))
        
        // calculate the detection size (rects always start at x/y)
        haarCascade.detectionSizeWidth = haarCascade.windowSizeWidth - haarCascade.realWindowSizeWidth
        haarCascade.detectionSizeHeight = haarCascade.windowSizeHeight - haarCascade.realWindowSizeHeight
    
        // scale the features accordingly
        for i in 0..<haarCascade.totalNumOfClassifiers {
            
            let originalFeature = haarClassifiers[Int(i)].haarFeature
            var scaledFeature = scaledHaarClassifiers[Int(i)].haarFeature
            
            scaledFeature.r1.r.x = originalFeature.r1.r.x * scale
            scaledFeature.r1.r.y = originalFeature.r1.r.y * scale
            scaledFeature.r1.r.width = originalFeature.r1.r.width * scale
            scaledFeature.r1.r.height = originalFeature.r1.r.height * scale
            
            scaledFeature.r2.r.x = originalFeature.r2.r.x * scale
            scaledFeature.r2.r.y = originalFeature.r2.r.y * scale
            scaledFeature.r2.r.width = originalFeature.r2.r.width * scale
            scaledFeature.r2.r.height = originalFeature.r2.r.height * scale
            
            if originalFeature.r3.weight != 0 {
                scaledFeature.r3.r.x = originalFeature.r3.r.x * scale
                scaledFeature.r3.r.y = originalFeature.r3.r.y * scale
                scaledFeature.r3.r.width = originalFeature.r3.r.width * scale
                scaledFeature.r3.r.height = originalFeature.r3.r.height * scale
            }
            
            scaledHaarClassifiers[Int(i)].haarFeature = scaledFeature
        }
        
    }
    
    func getBuffers() -> Buffers {
        
        let scale = self.haarCascade.step
        let haarCascadeBuffer = Context.device().makeBuffer(bytes: &self.haarCascade, length: MemoryLayout<sHaarCascade>.size, options: MTLResourceOptions.storageModeShared)
        haarCascadeBuffer.label = "haarCascadeBuffer (\(scale))"
        
        let haarStageClassifiersBuffer = Context.device().makeBuffer(bytes: &self.haarStageClassifiers, length: MemoryLayout<sHaarStageClassifier>.size * haarStageClassifiers.count, options: MTLResourceOptions.storageModeShared)
        haarStageClassifiersBuffer.label = "haarStageClassifiersBuffer (\(scale))"
        
        let scaledHaarClassifiersBuffer = Context.device().makeBuffer(bytes: &self.scaledHaarClassifiers, length: MemoryLayout<sHaarClassifier>.size * scaledHaarClassifiers.count, options: MTLResourceOptions.storageModeShared)
        scaledHaarClassifiersBuffer.label = "scaledHaarClassifiersBuffer (\(scale))"
        
        // Copy the haar classifiers buffer to the gpu
        let scaledHaarClassifiersGPUBuffer = Context.device().makeBuffer(length: MemoryLayout<sHaarClassifier>.size * scaledHaarClassifiers.count, options: MTLResourceOptions.storageModePrivate)
        
        // get the number of classifiers
        var numberOfClassifiers = scaledHaarClassifiers.count
        
        // copy the cpu buffer to the gpu
        let commandBuffer = Context.commandQueue().makeCommandBuffer()
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.setComputePipelineState(self.copyPipeline)
        encoder.setBytes(&numberOfClassifiers, length: MemoryLayout<Int>.size, at: 0)
        encoder.setBuffer(scaledHaarClassifiersBuffer, offset: 0, at: 1)
        encoder.setBuffer(scaledHaarClassifiersGPUBuffer, offset: 0, at: 2)
        encoder.dispatchThreadgroups(MTLSizeMake(32, 1, 1), threadsPerThreadgroup: MTLSizeMake((numberOfClassifiers+31)/32, 1, 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
                
        return Buffers(haarCascadeBuffer: haarCascadeBuffer, haarStageClassifiersBuffer: haarStageClassifiersBuffer, scaledHaarClassifiersBuffer: scaledHaarClassifiersGPUBuffer)
    }
    
    func getWindowSizeWidth() -> Int {
        return Int(self.haarCascade.originalWindowSizeWidth)
    }
    
    func getWindowSizeHeight() -> Int {
        return Int(self.haarCascade.originalWindowSizeHeight)
    }
    
    func getDetectionSizeWidth() -> Int {
        return Int(self.haarCascade.detectionSizeWidth)
    }
    
    func getDetectionSizeHeight() -> Int {
        return Int(self.haarCascade.detectionSizeHeight)
    }
    
    func getStep() -> Int {
        return Int(self.haarCascade.step)
    }
}

extension HaarCascade {
    enum Errors : Error {
        case invalidSyntax
        case parsingError
        case classifierMoreThanOneHaarFeature
    }
}

extension HaarCascade {
    
    class func getRootNode(data: Data) throws -> [String : AnyObject] {
        
        guard let _obj = try? JSONSerialization.jsonObject(with: data, options: JSONSerialization.ReadingOptions.mutableContainers), let jsonObject = _obj as? [String:AnyObject] else {
            throw Errors.invalidSyntax
        }
        
        guard let jsonRoot = jsonObject["opencv_storage"] as? [String : AnyObject], jsonRoot.count == 1 else {
            throw Errors.invalidSyntax
        }
        
        guard let cascadeRoot = jsonRoot.first?.value as? [String : AnyObject] else {
            throw Errors.invalidSyntax
        }
        
        return cascadeRoot
    }
    
    class func getStages(root: [String : AnyObject]) throws -> [AnyObject]{
        
        guard let stages = (root["stages"] as? [String:AnyObject])?["_"] as? [AnyObject] else {
            throw Errors.invalidSyntax
        }
        
        return stages
    }
    
    class func getWindowSize(root: [String : AnyObject]) throws -> (Int32, Int32) {
        guard
            let sizeStr = HaarCascade.parseIntArrayFromString(root, "size"),
            sizeStr.count == 2
            else {
                throw Errors.invalidSyntax
        }
        return(sizeStr[0], sizeStr[1])
    }
    
    class func getHaarStageClassifier(_ _stage: AnyObject, count: Int32) throws -> (sHaarStageClassifier, [[String: AnyObject]]) {
        
        guard let stage = _stage as? [String:AnyObject] else {
            throw Errors.invalidSyntax
        }
        
        guard let threshold = HaarCascade.parseAsFloat(stage, "stage_threshold") else {
            throw Errors.invalidSyntax
        }
        
        guard var classifiers = (stage["trees"] as? [String:AnyObject])?["_"] as? [[String: AnyObject]] else {
            throw Errors.invalidSyntax
        }
        
        classifiers = try classifiers.map({ (v) -> [String: AnyObject] in
            guard let _v = v["_"] as? [String: AnyObject] else {
                if let _ = v["_"] as? [AnyObject] {
                    throw Errors.classifierMoreThanOneHaarFeature
                } else {
                    throw Errors.invalidSyntax
                }
            }
            return _v
        })
        
        let numofClassifiers = Int32(classifiers.count)
        
        return (sHaarStageClassifier(numOfClassifiers: numofClassifiers, classifierOffset: count, threshold: threshold), classifiers)
        
    }
    
    class func getHaarFeature(_ classifier : [String:AnyObject]) throws -> sHaarFeature {
        
        guard let rects = (((classifier["feature"] as? [String:AnyObject])?["rects"]) as? [String:AnyObject])?["_"] as? [String] else {
            throw Errors.invalidSyntax
        }
        
        var rectsMapped = try rects.map({ (s) -> sFeatureRect in
            guard let vals = HaarCascade.parseSeparatedStringAsFloats(s: s), vals.count == 5 else {
                throw Errors.invalidSyntax
            }
            let rect = sRect(x: vals[0], y: vals[1], width: vals[2], height: vals[3])
            return sFeatureRect(r: rect, weight: vals[4])
        })
        
        guard rectsMapped.count >= 2 else {
            throw Errors.invalidSyntax
        }
        
        if rectsMapped.count == 2 {
            rectsMapped.append(sFeatureRect(r: sRect(x: 0, y: 0, width: 0, height: 0), weight: 0))
        }
        
        return sHaarFeature(r1: rectsMapped[0], r2: rectsMapped[1], r3: rectsMapped[2])
    }
    
    class func getHaarClassifier(_ classifier: [String:AnyObject], feature: sHaarFeature) throws -> sHaarClassifier {
        
        guard let classifierThreshold = HaarCascade.parseAsFloat(classifier, "threshold") else {
            throw Errors.invalidSyntax
        }
        
        guard let alpha0 = HaarCascade.parseAsFloat(classifier, "left_val") else {
            throw Errors.invalidSyntax
        }
        
        guard let alpha1 = HaarCascade.parseAsFloat(classifier, "right_val") else {
            throw Errors.invalidSyntax
        }
        
        return sHaarClassifier(haarFeature: feature, threshold: classifierThreshold, alpha0: alpha0, alpha1: alpha1)
    }
    
}

extension HaarCascade {
    
    class func parseAsFloat(_ o: [String: AnyObject], _ k: String) -> Float32? {
        if let _v = o[k] as? String {
            if let __v = Float(_v) {
                return Float32(__v)
            } else {
                return nil
            }
        } else if let _v = o[k] as? Double {
            return Float32(Float(_v))
        } else {
            return nil
        }
    }
    
    class func parseAsFloat(_ _o: AnyObject, _ k: String) -> Float32? {
        if let o = _o as? [String:AnyObject] {
            return HaarCascade.parseAsFloat(o, k)
        } else {
            return nil
        }
    }
    
    class func parseSeparatedStringAsFloats(s: String) -> [Float32]? {
        let v = s.components(separatedBy: " ")
        guard v.count > 0 else { return nil }
        return try? v.map { (s) -> Float32 in
            guard let ii = Float(s) else { throw Errors.parsingError }
            return Float32(ii)
        }
    }
    
    class func parseSeparatedStringAsInts(s: String) -> [Int32]? {
        let v = s.components(separatedBy: " ")
        guard v.count > 0 else { return nil }
        return try? v.map { (s) -> Int32 in
            guard let ii = Int(s) else { throw Errors.parsingError }
            return Int32(ii)
        }
    }
    
    class func parseIntArrayFromString(_ o: [String: AnyObject], _ k: String) -> [Int32]? {
        guard let _v = o[k] as? String else { return nil }
        return parseSeparatedStringAsInts(s: _v)
    }
    
    class func parseFloatArrayFromString(_ o: [String: AnyObject], _ k: String) -> [Float32]? {
        if let _v = o[k] as? String {
            return parseSeparatedStringAsFloats(s: _v)
        } else {
            return nil
        }
    }
    
}





