//
//  HaarCascade+Structs.swift
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 07.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation
import MetalKit


extension HaarCascade {
    
    public struct HaarCascadeDescriptor {
        var scale: Float32
        var scaledArea: Float32
        var width: UInt32
        var height: UInt32
    }
    
    struct HaarStage {
        var classifiers: [HaarClassifier]
        var stageThreshold: Float32
    }
    
    struct HaarClassifier {
        var rects: [HaarFeatureRect]
        var threshold : Float32
        var left: Float32
        var right: Float32
    }
    
    struct HaarFeatureRect {
        var x : UInt8
        var y : UInt8
        var width : UInt8
        var height : UInt8
        var weight : Float32
    }
    
    func getScaledHaarCascadeBuffer(scale: Float32) -> Buffer<HaarCascadeDescriptor> {
        let width = UInt32(Float32(self.windowWidth) * scale)
        let height = UInt32(Float32(self.windowWidth) * scale)
        let scaledArea = Float32(width*height)
        var descriptor = HaarCascadeDescriptor(scale: scale, scaledArea: scaledArea, width: width, height: height)
        let buffer = Context.device().makeBuffer(bytes: &descriptor, length: MemoryLayout<HaarCascadeDescriptor>.size, options: .storageModeShared)
        buffer.label = "HaarCascadeBuffer"
        return Buffer(buffer: buffer)
    }
    
    func transformHaarFeatureRect(rect: HaarFeatureRect) -> [UInt32] {
        var weight = rect.weight
        let u1 = UInt32(rect.x) | UInt32(rect.y) << 8 | UInt32(rect.width) << 16 | UInt32(rect.height) << 24
        var u2 = UInt32(0)
        memcpy(&u2, &weight, 4)
        return [u1, u2]
    }
    
    func transformClassifier(classifier: HaarClassifier, rectOffset: UInt32) -> [Float32] {
        var a = UInt32(rectOffset) | UInt32(classifier.rects.count) << 24;
        var b : Float32 = 0
        memcpy(&b, &a, 4)
        return [b,classifier.threshold,classifier.left,classifier.right]
    }
    
    func transformStage(stage: HaarStage, classifierOffset: UInt32) -> [Float32] {
        var a = UInt32(stage.classifiers.count) | classifierOffset << 16
        var b : Float32 = 0
        memcpy(&b, &a, 4)
        return [b,stage.stageThreshold]
    }
    
    func getBuffersAndTexture() -> (Buffer<Float32>, Texture, Texture) {
        
        // our three buffers
        var stageArray = [Float32]()
        var classifierArray = [Float32]()
        var rectsArray = [UInt32]()
        
        // offsets for the features and classifiers
        var rectOffset : UInt32 = 0
        var classifierOffset : UInt32 = 0
        
        // iterate through all stages
        for stage in self.stages {
            
            // store the current stage along with the offset
            stageArray = stageArray + transformStage(stage: stage, classifierOffset: classifierOffset)
            
            // iterate through all classifiers
            for classifier in stage.classifiers {
                
                // store the current classifier
                classifierArray = classifierArray + transformClassifier(classifier: classifier, rectOffset: rectOffset)
                
                // iterate through all features
                for rect in classifier.rects {
                    
                    // store the feature
                    rectsArray = rectsArray + transformHaarFeatureRect(rect: rect)
                    
                    // increase the rect offset
                    rectOffset = rectOffset + 1
                }
                
                // increase the classifier offset
                classifierOffset = classifierOffset + 1
            }
        }
        
        // create the buffer for the stages
        let stageBuffer = Context.device().makeBuffer(bytes: &stageArray, length: stageArray.count * MemoryLayout<Float32>.size, options: .storageModeShared)
        stageBuffer.label = "StageBuffer"
        
        // create the texture for the classifiers
        let classifiersBuffer = Context.device().makeBuffer(bytes: &classifierArray, length: classifierArray.count * MemoryLayout<Float32>.size, options: .storageModeShared)
        let classifiersDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: Int(classifierOffset), height: 1, mipmapped: false)
        let classifiersTexture = classifiersBuffer.makeTexture(descriptor: classifiersDescriptor, offset: 0, bytesPerRow: classifierArray.count * MemoryLayout<Float32>.size)
        classifiersTexture.label = "ClassifiersTexture"
        
        // create the texture for the rectangles
        let rectBuffer = Context.device().makeBuffer(bytes: &rectsArray, length: rectsArray.count * MemoryLayout<UInt32>.size, options: .storageModeShared)
        let rectDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Uint, width: Int(rectOffset), height: 1, mipmapped: false)
        
        let rectsArrayBytesPerRow = rectsArray.count * MemoryLayout<UInt32>.size + 16
        
        let rectTex = rectBuffer.makeTexture(descriptor: rectDescriptor, offset: 0, bytesPerRow: (rectsArrayBytesPerRow+rectsArrayBytesPerRow%16))
        rectTex.label = "FeatureTexture"
        
        return (stage: Buffer(buffer: stageBuffer), classifiers: Texture(classifiersTexture), rects: Texture(rectTex))
    
    }
    
    
    
    
    
    
    

    
    
    
    
}

extension HaarCascade.HaarCascadeDescriptor : ValidMetalBufferContent {
    static func getSizeInBytes() -> Int {
        return MemoryLayout<HaarCascade.HaarCascadeDescriptor>.size
    }
}
