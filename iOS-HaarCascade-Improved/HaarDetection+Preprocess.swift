//
//  HaarDetection+Preprocess.swift
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 07.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation
import Metal
import MetalPerformanceShaders

extension HaarDetection {
    
    struct PreprocessingData {
        var grayscale : Texture
        var integralImage : Texture
        var squareIntegralImage : Texture
        var descriptor : MPSImageDescriptor
    }
    
    struct PreprocessingShaders {
        let conversion : MPSImageConversion
        let integral : MPSImageIntegral
        let sqIntegral : MPSImageIntegralOfSquares
        let histogram : MPSImageHistogramEqualizer
        let scaling : MPSImageLanczosScale
    }
    
    class func initPreprocessingShaders() -> PreprocessingShaders {
        
        let conversionInfo = CGColorConversionInfo(src: CGColorSpaceCreateDeviceRGB(), dst: CGColorSpaceCreateDeviceGray())
        let conversion = MPSImageConversion(device: Context.device(), srcAlpha: .alphaIsOne, destAlpha: .alphaIsOne, backgroundColor: nil, conversionInfo: conversionInfo)
        conversion.label = "imageConversion"
        
        let integral = MPSImageIntegral(device: Context.device())
        integral.label = "imageIntegral"
        
        let sqIntegral = MPSImageIntegralOfSquares(device: Context.device())
        sqIntegral.label = "imageIntegralSquares"
        
        let histogram = MPSImageHistogramEqualizer(device: Context.device())
        
        let scaling = MPSImageLanczosScale(device: Context.device())
        scaling.label = "lanczosScale"
        
        return PreprocessingShaders(conversion: conversion, integral: integral, sqIntegral: sqIntegral,histogram: histogram, scaling: scaling)
        
    }
    
    class func initPreprocessingData(scaledWidth: Int, scaledHeight: Int, width: Int, height: Int) -> PreprocessingData {
        
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: MTLPixelFormat.r32Float,
            width: scaledWidth,
            height: scaledHeight,
            mipmapped: false)
        
        let imageDescriptor = MPSImageDescriptor(channelFormat: .float32, width: width, height: height, featureChannels: 1)
        
        let grayscaleTex = Texture(descriptor, name: "grayScale")
        let integralTex = Texture(descriptor, name: "imageIntegralTexture")
        let sqIntegralTex = Texture(descriptor, name: "imageIntegralSquaresTexture")
        
        return PreprocessingData(grayscale: grayscaleTex, integralImage: integralTex, squareIntegralImage: sqIntegralTex, descriptor: imageDescriptor)
    }
    
    func createPreprocessPipeline() -> MetalComputePipeline {
        
        let pColorEncode = MetalPerformanceShader(name: "PreprocessColorConversion", kernel: preprocessingShaders.conversion)
        let pScalingEncode = MetalPerformanceShader(name: "PreprocessScaling", kernel: preprocessingShaders.scaling)
        let pHistogramEqEncode = MetalPerformanceShader(name: "PreprocessHistogramEqualization", kernel: preprocessingShaders.histogram)
        let pImageIntegral = MetalPerformanceShader(name: "PreprocessImageIntegral", kernel: preprocessingShaders.integral)
        let pImageIntegralSquared = MetalPerformanceShader(name: "PreprocessImageIntegralSquared", kernel: preprocessingShaders.sqIntegral)
        
        let preprocessPipeline = MetalComputePipeline(name: "PreprocessPipeline")
        
        if self.initialScale != 1.0 {
            pColorEncode.addSourceReference(0) // input texture
            pColorEncode.addTargetReference(1) // intermediary texture
            preprocessPipeline.addShader(pColorEncode)
            
            pScalingEncode.addSourceReference(1)
            pScalingEncode.addTarget(preprocessingData.grayscale)
            preprocessPipeline.addShader(pScalingEncode)
            
            preprocessPipeline.addImageToPrefetch(preprocessingData.descriptor)
            
        } else {
            pColorEncode.addSourceReference(0)
            pColorEncode.addTarget(preprocessingData.grayscale)
            preprocessPipeline.addShader(pColorEncode)
        }
        
        if histogramEqualization {
            pHistogramEqEncode.addSource(preprocessingData.grayscale)
            preprocessPipeline.addShader(pHistogramEqEncode)
        }
        
        pImageIntegral.addSource(preprocessingData.grayscale)
        pImageIntegral.addTarget(preprocessingData.integralImage)
        
        pImageIntegralSquared.addSource(preprocessingData.grayscale)
        pImageIntegralSquared.addTarget(preprocessingData.squareIntegralImage)
        
        preprocessPipeline.addShader(pImageIntegral)
        preprocessPipeline.addShader(pImageIntegralSquared)
        
        return preprocessPipeline
    }

    
    func preprocess(texture: Texture, commandBuffer: MTLCommandBuffer) {
        
        var inputs = Array<Texture>()
        inputs.append(texture)
        
        if self.initialScale != 1.0 {
            let intermediate = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: preprocessingData.descriptor)
            intermediate.label = "Intermediate"
            intermediate.readCount = 0
            inputs.append(Texture(intermediate.texture))
        }
        
        preprocessingPipeline.setInputTextures(inputs)
        preprocessingPipeline.encode(commandBuffer)
    }
    
}
