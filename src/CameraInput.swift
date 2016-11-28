//
//  Camera.swift
//
//  Created by Christopher Helf on 03.11.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation
import UIKit
import AVFoundation
import CoreGraphics
import ImageIO

class CameraInput : ProcessorInput, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    private let targetPosition : AVCaptureDevicePosition = .front
    private let targetFps : Float64 = 30
    static let targetDimensions = CMVideoDimensions(width: 1280, height: 720)
    
    private let targetType = NSNumber(value: kCVPixelFormatType_32BGRA)
    
    private let orientation = AVCaptureVideoOrientation.portrait;
    private var session : AVCaptureSession! = nil
    private let queue : DispatchQueue = DispatchQueue(label: "VideoDataOutputQueue")
    private var cache : CVMetalTextureCache?
    
    override init(delegate: ProcessorInputDelegate) {
        super.init(delegate: delegate)
        self.initCache()
        self.initCamera()
    }
    
    override func getDimensions() -> (width: Int, height: Int) {
        // rotate them
        return (Int(CameraInput.targetDimensions.height), Int(CameraInput.targetDimensions.width))
    }
    
    override func start() {
        session.startRunning()
    }
    
    override func stop() {
        session.stopRunning()
    }
    
    func captureOutput(_ captureOutput: AVCaptureOutput!, didOutputSampleBuffer sampleBuffer: CMSampleBuffer!, from connection: AVCaptureConnection!) {
        
        if connection.videoOrientation != orientation {
            connection.videoOrientation = orientation
            return
        }
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            fatalError("Could not retreive pixelbuffer")
        }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        var outTexture: CVMetalTexture? = nil
        
        guard CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache!, pixelBuffer, nil, .bgra8Unorm, width, height, 0, &outTexture) != kCVReturnError else {
            fatalError("Could not create texture cache")
        }
        
        guard let videoTexture = CVMetalTextureGetTexture(outTexture!) else {
            fatalError("Could not retreive video texture")
        }
        
        videoTexture.label = "bgraTexture"
        
        let time = Date().timeIntervalSince1970
        self.frameCount += 1
        delegate.gotTexture(texture: videoTexture, time: time, frame: self.frameCount)
    }
    
    func captureOutput(_ captureOutput: AVCaptureOutput!, didDrop sampleBuffer: CMSampleBuffer!, from connection: AVCaptureConnection!) {
        delegate.droppedFrame?()
    }
    
    private func initCache() {
        guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, Context.device(), nil, &cache) == kCVReturnSuccess else {
            fatalError("Could not create texture cache")
        }
    }
    
    private func initCamera() {
        
        session = AVCaptureSession()
        session.beginConfiguration()
        session.sessionPreset = AVCaptureSessionPresetHigh
        
        guard let device = AVCaptureDevice.defaultDevice(withDeviceType: .builtInWideAngleCamera, mediaType: AVMediaTypeVideo, position: targetPosition) else {
            fatalError("Could not retreive valid device")
        }
        
        configureCamera(device: device)
        
        session.commitConfiguration()
        
    }
    
    private func configureCamera(device: AVCaptureDevice) {
        
        var chosenFormat : AVCaptureDeviceFormat? = nil
        var chosenRange : AVFrameRateRange? = nil
        
        for _format in device.formats {
            
            guard let format = _format as? AVCaptureDeviceFormat else { continue }
            guard let description = format.formatDescription else { continue }
            let dimensions = CMVideoFormatDescriptionGetDimensions(description)
            guard dimensions.width == CameraInput.targetDimensions.width && dimensions.height == CameraInput.targetDimensions.height else { continue }
            guard CMFormatDescriptionGetMediaSubType(description)==kCVPixelFormatType_420YpCbCr8BiPlanarFullRange else { continue }
            
            for _range in format.videoSupportedFrameRateRanges {
                guard let range = _range as? AVFrameRateRange else { continue }
                
                if range.maxFrameRate >= targetFps {
                    chosenFormat = format
                    chosenRange = range
                    break
                }
            }
        }
        
        guard chosenFormat != nil && chosenRange != nil else {
            fatalError("Could not determine device format and frame rate")
        }
        
        guard let input = try? AVCaptureDeviceInput(device: device) else {
            fatalError("Could not find video input device")
        }
        
        guard session.canAddInput(input) else {
            fatalError("Could not add input to session")
        }
        
        session.addInput(input)
        
        do {
            try device.lockForConfiguration()
            device.activeFormat = chosenFormat
            device.activeVideoMinFrameDuration = CMTimeMake(1, Int32(targetFps))
            device.activeVideoMaxFrameDuration = CMTimeMake(1, Int32(targetFps))
            device.unlockForConfiguration()
        } catch {
            fatalError("Could not lock device for configuring format and frame rate")
        }
        
        let output = AVCaptureVideoDataOutput()
        let targetType = NSNumber(value: kCVPixelFormatType_32BGRA)
        
        guard (output.availableVideoCVPixelFormatTypes.filter { (format) -> Bool in
            guard let _format = format as? NSNumber else { return false }
            if _format == targetType {
                return true
            } else {
                return false
            }
        }).count > 0 else {
            fatalError("AVCaptureVideoDataOutput does not support 32bgra")
        }
        
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: targetType]
        
        guard session.canAddOutput(output) else {
            fatalError("Could not add Output to Session")
        }
        
        session.addOutput(output)
        
        output.setSampleBufferDelegate(self, queue: queue)
        
    }
    
}

