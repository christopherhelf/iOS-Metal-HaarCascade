//
//  View.swift
//
//  Created by Christopher Helf on 10.11.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation
import MetalKit

protocol ViewDelegate {
    func process(drawable: CAMetalDrawable, descriptor: MTLRenderPassDescriptor, texture: MTLTexture, time: Double, frame: Int)
}

class View : MTKView, ProcessorInputDelegate {
    
    struct Vertex{
        var x,y,z: Float
        var r,g,b,a: Float
        var s,t: Float
        
        func floatBuffer() -> [Float]{
            return [x,y,z,r,g,b,a,s,t]
        }
    };
    
    private var samplerState : MTLSamplerState!
    private var vertexBuffer: MTLBuffer!
    private var vertexCount: Int!
    private var pipeline : MTLRenderPipelineState!
    var lastTime : CFTimeInterval?
    var fpsView : UILabel! = nil
    var viewDelegate : ViewDelegate
    var rgbTexture : MTLTexture? = nil
    var rgbTime : Double? = nil
    var rgbFrameCount : Int? = nil
    var frameCount : Int = 0
    
    init(frame: CGRect, viewDelegate: ViewDelegate) {
        self.viewDelegate = viewDelegate
        super.init(frame: frame, device: Context.device())
        fpsView = UILabel(frame: CGRect(x: 20, y: 20, width: frame.width-20, height: 20))
        fpsView.textAlignment = .left
        fpsView.backgroundColor = UIColor.clear
        self.isPaused = true
        self.enableSetNeedsDisplay = false
        self.addSubview(fpsView)
        self.framebufferOnly = false
        self.clearColor = MTLClearColorMake(1, 1, 1, 1.0)
        self.samplerState = self.generateSamplerStateForTexture(Context.device())
        self.setupVertexBuffer()
        self.setupPipeline()
    }
    
    required init(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func getDrawable() -> CAMetalDrawable? {
        return self.currentDrawable
    }
    
    private func getDescriptor() -> MTLRenderPassDescriptor? {
        return self.currentRenderPassDescriptor
    }
    
    override func draw(_ dirtyRect: CGRect)
    {
        guard
            let drawable = self.getDrawable(),
            let texture = self.rgbTexture,
            let time = self.rgbTime,
            let descriptor = self.getDescriptor(),
            let count = self.rgbFrameCount else { return }
        viewDelegate.process(drawable: drawable, descriptor: descriptor, texture: texture, time: time, frame: count)
    }
    
    func gotTexture(texture: MTLTexture, time: Double, frame: Int) {
        self.rgbTexture = texture
        self.rgbTime = time
        self.rgbFrameCount = frame
        self.draw()
    }
    
    func encode(drawable: CAMetalDrawable, descriptor: MTLRenderPassDescriptor, commandBuffer: MTLCommandBuffer, input: MTLTexture) {
        
        // get the encoder
        let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor)
        
        // render encoding
        encoder.pushDebugGroup("render")
        encoder.setRenderPipelineState(pipeline)
        encoder.setVertexBuffer(vertexBuffer, offset: 0, at: 0)
        encoder.setFragmentTexture(input, at: 0)
        encoder.setFragmentSamplerState(samplerState, at: 0)
        encoder.setCullMode(MTLCullMode.none)
        
        // Draw primitives
        encoder.drawPrimitives(
            type: .triangle,
            vertexStart: 0,
            vertexCount: vertexCount,
            instanceCount: vertexCount/2
        )
        
        encoder.popDebugGroup()
        encoder.endEncoding()
        
    }
    
    private func generateSamplerStateForTexture(_ device: MTLDevice) -> MTLSamplerState {
        
        let pSamplerDescriptor:MTLSamplerDescriptor? = MTLSamplerDescriptor();
        
        if let sampler = pSamplerDescriptor
        {
            sampler.minFilter             = MTLSamplerMinMagFilter.linear
            sampler.magFilter             = MTLSamplerMinMagFilter.linear
            sampler.mipFilter             = MTLSamplerMipFilter.linear
            sampler.maxAnisotropy         = 1
            sampler.sAddressMode          = MTLSamplerAddressMode.clampToEdge
            sampler.tAddressMode          = MTLSamplerAddressMode.clampToEdge
            sampler.rAddressMode          = MTLSamplerAddressMode.clampToEdge
            sampler.normalizedCoordinates = true
            sampler.lodMinClamp           = 0
            sampler.lodMaxClamp           = FLT_MAX
        }
        else
        {
            fatalError()
        }
        
        return device.makeSamplerState(descriptor: pSamplerDescriptor!)
    }
    
    private func setupVertexBuffer() {
        
        let A = Vertex(x: -1.0, y: -1.0, z:   0.0, r:  1.0, g:  0.0, b:  0.0, a:  1.0, s: 1.0, t: 1.0)
        let B = Vertex(x: -1.0, y:  1.0, z:   0.0, r:  0.0, g:  1.0, b:  0.0, a:  1.0, s: 1.0, t: 0.0)
        let C = Vertex(x:  1.0, y:  1.0, z:   0.0, r:  0.0, g:  0.0, b:  1.0, a:  1.0, s: 0.0, t: 0.0)
        let D = Vertex(x:  1.0, y: -1.0, z:   0.0, r:  0.1, g:  0.6, b:  0.4, a:  1.0, s: 0.0, t: 1.0)
        
        let vertices:Array<Vertex> = [
            A,B,C ,A,C,D   //Front
        ]
        
        var vertexData = Array<Float>()
        for vertex in vertices
        {
            vertexData += vertex.floatBuffer()
        }
        
        let dataSize = vertexData.count * MemoryLayout.size(ofValue: vertexData[0])
        vertexCount = vertices.count
        vertexBuffer = Context.device().makeBuffer(bytes: vertexData, length: dataSize, options: MTLResourceOptions())
        
    }
    
    private func setupPipeline() {
        
        // Setup pipeline
        let desc = MTLRenderPipelineDescriptor()
        
        let basicVert = Context.library().makeFunction(name: "basic_vertex")
        let copyFrag = Context.library().makeFunction(name: "copy")
        
        desc.label = "copy"
        desc.vertexFunction = basicVert
        desc.fragmentFunction = copyFrag
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipeline = try! Context.device().makeRenderPipelineState(descriptor: desc)
        
    }
    
    func updateFps() {
        
        DispatchQueue.main.async {
            
            let time = CFAbsoluteTimeGetCurrent();
            guard let lastTime = self.lastTime else {
                self.lastTime = time
                return
            }
            
            if (self.frameCount >= 25) {
                self.fpsView.text = "fps: \(Int(Double(self.frameCount)/(time-lastTime)))"
                self.lastTime = time
                self.frameCount = 0
                return
            } else {
                self.frameCount+=1
            }
            
        }
        
    }
    
}
