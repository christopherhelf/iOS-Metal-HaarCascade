//
//  MetalHelpers.swift
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 17.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation
import Metal

/// protocol we use as a wrapper for instructions to the pipeline
protocol MetalShader {
    var encodeViaCommandBuffer : Bool { get }
    func encode(_ encoder: MTLComputeCommandEncoder, previous: MetalComputeShader?)
    func encode(_ commandBuffer: MTLCommandBuffer)
    func setInputTextures(_ textures: [Texture])
}

/// A simple protocol for valid metal numbers
protocol ValidMetalBufferContent {
    static func getSizeInBytes() -> Int
}

extension Float32 : ValidMetalBufferContent {
    static func getSizeInBytes() -> Int {
        return MemoryLayout<Float32>.size
    }
}

extension Int32 : ValidMetalBufferContent {
    static func getSizeInBytes() -> Int {
        return MemoryLayout<Int32>.size
    }
}

extension UInt32 : ValidMetalBufferContent {
    static func getSizeInBytes() -> Int {
        return MemoryLayout<UInt32>.size
    }
}

/// A simple wrapper for a buffer containing a specific type
class Buffer<T> where T: ValidMetalBufferContent {
    
    let buffer : MTLBuffer
    let uuid = UUID().uuidString
    
    init(buffer: MTLBuffer) {
        self.buffer = buffer
    }
}

/// A simple struct containg a buffer and a pointer to the buffer data
class BufferWithPointer<T> : Buffer<T> where T: ValidMetalBufferContent {
    
    let pointer : UnsafeMutableBufferPointer<T>
    
    init(buffer: MTLBuffer, pointer: UnsafeMutableBufferPointer<T>) {
        self.pointer = pointer
        super.init(buffer: buffer)
    }
    
    /// Copies the contents of the buffer to an array
    func convert() -> [T] {
        var array = Array<T>()
        for i in 0..<self.buffer.length/T.getSizeInBytes() {
            array.append(pointer[i])
        }
        return array
    }
    
    // Sets a value to the pointer and thus the buffer
    func set(_ value: T, at: Int = 0) {
        self.pointer[at] = value
    }
}

/// Protocol for possible arguments for a compute command encoder
protocol ComputeCommandEncoderArgument {
    var uuid : String { get }
    func set(_ encoder: MTLComputeCommandEncoder, at: Int, inputs: [Texture])
}

/// A simple MTLTexture wrapper containg an ID so we can compare textures
/// at computecommandencoder positions
class Texture {
    
    let uuid = UUID().uuidString
    var texture : MTLTexture
    
    init(_ texture: MTLTexture) {
        self.texture = texture
    }
    
    init(_ descriptor: MTLTextureDescriptor, name: String) {
        self.texture = Context.device().makeTexture(descriptor: descriptor)
        self.texture.label = name
    }
}



// Texture protocol implementation
extension Texture : ComputeCommandEncoderArgument {
    func set(_ encoder: MTLComputeCommandEncoder, at: Int, inputs: [Texture]) {
        encoder.setTexture(self.texture, at: at)
    }
}

// Buffer protocol implementation
extension Buffer : ComputeCommandEncoderArgument {
    func set(_ encoder: MTLComputeCommandEncoder, at: Int, inputs: [Texture]) {
        encoder.setBuffer(self.buffer, offset: 0, at: at)
    }
}

// Int protocol implementation, sets bytes directly
extension Int : ComputeCommandEncoderArgument {
    
    internal var uuid: String {
        return String(self)
    }
    
    func set(_ encoder: MTLComputeCommandEncoder, at: Int, inputs: [Texture]) {
        var v : Int32 = Int32(self)
        encoder.setBytes(&v, length: MemoryLayout<Int32>.size, at: at)
    }
}

// Float protocol implementation, sets bytes directly
extension Float : ComputeCommandEncoderArgument {
    
    internal var uuid: String {
        return String(self)
    }
    
    func set(_ encoder: MTLComputeCommandEncoder, at: Int, inputs: [Texture]) {
        var v : Float32 = Float32(self)
        encoder.setBytes(&v, length: MemoryLayout<Float32>.size, at: at)
    }
}

// UInt32 protocol implementation, sets bytes directly
extension UInt32 : ComputeCommandEncoderArgument {
    
    internal var uuid: String {
        return String(self)
    }
    
    func set(_ encoder: MTLComputeCommandEncoder, at: Int, inputs: [Texture]) {
        var v : UInt32 = self
        encoder.setBytes(&v, length: MemoryLayout<UInt32>.size, at: at)
    }
}

// Int32 protocol implementation, sets bytes directly
extension Int32 : ComputeCommandEncoderArgument {
    
    internal var uuid: String {
        return String(self)
    }
    
    func set(_ encoder: MTLComputeCommandEncoder, at: Int, inputs: [Texture]) {
        var v : Int32 = self
        encoder.setBytes(&v, length: MemoryLayout<Int32>.size, at: at)
    }
}

// This is a possible input to a compute commander that references an input texture
class TextureInputReference : ComputeCommandEncoderArgument {
    
    let uuid = UUID().uuidString
    var reference : Int
    
    init(_ reference: Int) {
        self.reference = reference
    }
    
    func set(_ encoder: MTLComputeCommandEncoder, at: Int, inputs: [Texture]) {
        assert(inputs.count > self.reference, "Invalid Texture Reference for Inputs!")
        encoder.setTexture(inputs[self.reference].texture, at: at)
    }
    
}

extension MTLDevice {
    
    // simple wrapper that assigns the texture a name
    func makeTexture(_ descriptor: MTLTextureDescriptor, name: String) -> MTLTexture {
        let tex = self.makeTexture(descriptor: descriptor)
        tex.label = name
        return tex
    }
    
    func makeTexture(_ descriptor: MTLTextureDescriptor, name: String) -> Texture {
        let tex = self.makeTexture(descriptor: descriptor)
        tex.label = name
        return Texture(tex)
    }
}

/// Some Metal utilities
class MetalHelpers {
    
    class func createBufferWithPointer<T>(_ value: T, _ label: String? = nil) -> BufferWithPointer<T> {
        var value = value
        let buffer = Context.device().makeBuffer(bytes: &value, length: MemoryLayout<T>.size, options: MTLResourceOptions.storageModeShared)
        buffer.label = label
        let ptr = UnsafeMutablePointer<T>(OpaquePointer(buffer.contents()))
        let mutablePtr = UnsafeMutableBufferPointer<T>(start: ptr, count: 1)
        return BufferWithPointer(buffer: buffer, pointer: mutablePtr)
    }
    
    class func createBufferWithPointer<T>(_ value: [T], _ label: String? = nil) -> BufferWithPointer<T> {
        var value = value
        let buffer = Context.device().makeBuffer(bytes: &value, length: MemoryLayout<T>.size * value.count, options: MTLResourceOptions.storageModeShared)
        buffer.label = label
        let ptr = UnsafeMutablePointer<T>(OpaquePointer(buffer.contents()))
        let mutablePtr = UnsafeMutableBufferPointer<T>(start: ptr, count: value.count)
        return BufferWithPointer(buffer: buffer, pointer: mutablePtr)
    }
    
    class func createBuffer<T>(_ value: [T], _ label: String? = nil) -> Buffer<T> {
        var value = value
        let buffer = Context.device().makeBuffer(bytes: &value, length: value.count * MemoryLayout<T>.size, options: .storageModeShared)
        buffer.label = label
        return Buffer(buffer: buffer)
    }
    
    class func createBuffer<T>(_ count: Int, _ label: String? = nil) -> Buffer<T> {
        let buffer = Context.device().makeBuffer(length: count * MemoryLayout<T>.size, options: .storageModePrivate)
        buffer.label = label
        return Buffer(buffer: buffer)
    }
    
    class func createBuffer<T>(_ label: String? = nil) -> Buffer<T> {
        let buffer = Context.device().makeBuffer(length: MemoryLayout<T>.size, options: .storageModePrivate)
        buffer.label = label
        return Buffer(buffer: buffer)
    }
    
    class func createBuffer<T>(_ value: T,  _ label: String? = nil) -> Buffer<T> {
        var value = value
        let buffer = Context.device().makeBuffer(bytes: &value, length: MemoryLayout<T>.size, options: MTLResourceOptions.storageModeShared)
        buffer.label = label
        return Buffer(buffer: buffer)
    }
    
}

