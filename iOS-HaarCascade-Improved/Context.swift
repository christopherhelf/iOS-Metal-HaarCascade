//
//  Context.swift
//
//  Created by Christopher Helf on 03.11.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation
import Metal
import MetalKit

class Context {
    
    var kWarpSize : UInt32 = 32
    var kLog2WarpSize : UInt32  = 5
    var numThreadsAnchorsParallel : UInt32  = 64
    var log2NumThreadsAnchorsParallel : UInt32 = 6
    var functionConstantValues = MTLFunctionConstantValues()
    var minNeighbors : Int32 = 1
    var eps : Float32 = 0.4
    
    static let sharedInstance = Context()
    private var _device : MTLDevice
    private var _library : MTLLibrary
    private var _commandQueue : MTLCommandQueue
    private var _context : CIContext
    
    init() {
        
        functionConstantValues.setConstantValue(&kWarpSize, type: MTLDataType.uint, at: 0)
        functionConstantValues.setConstantValue(&kLog2WarpSize, type: MTLDataType.uint, at: 1)
        functionConstantValues.setConstantValue(&numThreadsAnchorsParallel, type: MTLDataType.uint, at: 2)
        functionConstantValues.setConstantValue(&log2NumThreadsAnchorsParallel, type: MTLDataType.uint, at: 3)
        functionConstantValues.setConstantValue(&eps, type: MTLDataType.float, at: 4)
        functionConstantValues.setConstantValue(&minNeighbors, type: MTLDataType.int, at: 5)
        
        _device = MTLCreateSystemDefaultDevice()!;
        _library = _device.newDefaultLibrary()!;
        _commandQueue = _device.makeCommandQueue()
        _context = CIContext(mtlDevice: _device);
    }
    
    class func device() -> MTLDevice {
        return Context.sharedInstance._device
    }
    
    class func library() -> MTLLibrary {
        return Context.sharedInstance._library
    }
    
    class func commandQueue() -> MTLCommandQueue {
        return Context.sharedInstance._commandQueue
    }
    
    class func context() -> CIContext {
        return Context.sharedInstance._context
    }
    
    class func makeFunction(name: String, withConstants: Bool = true) -> MTLFunction {
        if withConstants {
            return try! Context.library().makeFunction(name: name, constantValues: Context.sharedInstance.functionConstantValues)
        } else {
            return Context.library().makeFunction(name: name)!
        }
    }
    
    class func makeComputePipeline(name: String, withConstants: Bool = true) -> MTLComputePipelineState {
        let function = Context.makeFunction(name: name, withConstants: withConstants)
        return try! Context.device().makeComputePipelineState(function: function)
    }
    
    
}

