//
//  Context.swift
//
//  Created by Christopher Helf on 03.11.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation
import Metal
import MetalKit

// simple wrapper class around mtldevice
class Context {
    
    static let sharedInstance = Context()
    private var _device : MTLDevice
    private var _library : MTLLibrary
    private var _commandQueue : MTLCommandQueue
    private var _context : CIContext
    
    init() {
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
}
