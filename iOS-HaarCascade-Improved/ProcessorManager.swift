//
//  ProcessorManager.swift
//
//  Created by Christopher Helf on 16.11.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation
import MetalKit

class ProcessorManager : ViewDelegate {
    
    var view : View? = nil
    
    // triple buffering
    private var _resources = [Processor]()
    private static let _maxBuffers = 3
    private let _semaphore = DispatchSemaphore(value: _maxBuffers)
    private var _currentIdx = 0
    
    init() {}
    
    // setup our processors
    func setup(view: View) {
        self.view = view
        for i in 0..<ProcessorManager._maxBuffers {
            _resources.append(Processor(view: view, id: i))
        }
    }
    
    // get the current processor depending on the index
    private func getProcessor() -> Processor {
        _currentIdx = (_currentIdx + 1) % ProcessorManager._maxBuffers
        return _resources[_currentIdx]
    }

    func process(drawable: CAMetalDrawable, descriptor: MTLRenderPassDescriptor, texture: MTLTexture, time: Double, frame: Int) {
        
        // wait for the next free slot
        guard _semaphore.wait(timeout: DispatchTime.distantFuture) == DispatchTimeoutResult.success else {
            fatalError("Semaphore never released")
        }
        
        // get our resource
        let processor = getProcessor()
        
        // update the fps
        self.view?.updateFps()
        
        // set the dimensions
        processor.setDimensions(width: texture.width, height: texture.height)
        
        // process
        //DispatchQueue.global(qos: .background).async {
            processor.process(drawable: drawable, descriptor: descriptor, texture: texture, time: time, semaphore: self._semaphore)
        //}
        

    }
    
}
