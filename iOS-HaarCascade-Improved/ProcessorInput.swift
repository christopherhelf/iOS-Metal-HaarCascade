//
//  ProcessorInput.swift
//
//  Created by Christopher Helf on 11.11.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation
import MetalKit

@objc protocol ProcessorInputDelegate {
    func gotTexture(texture: MTLTexture, time: Double, frame: Int)
    @objc optional func droppedFrame()
}

class ProcessorInput : NSObject {
    
    let delegate : ProcessorInputDelegate
    static let notificationName = Notification.Name("processingFinishedNotification")
    var frameCount = 0
    
    
    init(delegate: ProcessorInputDelegate) {
        self.delegate = delegate
        super.init()
        NotificationCenter.default.addObserver(self, selector: #selector(ProcessorInput.processingFinished), name: ProcessorInput.notificationName, object: nil)
    }
    
    func getDimensions() -> (width: Int, height: Int) {
        fatalError()
    }
    
    func start() {}
    
    func stop() {}
    
    class func sendprocessingFinishedNotification() {
        NotificationCenter.default.post(name: notificationName, object: nil)
    }
    
    func processingFinished() {}
    
}
