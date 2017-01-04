//
//  ViewController.swift
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 05.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
    
    var metalView : View?
    var processorManager : ProcessorManager?
    var processor : Processor?
    var camera : CameraInput?

    override func viewDidLoad() {
        super.viewDidLoad()
        
        processorManager = ProcessorManager()
        
        let metalView = View(frame: self.view.frame, viewDelegate: processorManager!)
        self.metalView = metalView
        
        processorManager?.setup(view: metalView)
        
        camera = CameraInput(delegate: metalView)
        self.view.addSubview(metalView)
        camera!.start()
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}

