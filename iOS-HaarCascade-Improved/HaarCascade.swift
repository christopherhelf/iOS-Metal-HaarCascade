//
//  HaarCascade.swift
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 07.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation

class HaarCascade {

    var windowWidth: UInt32
    var windowHeight: UInt32
    var stages : [HaarStage]
    
    init(data: Data) throws {
        let root = try HaarCascade.getRootNode(data: data)
        (windowWidth, windowHeight) = try HaarCascade.getWindowSize(root: root)
        stages = try HaarCascade.parse(root: root)
    }
    
    class func parse(root: [String: AnyObject]) throws -> [HaarStage] {
        
        var stages = [HaarStage]()
        
        let jsonStages = try HaarCascade.getStages(root: root)
        
        for stage in jsonStages {
            stages.append(try HaarCascade.getHaarStageClassifier(stage))
        }
        
        return stages
    }
    
}
