//
//  HaarCascade+Read.swift
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 07.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation

extension HaarCascade {
    
    class func getRootNode(data: Data) throws -> [String : AnyObject] {
        
        guard let _obj = try? JSONSerialization.jsonObject(with: data, options: JSONSerialization.ReadingOptions.mutableContainers), let jsonObject = _obj as? [String:AnyObject] else {
            throw Errors.invalidSyntax
        }
        
        guard let jsonRoot = jsonObject["opencv_storage"] as? [String : AnyObject], jsonRoot.count == 1 else {
            throw Errors.invalidSyntax
        }
        
        guard let cascadeRoot = jsonRoot.first?.value as? [String : AnyObject] else {
            throw Errors.invalidSyntax
        }
        
        return cascadeRoot
    }
    
    class func getStages(root: [String : AnyObject]) throws -> [AnyObject]{
        
        guard let stages = (root["stages"] as? [String:AnyObject])?["_"] as? [AnyObject] else {
            throw Errors.invalidSyntax
        }
        
        return stages
    }
    
    class func getWindowSize(root: [String : AnyObject]) throws -> (UInt32, UInt32) {
        guard
            let sizeStr = HaarCascade.parseIntArrayFromString(root, "size"),
            sizeStr.count == 2
            else {
                throw Errors.invalidSyntax
        }
        return(UInt32(sizeStr[0]), UInt32(sizeStr[1]))
    }
    
    class func getHaarStageClassifier(_ _stage: AnyObject) throws -> HaarStage {
        
        guard let stage = _stage as? [String:AnyObject] else {
            throw Errors.invalidSyntax
        }
        
        guard let threshold = HaarCascade.parseAsFloat(stage, "stage_threshold") else {
            throw Errors.invalidSyntax
        }
        
        guard var classifiers = (stage["trees"] as? [String:AnyObject])?["_"] as? [[String: AnyObject]] else {
            throw Errors.invalidSyntax
        }
        
        classifiers = try classifiers.map({ (v) -> [String: AnyObject] in
            guard let _v = v["_"] as? [String: AnyObject] else {
                if let _ = v["_"] as? [AnyObject] {
                    throw Errors.classifierMoreThanOneHaarFeature
                } else {
                    throw Errors.invalidSyntax
                }
            }
            return _v
        })
        
        let classifiersArray = try classifiers.map { (v) -> HaarClassifier in
            return try HaarCascade.getHaarClassifier(v)
        }
        
        return HaarStage(classifiers: classifiersArray, stageThreshold: threshold)
        
    }
    
    class func getHaarClassifier(_ classifier: [String:AnyObject]) throws -> HaarClassifier {
        
        guard let classifierThreshold = HaarCascade.parseAsFloat(classifier, "threshold") else {
            throw Errors.invalidSyntax
        }
        
        guard let left = HaarCascade.parseAsFloat(classifier, "left_val") else {
            throw Errors.invalidSyntax
        }
        
        guard let right = HaarCascade.parseAsFloat(classifier, "right_val") else {
            throw Errors.invalidSyntax
        }
        
        let features = try HaarCascade.getHaarFeatures(classifier)
        
        return HaarClassifier(rects: features, threshold: classifierThreshold, left: left, right: right)
    }
    
    class func getHaarFeatures(_ classifier : [String:AnyObject]) throws -> [HaarCascade.HaarFeatureRect] {
        
        guard let rects = (((classifier["feature"] as? [String:AnyObject])?["rects"]) as? [String:AnyObject])?["_"] as? [String] else {
            throw Errors.invalidSyntax
        }
        
        let rectsMapped = try rects.map({ (s) -> HaarFeatureRect in
            guard let vals = HaarCascade.parseSeparatedStringAsFloats(s: s), vals.count == 5 else {
                throw Errors.invalidSyntax
            }
            return HaarFeatureRect(x: UInt8(vals[0]), y: UInt8(vals[1]), width: UInt8(vals[2]), height: UInt8(vals[3]), weight: vals[4])
        })
        
        guard rectsMapped.count >= 2 else {
            throw Errors.invalidSyntax
        }
        
        return rectsMapped
    }
    
    class func parseAsFloat(_ o: [String: AnyObject], _ k: String) -> Float32? {
        if let _v = o[k] as? String {
            if let __v = Float(_v) {
                return Float32(__v)
            } else {
                return nil
            }
        } else if let _v = o[k] as? Double {
            return Float32(Float(_v))
        } else {
            return nil
        }
    }
    
    class func parseAsFloat(_ _o: AnyObject, _ k: String) -> Float32? {
        if let o = _o as? [String:AnyObject] {
            return HaarCascade.parseAsFloat(o, k)
        } else {
            return nil
        }
    }
    
    class func parseSeparatedStringAsFloats(s: String) -> [Float32]? {
        let v = s.components(separatedBy: " ")
        guard v.count > 0 else { return nil }
        return try? v.map { (s) -> Float32 in
            guard let ii = Float(s) else { throw Errors.parsingError }
            return Float32(ii)
        }
    }
    
    class func parseSeparatedStringAsInts(s: String) -> [Int32]? {
        let v = s.components(separatedBy: " ")
        guard v.count > 0 else { return nil }
        return try? v.map { (s) -> Int32 in
            guard let ii = Int(s) else { throw Errors.parsingError }
            return Int32(ii)
        }
    }
    
    class func parseIntArrayFromString(_ o: [String: AnyObject], _ k: String) -> [Int32]? {
        guard let _v = o[k] as? String else { return nil }
        return parseSeparatedStringAsInts(s: _v)
    }
    
    class func parseFloatArrayFromString(_ o: [String: AnyObject], _ k: String) -> [Float32]? {
        if let _v = o[k] as? String {
            return parseSeparatedStringAsFloats(s: _v)
        } else {
            return nil
        }
    }
}
