//
//  HaarCascade+Errors.swift
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 07.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation

extension HaarCascade {
    
    /// simple enum of possible errors when reading in the json file
    enum Errors : Error {
        case invalidSyntax
        case parsingError
        case classifierMoreThanOneHaarFeature
    }
}
