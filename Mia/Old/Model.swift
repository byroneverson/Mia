//
//  Model.swift
//  Mia
//
//  Created by Byron Everson on 2/17/23.
//

import Foundation
import CoreML
import Accelerate

// MARK: Abstract/super class for a model, should be able to process an input producing an output
class Model {
    
    // MARK: Inference - override this
    // Debug gpt-2 shape (1, 1, 2, 50257) (batch_size, num_choices, sequence_length, config.vocab_size)
    func infer(_ features: [MLMultiArray]) -> MLMultiArray {
        // Debug
        let vocabSize = 50257
        let sequenceLength = 2
        let shape = [1, 1, NSNumber(value: sequenceLength), NSNumber(value: vocabSize)]
        let array = try! MLMultiArray(shape: shape, dataType: .float32)
        //let count = array.count
        // !
        for i in 0 ..< vocabSize {
            array[i] = (i == 0) ? 1.0 : 0.0
        }
        // [EOS]
        for i in 0 ..< vocabSize {
            array[vocabSize + i] = (i == 50256) ? 1.0 : 0.0
        }
        return array
    }
    
    // MARK: Text generation - higher level of inference, used in NLP
    func generate(_ input: String, _ outputCount: Int = 64, _ callback: ((String, Double) -> Bool)) -> String {
        return ""
    }
}
