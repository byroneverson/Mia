//
//  Model.swift
//  Mia
//
//  Created by Byron Everson on 4/28/23.
//

import Foundation

private typealias _ModelProgressCallback = (_ progress: Float, _ userData: UnsafeMutableRawPointer?) -> Void

public typealias ModelProgressCallback = (_ progress: Float, _ model: Model) -> Void

public struct ModelContextParams {
    public var context: Int32 = 512    // text context
    public var parts: Int32 = -1   // -1 for default
    public var seed: Int32 = 0      // RNG seed, 0 for random
    public var numberOfThreads: Int32 = 8 //4

    public var f16Kv = true         // use fp16 for KV cache
    public var logitsAll = false    // the llama_eval() call computes all logits, not just the last one
    public var vocabOnly = false    // only load the vocabulary, no weights
    public var useMlock = false     // force system to keep model in RAM
    public var embedding = false    // embedding mode only

    public static let `default` = ModelContextParams()

    public init(context: Int32 = 2048 /*512*/, parts: Int32 = -1, seed: Int32 = 0, numberOfThreads: Int32 = 8 /*4*/, f16Kv: Bool = true, logitsAll: Bool = false, vocabOnly: Bool = false, useMlock: Bool = false, embedding: Bool = false) {
        self.context = context
        self.parts = parts
        self.seed = seed
        // Set numberOfThreads to processorCount, processorCount is actually thread count of cpu
        self.numberOfThreads = Int32(ProcessInfo.processInfo.processorCount) // numberOFThread
        self.f16Kv = f16Kv
        self.logitsAll = logitsAll
        self.vocabOnly = vocabOnly
        self.useMlock = useMlock
        self.embedding = embedding
    }
}

public struct ModelSampleParams {
    public var topK: Int32
    public var topP: Float
    public var temperature: Float
    public var repeatLastN: Int32
    public var repeatPenalty: Float
    public var batchSize: Int32

    public static let `default` = ModelSampleParams(
        topK: 40,
        topP: 0.95,
        temperature: 0.8,
        repeatLastN: 64,
        repeatPenalty: 1.1,
        batchSize: 8
    )

    public init(topK: Int32 = 30,
                topP: Float = 0.95,
                temperature: Float = 0.8,
                repeatLastN: Int32 = 128,
                repeatPenalty: Float = 1.2,
                batchSize: Int32 = 8) {
        self.topK = topK
        self.topP = topP
        self.temperature = temperature
        self.repeatLastN = repeatLastN
        self.repeatPenalty = repeatPenalty
        // Set batchSize to processorCount, processorCount is actually thread count of cpu
        self.batchSize = Int32(ProcessInfo.processInfo.processorCount) //batchSize
    }
}

public enum ModelError: Error {
    case modelNotFound(String)
    case inputTooLong
    case failedToEval
}


public class Model {
    
    var context: OpaquePointer?
    var contextParams: ModelContextParams
    var sampleParams: ModelSampleParams = .default
    
    // Init
    public init(path: String = "", contextParams: ModelContextParams = .default) throws {
        self.contextParams = contextParams
        self.context = nil
    }
    
    // Predict
    public func predict(_ input: String, _ count: Int = 128, _ callback: ((String, Double) -> Bool) ) throws -> String {
        return ""
    }
    
}
