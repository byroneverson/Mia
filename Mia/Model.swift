//
//  Model.swift
//  Mia
//
//  Created by Byron Everson on 4/28/23.
//

import Foundation
import CGPTNeoX

private typealias _ModelProgressCallback = (_ progress: Float, _ userData: UnsafeMutableRawPointer?) -> Void

public typealias ModelProgressCallback = (_ progress: Float, _ model: Model) -> Void

public struct ModelContextParams {
    public var context: Int32 = 512    // text context
    public var parts: Int32 = -1   // -1 for default
    public var seed: Int32 = 0      // RNG seed, 0 for random
    public var numberOfThreads: Int32 = 1

    public var f16Kv = true         // use fp16 for KV cache
    public var logitsAll = false    // the llama_eval() call computes all logits, not just the last one
    public var vocabOnly = false    // only load the vocabulary, no weights
    public var useMlock = false     // force system to keep model in RAM
    public var embedding = false    // embedding mode only

    public static let `default` = ModelContextParams()

    public init(context: Int32 = 2048 /*512*/, parts: Int32 = -1, seed: Int32 = 0, numberOfThreads: Int32 = 0, f16Kv: Bool = true, logitsAll: Bool = false, vocabOnly: Bool = false, useMlock: Bool = false, embedding: Bool = false) {
        self.context = context
        self.parts = parts
        self.seed = seed
        // Set numberOfThreads to processorCount, processorCount is actually thread count of cpu
        self.numberOfThreads = numberOfThreads == 0 ? Int32(ProcessInfo.processInfo.processorCount) : numberOfThreads
        self.f16Kv = f16Kv
        self.logitsAll = logitsAll
        self.vocabOnly = vocabOnly
        self.useMlock = useMlock
        self.embedding = embedding
    }
}

public struct ModelSampleParams {
    public var n_batch: Int32
    public var temp: Float
    public var top_k: Int32
    public var top_p: Float
    public var tfs_z: Float
    public var typical_p: Float
    public var repeat_penalty: Float
    public var repeat_last_n: Int32
    public var frequence_penalty: Float
    public var presence_penalty: Float
    public var mirostat: Int32
    public var mirostat_tau: Float
    public var mirostat_eta: Float
    public var penalize_nl: Bool
    
    public static let `default` = ModelSampleParams(
        n_batch: 512,
        temp: 0.8,
        top_k: 40,
        top_p: 0.95,
        tfs_z: 1.0,
        typical_p: 1.0,
        repeat_penalty: 1.1,
        repeat_last_n: 64,
        frequence_penalty: 0.0,
        presence_penalty: 0.0,
        mirostat: 0,
        mirostat_tau: 5.0,
        mirostat_eta: 0.1,
        penalize_nl: true
    )

    public init(n_batch: Int32 = 512,
                temp: Float = 0.8,
                top_k: Int32 = 40,
                top_p: Float = 0.95,
                tfs_z: Float = 1.0,
                typical_p: Float = 1.0,
                repeat_penalty: Float = 1.1,
                repeat_last_n: Int32 = 64,
                frequence_penalty: Float = 0.0,
                presence_penalty: Float = 0.0,
                mirostat: Int32 = 0,
                mirostat_tau: Float = 5.0,
                mirostat_eta: Float = 0.1,
                penalize_nl: Bool = true) {
        self.n_batch = n_batch
        self.temp = temp
        self.top_k = top_k
        self.top_p = top_p
        self.tfs_z = tfs_z
        self.typical_p = typical_p
        self.repeat_penalty = repeat_penalty
        self.repeat_last_n = repeat_last_n
        self.frequence_penalty = frequence_penalty
        self.presence_penalty = presence_penalty
        self.mirostat = mirostat
        self.mirostat_tau = mirostat_tau
        self.mirostat_eta = mirostat_eta
        self.penalize_nl = penalize_nl
    }
}

public enum ModelError: Error {
    case modelNotFound(String)
    case inputTooLong
    case failedToEval
}

public enum ModelPromptStyle {
    case None
    case ChatBase
    case OpenAssistant
    case StableLM_Tuned
    case LLaMa
    case LLaMa_QA
}

public typealias ModelToken = Int32

public class Model {
    
    var context: OpaquePointer?
    var contextParams: ModelContextParams
    var sampleParams: ModelSampleParams = .default
    var promptStyle: ModelPromptStyle = .None
    
    // Init
    public init(path: String = "", contextParams: ModelContextParams = .default) throws {
        self.contextParams = contextParams
        self.context = nil
    }
    
    // Predict
    public func predict(_ input: String, _ callback: ((String, Double) -> Bool) ) throws -> String {
        return ""
    }
    
    public func tokenize(_ input: String, bos: Bool = false, eos: Bool = false) -> [ModelToken] {
        return []
    }
    
    public func tokenizePrompt(_ input: String, _ style: ModelPromptStyle) -> [ModelToken] {
        switch style {
        case .None:
            return tokenize(input)
        case .ChatBase:
            return tokenize("<human>: " + input + "\n<bot>:")
        case .OpenAssistant:
            var inputTokens = tokenize(input)
            inputTokens.insert(gptneox_str_to_token(context, "<|prompter|>"), at: 0)
            inputTokens.append(gptneox_str_to_token(context, "<|endoftext|>"))
            inputTokens.append(gptneox_str_to_token(context, "<|assistant|>"))
            return inputTokens
        case .StableLM_Tuned:
            var inputTokens = tokenize(input)
            //let systemTokens = tokenize("# StableLM Tuned (Alpha version)\n- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.\n- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.\n- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.\n- StableLM will refuse to participate in anything that could harm a human.")
            inputTokens.insert(gptneox_str_to_token(context, "<|USER|>"), at: 0)
            //inputTokens.insert(contentsOf: systemTokens, at: 0)
            //inputTokens.insert(gptneox_str_to_token(context, "<|SYSTEM|>"), at: 0)
            inputTokens.append(gptneox_str_to_token(context, "<|ASSISTANT|>"))
            return inputTokens
        case .LLaMa:
            let bos = llama_token_bos()
            // Add a space in front of the first character to match OG llama tokenizer behavior
            let input = " " + input
            // Tokenize
            return tokenize(input, bos: true)
        case .LLaMa_QA:
            // Add a space in front of the first character to match OG llama tokenizer behavior
            // Question answering prompt
            // Is the space in front of Question needed for Stack LLaMa?
            let input = " Question: " + input + "\n\nAnswer: "
            // Tokenize
            return tokenize(input, bos: true)
        }
    }
    
}
