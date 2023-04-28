//
//  Llama.swift
//  Mia
//
//  Created by Byron Everson on 4/15/23.
//

import Foundation 
import CLlama

private typealias _LlamaProgressCallback = (_ progress: Float, _ userData: UnsafeMutableRawPointer?) -> Void

public typealias LlamaProgressCallback = (_ progress: Float, _ llama: Llama) -> Void

public struct LlamaContextParams {
    public var context: Int32 = 512    // text context
    public var parts: Int32 = -1   // -1 for default
    public var seed: Int32 = 0      // RNG seed, 0 for random
    public var numberOfThreads: Int32 = 8 //4

    public var f16Kv = true         // use fp16 for KV cache
    public var logitsAll = false    // the llama_eval() call computes all logits, not just the last one
    public var vocabOnly = false    // only load the vocabulary, no weights
    public var useMlock = false     // force system to keep model in RAM
    public var embedding = false    // embedding mode only

    public static let `default` = LlamaContextParams()

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

public struct LlamaSampleParams {
    public var topK: Int32
    public var topP: Float
    public var temperature: Float
    public var repeatLastN: Int32
    public var repeatPenalty: Float
    public var batchSize: Int32

    public static let `default` = LlamaSampleParams(
        topK: 40,
        topP: 0.95,
        temperature: 0.8,
        repeatLastN: 64,
        repeatPenalty: 1.1,
        batchSize: 8
    )

    public init(topK: Int32 = 30,
                topP: Float = 0.95,
                temperature: Float = 0.7,
                repeatLastN: Int32 = 256,
                repeatPenalty: Float = 1.25,
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

public enum LlamaError: Error {
    case modelNotFound(String)
    case inputTooLong
    case failedToEval
}

public class Llama {
    private let context: OpaquePointer?
    private var contextParams: LlamaContextParams
    private var sampleParams: LlamaSampleParams = .default

    public init(path: String, contextParams: LlamaContextParams = .default) throws {
        self.contextParams = contextParams
        var params = llama_context_default_params()
        params.n_ctx = contextParams.context
        params.n_parts = contextParams.parts
        params.seed = contextParams.seed
        params.f16_kv = contextParams.f16Kv
        params.logits_all = contextParams.logitsAll
        params.vocab_only = contextParams.vocabOnly
        params.use_mlock = contextParams.useMlock
        params.embedding = contextParams.embedding
        // Check if model file exists
        if !FileManager.default.fileExists(atPath: path) {
            throw LlamaError.modelNotFound(path)
        }
        // Load model at path
        context = llama_init_from_file(path, params)
        // Print llama arch and cpu features info
        print(String(cString: llama_print_system_info()))
    }

    public func predict(_ input: String, _ count: Int = 128, _ callback: ((String, Double) -> Bool) ) throws -> String {
        // Sample parameters
        let params = sampleParams
        // Add a space in front of the first character to match OG llama tokenizer behavior
        let input = " " + input
        // Question answering prompt
        let prompt = "Question: " + input + "\n\nAnswer: "
        // Tokenize prompt
        var inputTokens = tokenize(prompt)
        let inputTokensCount = inputTokens.count
        print("Input tokens: \(inputTokens)")
        // How many tokens to generate - check if theres space in context for atleast one token (or batch size tokens?)
        var outputRemaining = min(count, Int(contextParams.context) - inputTokens.count)
        guard outputRemaining > 0 else {
            throw LlamaError.inputTooLong
        }
        // Past count - how many tokens have been fed so far
        var nPast = Int32(0)
        // Input
        var inputBatch = Array<llama_token>()
        // Inputs tokens, do not include inputs in output for conversational usage
        while inputTokens.count > 0 {
            // Clear input batch
            inputBatch.removeAll()
            // See how many to eval (up to batch size??? or can we feed the entire input)
            let evalCount = min(inputTokens.count, Int(params.batchSize))
            // Move tokens to batch
            inputBatch.append(contentsOf: inputTokens[0 ..< evalCount])
            inputTokens.removeFirst(evalCount)
            // Eval batch
            if llama_eval(context, inputBatch, Int32(inputBatch.count), nPast, contextParams.numberOfThreads) != 0 {
                throw LlamaError.failedToEval
            }
            // Increment past count
            nPast += Int32(evalCount)
            /*
            if lastNTokens.count > 0 {
                lastNTokens.removeFirst()
            }
            lastNTokens.append(inputs[consumed])
             */
        }
        // Output
        //var outputTokens = Array<llama_token>()
        var output = [String]()
        var outputRepeatTokens = Array<llama_token>()
        var outputTokensCount = 0
        // Loop until target count is reached
        while outputRemaining > 0 {
            // Pull a generation from context
            let outputToken = llama_sample_top_p_top_k(
                context,
                outputRepeatTokens,
                Int32(outputRepeatTokens.count),
                params.topK,
                params.topP,
                params.temperature,
                params.repeatPenalty
            )
            // Add output token to list (needed?)
            outputTokensCount += 1
            //outputTokens.append(outputToken)
            // Decrement remainder
            outputRemaining -= 1
            // Repeat tokens update
            outputRepeatTokens.append(outputToken)
            if outputRepeatTokens.count > params.repeatLastN {
                outputRepeatTokens.removeFirst()
            }
            // Check for eos - end early - check eos before bos in case they are the same
            if outputToken == llama_token_eos() {
                outputRemaining = 0
                print("🤖 [EOS]")
                break
            }
            // Check for bos - skip callback if so
            if outputToken == llama_token_bos() {
                print("🤖 [BOS]")
                continue
            }
            // Convert token to string and callback
            if let cStr = llama_token_to_str(context, outputToken) {
                let str = String(cString: cStr)
                // Append string to output
                output.append(str)
                // Per token callback
                let (output, time) = Utils.time {
                    return str
                }
                print("🤖 \(output) \(outputToken)") //" \(tokenProb)")
                if callback(output, time) {
                    // Early exit if requested by callback
                    print("💀 Early exit")
                    //generating = false
                    outputRemaining = 0
                    break
                }
            }
            // Check if we need to run another eval
            if outputRemaining > 0 {
                // Send generated token back into model for next generation
                if llama_eval(context, [outputToken], 1, nPast, contextParams.numberOfThreads) != 0 {
                    throw LlamaError.failedToEval
                }
                // Increment past count
                nPast += 1
            }
        }
        // Total tokens used
        print("Total tokens: \(inputTokensCount + outputTokensCount) (\(inputTokensCount) -> \(outputTokensCount))")
        // Return full string (for cases where callback is not used)
        return output.joined()
    }

    public func embeddings(_ input: String) throws -> [Float] {
        // Add a space in front of the first character to match OG llama tokenizer behavior
        let input = " " + input

        // tokenize the prompt
        let inputs = tokenize(input)

        guard inputs.count > 0 else {
            return []
        }

        if llama_eval(context, inputs, Int32(inputs.count), Int32(0), contextParams.numberOfThreads) != 0 {
            throw LlamaError.failedToEval
        }

        let embeddingsCount = Int(llama_n_embd(context))
        guard let embeddings = llama_get_embeddings(context) else {
            return []
        }
        return Array(UnsafeBufferPointer(start: embeddings, count: embeddingsCount))
    }

    deinit {
        llama_free(context)
    }

    // MARK: Internal

    func tokenize(_ input: String, bos: Bool = true, eos: Bool = false) -> [llama_token] {
        if input.count == 0 {
            return []
        }

        var embeddings: [llama_token] = Array<llama_token>(repeating: llama_token(), count: input.utf8.count)
        let n = llama_tokenize(context, input, &embeddings, Int32(input.utf8.count), bos)
        assert(n >= 0)
        embeddings.removeSubrange(Int(n)..<embeddings.count)
        
        if eos {
            embeddings.append(llama_token_eos())
        }
        
        return embeddings
    }
}

