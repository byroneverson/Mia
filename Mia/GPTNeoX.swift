//
//  GPTNeoX.swift
//  Mia
//
//  Created by Byron Everson on 4/19/23.
//

import Foundation
import CGPTNeoX

public class GPTNeoX: Model {

    public override init(path: String, contextParams: ModelContextParams = .default) throws {
        try super.init()
        
        self.contextParams = contextParams
        var params = gptneox_context_default_params()
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
            throw GPTNeoXError.modelNotFound(path)
        }
        // Load model at path
        self.context = gptneox_init_from_file(path, params)
        // Print gptneox arch and cpu features info
        print(String(cString: gptneox_print_system_info()))
    }

    public override func predict(_ input: String, _ count: Int = 128, _ callback: ((String, Double) -> Bool) ) throws -> String {
        // Sample parameters
        let params = sampleParams
        // Add a space in front of the first character to match OG llama tokenizer behavior
        //let input = " " + input
        //print("bos \(gptneox_token_bos())")
        //print("eos \(gptneox_token_eos())")
        // oasst style prompt
        // TODO: Why does text special tokens not work?
        let prompt = input //"<|prompter|>" + input + "<|endoftext|><|assistant|>"
        // Tokenize prompt
        var inputTokens = tokenize(prompt)
        
        // OpenAssistant style prompt
        inputTokens.insert(gptneox_str_to_token(context, "<|prompter|>"), at: 0)
        inputTokens.append(gptneox_str_to_token(context, "<|endoftext|>"))
        inputTokens.append(gptneox_str_to_token(context, "<|assistant|>"))
        
        // StableLM Tuned Alpha
        /*
        //let systemTokens = tokenize("# StableLM Tuned (Alpha version)\n- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.\n- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.\n- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.\n- StableLM will refuse to participate in anything that could harm a human.")
        inputTokens.insert(gptneox_str_to_token(context, "<|USER|>"), at: 0)
        //inputTokens.insert(contentsOf: systemTokens, at: 0)
        //inputTokens.insert(gptneox_str_to_token(context, "<|SYSTEM|>"), at: 0)
        inputTokens.append(gptneox_str_to_token(context, "<|ASSISTANT|>"))
         */
        
        let inputTokensCount = inputTokens.count
        print("Input tokens: \(inputTokens)")
        // How many tokens to generate - check if theres space in context for atleast one token (or batch size tokens?)
        var outputRemaining = min(count, Int(contextParams.context) - inputTokens.count)
        guard outputRemaining > 0 else {
            throw GPTNeoXError.inputTooLong
        }
        // Past count - how many tokens have been fed so far
        var nPast = Int32(0)
        // Input
        var inputBatch = Array<gptneox_token>()
        // Inputs tokens, do not include inputs in output for conversational usage
        while inputTokens.count > 0 {
            // Clear input batch
            inputBatch.removeAll()
            // See how many to eval (up to batch size??? or can we feed the entire input)
            let evalCount = min(inputTokens.count, Int(params.batchSize))
            // Move tokens to batch
            inputBatch.append(contentsOf: inputTokens[0 ..< evalCount])
            inputTokens.removeFirst(evalCount)
            // Eval batch, "past_key_values" should be internally kept by eval and dictated by nPast
            if gptneox_eval(context, inputBatch, Int32(inputBatch.count), nPast, contextParams.numberOfThreads) != 0 {
                throw GPTNeoXError.failedToEval
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
        //var outputTokens = Array<gptneox_token>()
        var output = [String]()
        var outputRepeatTokens = Array<gptneox_token>()
        var outputTokensCount = 0
        // Loop until target count is reached
        while outputRemaining > 0 {
            // Pull a generation from context
            let outputToken = gptneox_sample_top_p_top_k(
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
            if outputToken == gptneox_token_eos() {
                outputRemaining = 0
                print("🤖 [EOS]")
                break
            }
            // Check for bos - skip callback if so
            if outputToken == gptneox_token_bos() {
                print("🤖 [BOS]")
                continue
            }
            // Convert token to string and callback
            if let cStr = gptneox_token_to_str(context, outputToken) {
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
                if gptneox_eval(context, [outputToken], 1, nPast, contextParams.numberOfThreads) != 0 {
                    throw GPTNeoXError.failedToEval
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
        //let input = " " + input

        // tokenize the prompt
        let inputs = tokenize(input)

        guard inputs.count > 0 else {
            return []
        }

        if gptneox_eval(context, inputs, Int32(inputs.count), Int32(0), contextParams.numberOfThreads) != 0 {
            throw GPTNeoXError.failedToEval
        }

        let embeddingsCount = Int(gptneox_n_embd(context))
        guard let embeddings = gptneox_get_embeddings(context) else {
            return []
        }
        return Array(UnsafeBufferPointer(start: embeddings, count: embeddingsCount))
    }

    deinit {
        gptneox_free(context)
    }

    // MARK: Internal

    func tokenize(_ input: String, bos: Bool = false, eos: Bool = false) -> [gptneox_token] {
        if input.count == 0 {
            return []
        }

        var embeddings: [gptneox_token] = Array<gptneox_token>(repeating: gptneox_token(), count: input.utf8.count)
        let n = gptneox_tokenize(context, input, &embeddings, Int32(input.utf8.count), bos)
        assert(n >= 0)
        embeddings.removeSubrange(Int(n)..<embeddings.count)
        
        if eos {
            embeddings.append(gptneox_token_eos())
        }
        
        return embeddings
    }
}


