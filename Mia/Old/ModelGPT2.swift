//
//  ModelGPT2.swift
//  Mia
//
//  Created by Byron Everson on 12/25/22.
//

import Foundation
import CoreML
import Accelerate

// TODO: Float16 implementation  - This might prove to be fun/difficult to replace or extend Accelerate
// TODO: The trained model probably is sentence completion, which expects no EOS in input, we want question answering model

// GPT2 - LM (Language Model)
class ModelGPT2: Model {
    
    enum DecodingStrategy {
        /// At each time step, we select the most likely next token
        case greedy
        /// Sample only from the top-k most-probable tokens (k is a hyper-parameter).
        case topK(Int)
        /// Sample from the top tokens with a cumulative probability just above a threshold (nucleus/top-p).
        case topP(Double)
    }
    
    // GPT-2 / "3" Byte-pair encoding (BPE)
    private(set) var tokenizer: TokenizerGPT2! = nil
    
    // Model config
    private(set) var eosToken = 0 // Model eos token e.g. 50256
    private(set) var contextSize = 0 // Model max input/sequence length, usually the same as n_positions for obvious reasons
    private(set) var embedSize = 0 // Model embedding "depth"
    private(set) var vocabSize = 0 // Vocab size
    private(set) var layerCount = 0 // Number of decoder layers
    private(set) var headCount = 0 // Number of heads per decoder layer
    private(set) var layerNormEpsilon: Float32 = 0 // Small value to prevent div by zero
    
    // Token embedding weights
    private(set) var wte: [Float32]! // (vocabSize, embedSize)
    // Position embedding weights
    private(set) var wpe: [Float32]! // (contextSize, embedSize)
    
    // Masked Self-Attention half of decoder
    private(set) var ln_1_weight: [[Float32]]!
    private(set) var ln_1_bias: [[Float32]]!
    private(set) var attn_qkv_weight: [[Float32]]!
    private(set) var attn_qkv_bias: [[Float32]]!
    private(set) var attn_proj_weight: [[Float32]]!
    private(set) var attn_proj_bias: [[Float32]]!
    
    // FF MLP half of decoder
    private(set) var ln_2_weight: [[Float32]]!
    private(set) var ln_2_bias: [[Float32]]!
    private(set) var mlp_fc_weight: [[Float32]]!
    private(set) var mlp_fc_bias: [[Float32]]!
    private(set) var mlp_proj_weight: [[Float32]]!
    private(set) var mlp_proj_bias: [[Float32]]!
    
    // Final weights - layer norm and head
    private(set) var ln_f_weight: [Float32]!
    private(set) var ln_f_bias: [Float32]!
    private(set) var lm_head_weight: [Float32]!
    
    // Prediction
    private var strategy: DecodingStrategy
    private var predict: ((_ probs: UnsafeMutablePointer<Float32>) -> (Int, Float))!
    private var position_ids: MLMultiArray!
    private var predict_p: Float = 0.75
    private var predict_k = 10 //50
    
    // Init gpt2 model
    init(_ modelName: String, decodingStrategy: DecodingStrategy = .greedy /*.topK(4)*/ ) {
        print("ModelGPT2 init \(modelName)")
        
        // Decoder strategy
        strategy = decodingStrategy
        
        // Super init
        super.init()
        
        // Bundle resources
        guard let resourceURL = Bundle.main.resourceURL else {
            print("ModelGPT2 init error resourceURL")
            return
        }
        // Model urls within bundle
        let modelURL = resourceURL.appendingPathComponent(modelName, isDirectory: true)
        
        // Pytorch loader - todo: if pytorch is used
        guard let (configDict, _ /*modelDict*/) = PyTorchLoader.load(modelURL) else {
            print("ModelGPT2 init error PyTorchLoader")
            return
        }
        
        // Config
        eosToken = configDict["eos_token_id"] as? Int ?? 0
        contextSize = configDict["n_ctx"] as? Int ?? 0
        embedSize = configDict["n_embd"] as? Int ?? 0
        vocabSize = configDict["vocab_size"] as? Int ?? 0
        // Decoder specific
        layerCount = configDict["n_layer"] as? Int ?? 0
        headCount = configDict["n_head"] as? Int ?? 0
        layerNormEpsilon = configDict["layer_norm_epsilon"] as? Float32 ?? 1e-05
        
        // Load layers weights/biases
        let paramsURL = modelURL.appendingPathComponent("params", isDirectory: true)
        
        // Token and position embeddings
        wte = loadFloat32Array("wte".binURL(paramsURL))
        wpe = loadFloat32Array("wpe".binURL(paramsURL))
        
        // Layer norm 1
        ln_1_weight = loadFloat32Array("ln_1_weight".binURL(paramsURL)).split(layerCount)
        ln_1_bias = loadFloat32Array("ln_1_bias".binURL(paramsURL)).split(layerCount)
        // Masked Self-Attention
        attn_qkv_weight = loadFloat32Array("attn_qkv_weight".binURL(paramsURL)).split(layerCount)
        attn_qkv_bias = loadFloat32Array("attn_qkv_bias".binURL(paramsURL)).split(layerCount)
        attn_proj_weight = loadFloat32Array("attn_proj_weight".binURL(paramsURL)).split(layerCount)
        attn_proj_bias = loadFloat32Array("attn_proj_bias".binURL(paramsURL)).split(layerCount)
        // Layer norm 2
        ln_2_weight = loadFloat32Array("ln_2_weight".binURL(paramsURL)).split(layerCount)
        ln_2_bias = loadFloat32Array("ln_2_bias".binURL(paramsURL)).split(layerCount)
        // FF MLP half
        mlp_fc_weight = loadFloat32Array("mlp_fc_weight".binURL(paramsURL)).split(layerCount)
        mlp_fc_bias = loadFloat32Array("mlp_fc_bias".binURL(paramsURL)).split(layerCount)
        mlp_proj_weight = loadFloat32Array("mlp_proj_weight".binURL(paramsURL)).split(layerCount)
        mlp_proj_bias = loadFloat32Array("mlp_proj_bias".binURL(paramsURL)).split(layerCount)
        
        // Final layer norm w&b
        ln_f_weight = loadFloat32Array("ln_f_weight".binURL(paramsURL))
        ln_f_bias = loadFloat32Array("ln_f_bias".binURL(paramsURL))
        
        // LM Head weights
        lm_head_weight = loadFloat32Array("lm_head_weight".binURL(paramsURL))
        let lm_head_weight_ptr = lm_head_weight.mutPtr
        // Huggingface Pytorch shape is (50257, 1280) so needs to be transposed to (1280, 50257)
        vDSP_mtrans(lm_head_weight_ptr, 1, lm_head_weight_ptr, 1, vDSP_Length(contextSize), vDSP_Length(vocabSize))
        
        // Tokenizer
        tokenizer = TokenizerGPT2(modelURL)
        
        // Strategy switch
        // TODO: Finish topK and topP
        switch strategy {
        case .greedy:
            predict = predictGreedy
        case .topK(let k):
            predict_k = k
            predict = predictTopK
        case .topP(let p):
            predict_p = Float(p)
            predict = predictGreedy //predictP
        }
        
        print("ModelGPT2 init completed!")
    }
    
    // Helpers to load float32 binary array from file
    func loadFloat32Array(_ url: URL) -> [Float32] {
        print("ModelGPT2 loadFloat32Array \(url.lastPathComponent)")
        guard let file = try? FileHandle(forReadingFrom: url) else {
            return []
        }
        guard var data = try? file.readToEnd() else {
            return []
        }
        let dataPtr = data.withUnsafeMutableBytes { $0 }.bindMemory(to: Float32.self)
        //let bufferPtr = UnsafeMutableBufferPointer<Float32>.allocate(capacity: dataPtr.count)
        //_ = bufferPtr.initialize(from: dataPtr)
        return Array<Float32>(dataPtr) //bufferPtr)
    }
    
    // MARK: Decoding according to the model "strategy"
    
    func predictGreedy(_ probs: UnsafeMutablePointer<Float32>) -> (Int, Float32) {
        // Best
        //let nextToken = Math.argmax(probs, vocabSize)
        //return nextToken
        let probsBuffer = UnsafeBufferPointer(start: probs, count: vocabSize)
        let array = Array<Float32>(probsBuffer)
        let x = array.enumerated().map { ($0, $1) }.sorted(by: { $0.1 > $1.1 })
        print(x[0 ..< 3])
        print(x[(vocabSize - 3) ..< vocabSize])
        return x[0]
    }
    
    func predictTopK(_ probs: UnsafeMutablePointer<Float32>) -> (Int, Float32) {
        // Top-K
        let topK = Math.topK(probs, vocabSize, predict_k)
        // Debug
        for k in 0 ..< topK.0.count {
            print(" \(k) \(topK.0[k]) \(tokenizer.decode([topK.0[k]])) \(topK.1[k])")
        }
        // Random
        //let random = Math.randomK(topK)
        //return random
        return (topK.0[0], topK.1[0])
    }
    
    /*
    func predictP(_ tokens: [Int]) -> Int {
        // Check
        let tokenCount = min(tokens.count, seqLen)
        // Pad input
        let input_ids = MLMultiArray.from(
            (tokens.count > seqLen ? Array(tokens[0 ..< seqLen]) : tokens) + Array(repeating: 0, count: seqLen - tokenCount)
        )
        // Position ids, cheaper to do once instead of every single token
        let position_ids = MLMultiArray.from(Array(0 ..< seqLen))
        // Prediction
        /*
        let output = try! debug_model.prediction(input_ids: input_ids, position_ids: position_ids)
        //let output = infer([input_ids, position_ids])
        // Slice? Re-shape?
        let outputLogits = MLMultiArray.slice(
            output.output_logits, //output,
            indexing: [.select(0), .select(tokenCount - 1), .slice, .select(0), .select(0)]
        )
        // Get "winner" using modified/optimized top-P algorithm
        let logits = MLMultiArray.toDoubleArray(outputLogits)
        return Math.topP(logits, predict_p)
         */
        return 0
    }
    */
    
    
    // MARK: Main generation loop
    // Will generate next `nTokens` (defaults to 10).
    // Calls an incremental `callback` for each new token, then returns the generated string at the end.
    
    // Based on n_ctx (history context)
    override func generate(_ input: String, _ maxOutputCount: Int = 64, _ callback: ((String, Double) -> Bool)) -> String {
        
        // MARK: Init
        // Encode text to tokens - will change as tokens are generated
        let inputIds = tokenizer.encode(input)
        print("inputIds \(inputIds)")
        print("decoded \(tokenizer.decode(inputIds))")
        
        var tokenIds = inputIds
        // Start token - probably not needed
        //tokens.insert(eosToken, at: 0)
        // Append EOS token
        //tokens.append(eosToken)
        
        // MARK: Const vars
        let layerCount = layerCount
        let headCount = headCount
        let contextSize = contextSize
        //let contextSize_vDSP = vDSP_Length(contextSize)
        let vocabSize = vocabSize
        let vocabSize_vDSP = vDSP_Length(vocabSize)
        let embedSize = embedSize
        let embedSize_vDSP = vDSP_Length(embedSize) // Features/channels
        let embedSize_vv = Int32(embedSize)
        // Each layer has its own heads, maybe in the future each layer should have its own unique head count?
        // The current embedSize / weights count is split between heads, e.g. 1280 / 20 = 64 per head
        // embedSize / headCount = 64
        let headEmbedSize = embedSize / headCount // 1280/20=64
        let headEmbedSize_vDSP = vDSP_Length(headEmbedSize)
        let headEmbedSize_vv = Int32(headEmbedSize)
        var headEmbedSizeSqrt: Float32 = sqrt(Float32(headEmbedSize))
        
        // MARK: Temp buffers
        // TODO: How many of these can be reused and how many can be "recycled" between generations, keep in mind layers...
        // Actual token embedding buffer
        let embedBuffer = UnsafeMutableBufferPointer<Float32>.allocate(capacity: embedSize /*tokenEmbedCount*/)
        var embed = Array<Float32>(embedBuffer)
        let embedPtr = embed.mutPtr
        // Input/residual buffer
        let residualBuffer = UnsafeMutableBufferPointer<Float32>.allocate(capacity: embedSize /*tokenEmbedCount*/)
        var residual = Array<Float32>(residualBuffer)
        let residualPtr = residual.mutPtr
        // QKV Buffer
        let qkvBuffer = UnsafeMutableBufferPointer<Float32>.allocate(capacity: embedSize /*tokenEmbedCount*/ * 3)
        var qkv = Array<Float32>(qkvBuffer)
        let qkvPtr = qkv.mutPtr
        // Each query, key, and values should be pre-concatenated, so splitting is easy for the contiguous data
        let queryPtr = qkvPtr
        let keyPtr = qkvPtr.advanced(by: embedSize)
        let valuePtr = qkvPtr.advanced(by: embedSize * 2)
        // Past Keys
        let pastKeyBuffer = UnsafeMutableBufferPointer<Float32>.allocate(capacity: layerCount * headCount * contextSize * headEmbedSize)
        pastKeyBuffer.initialize(repeating: 0) // Init with zeroes for assumed padding
        var pastKey = Array<Float32>(pastKeyBuffer)
        let pastKeyPtr = pastKey.mutPtr
        // Past Values
        let pastValueBuffer = UnsafeMutableBufferPointer<Float32>.allocate(capacity: layerCount * headCount * contextSize * headEmbedSize)
        pastValueBuffer.initialize(repeating: 0) // Init with zeroes for assumed padding
        var pastValue = Array<Float32>(pastValueBuffer)
        let pastValuePtr = pastValue.mutPtr
        // Keys transpose, split per head
        // TODO: Re-use qkv? Or can we keep qkv as is and recycle its values to save time each generated token?
        let transposeKeyBuffer = UnsafeMutableBufferPointer<Float32>.allocate(capacity: headCount * contextSize * headEmbedSize)
        var transposeKey = Array<Float32>(transposeKeyBuffer)
        let transposeKeyPtr = transposeKey.mutPtr
        // QKV Score
        let scoreBuffer = UnsafeMutableBufferPointer<Float32>.allocate(capacity: headCount * contextSize /* * contextSize*/)
        var score = Array<Float32>(scoreBuffer)
        let scorePtr = score.mutPtr
        // Weighted attention output - filled directly into embedPtr instead
        /*let weightedBuffer = UnsafeMutableBufferPointer<Float32>.allocate(capacity: headCount * headEmbedSize /*tokenEmbedCount*/)
        var weighted = Array<Float32>(weightedBuffer)
        let weightedPtr = weighted.mutPtr*/
        // MLP Buffer
        let mlpBuffer = UnsafeMutableBufferPointer<Float32>.allocate(capacity: embedSize /*tokenEmbedCount*/ * 4)
        var mlp = Array<Float32>(mlpBuffer)
        let mlpPtr = mlp.mutPtr
        // Generation probabilities aka output logits
        let probsBuffer = UnsafeMutableBufferPointer<Float32>.allocate(capacity: vocabSize)
        var probs = Array<Float32>(probsBuffer)
        let probsPtr = probs.mutPtr
        
        // MARK: Model w&b
        let wtePtr = wte.mutPtr
        let wpePtr = wpe.mutPtr
        let ln_f_weight_ptr = ln_f_weight.mutPtr
        let ln_f_bias_ptr = ln_f_bias.mutPtr
        let lm_head_weight_ptr = lm_head_weight.mutPtr
        
        // TODO: SHOULD BE SETUP TO PROCESS ONE TOKEN AT A TIME, SKIPPING GEN STEP FOR EXISTING TOKENS
        // TODO: THIS WILL ESSENTIALLY PROCESS THE SAME AS IT WOULD DURING TRAINING, RELATIVELY SPEAKING
        // Maybe just populate what is needed from the previous token right now, then start generating on last token?
        
        // contextSize one-hot and wte mmul
        // contextSize wpe add
        // for each layer, ln_1 and qkv
        // store qkv to past
        
        // MARK: Token generation
        // Generated token, then decoded
        var generated: [Int] = []
        var decoded: [String] = []
        // Generate next token
        var j = 0 //tokenIndex = 0
        var generating = true
        while generating {
            
            /*var skipping = false
            if j < (tokens.count - 1) {
                skipping = true
            }*/
            
            //print("generate benchmark")
            //var time: CFAbsoluteTime = 0
            
            // MARK: Init token embedding
            /*
            // Convert to tokenIds to one-hot tensor first
            var oneHotArray = Array<Float32>(repeating: 0, count: vocabSize)
            // Set appropriate ones in array
            oneHotArray[tokens[j]] = 1
            // Matrix mutiply or index lookup using "wte" (word token embeddings) to produce embedding tensor
            vDSP_mmul(oneHotArray, 1, wtePtr, 1, embedPtr, 1, 1, vDSP_Length(embedSize), vDSP_Length(vocabSize))
             */
            // Just copy individual token embedding from wte
            let wtePtr = wtePtr.advanced(by: tokenIds[j] * embedSize)
            cblas_scopy(embedSize_vv, wtePtr, 1, embedPtr, 1)
            
            //print("")
            //print("0 - Wte:")
            //print(embed[0 ..< 3])
            //print(embed[(embed.count - 3) ..< embed.count])
            
            // Positional embedding - Add positional embedding element-wise, shape is (contextSize, embedSize)
            // TODO: Figure out why adding token encoding to position encoding is better than length-concat or feature-concat
            let wpePtr = wpePtr.advanced(by: j * embedSize)
            vDSP_vadd(embedPtr, 1, wpePtr, 1, residualPtr, 1, embedSize_vDSP)
            
            //print("")
            //print("0 - Input embeds:")
            //print(residual[0 ..< 3])
            //print(residual[(residual.count - 3) ..< residual.count])
            
            // Feed the context (tokens) through the layers
            for i in 0 ..< 1 /*layerCount*/ {
                // Current tokens - for attention (assumes following tokens are only pad)
                let currentSize = j + 1
                let currentSize_vDSP = vDSP_Length(currentSize)
                
                // Input of each layer/block should already be inside residual
                
                //time = CFAbsoluteTimeGetCurrent()
                
                // TODO: THIS STEP CAN BE CACHED, PREVIOUS CONTEXT WILL REMAIN THE SAME
                // MARK: Layer norm 1 - normalize per token and scale with ln_1 w&b
                let ln_1_weight_ptr = ln_1_weight[i].mutPtr
                let ln_1_bias_ptr = ln_1_bias[i].mutPtr
                // MVN (mean variance normalize) - per token in contextSize / all features (channels) together
                Math.meanVarianceNormalize(residualPtr, embedPtr, embedSize, layerNormEpsilon)
                vDSP_vmul(embedPtr, 1, ln_1_weight_ptr, 1, embedPtr, 1, embedSize_vDSP)
                vDSP_vadd(embedPtr, 1, ln_1_bias_ptr, 1, embedPtr, 1, embedSize_vDSP)
                
                //print("")
                //print("0 - Layer norm 1:")
                //print(embed[0 ..< 3])
                //print(embed[(embed.count - 3) ..< embed.count])
                
                // MARK: Attention layer - Create QKV (3 x embedSize), one per token
                // Will automatically populate queryPtr, keyPtr, and valuePtr, as they point at qkvPtr
                // Weight: MP x PN = MN
                let attn_qkv_weight_ptr = attn_qkv_weight[i].mutPtr
                let attn_qkv_bias_ptr = attn_qkv_bias[i].mutPtr
                
                //print("")
                //print("0 - Attention Weight:")
                //print(attn_qkv_weight[i][0 ..< 3])
                //print(attn_qkv_weight[i][(attn_qkv_weight[i].count - 3) ..< attn_qkv_weight[i].count])
                //print("")
                //print("0 - Attention Bias:")
                //print(attn_qkv_bias[i][0 ..< 3])
                //print(attn_qkv_bias[i][(attn_qkv_bias[i].count - 3) ..< attn_qkv_bias[i].count])
                
                // Shape of weights per layer is (embedSize, embedSize * 3)
                // Copy bias to qkv to be auto added by cblas_sgemm
                cblas_scopy(embedSize_vv * 3, attn_qkv_bias_ptr, 1, qkvPtr, 1)
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            1, embedSize_vv * 3, embedSize_vv,
                            1.0, // alpha
                            embedPtr, 1,
                            attn_qkv_weight_ptr, embedSize_vv * 3,
                            1.0, // beta
                            qkvPtr, embedSize_vv * 3)
                //vDSP_mmul(embedPtr, 1, attn_qkv_weight_ptr, 1, qkvPtr, 1, 1, embedSize_vDSP * 3, embedSize_vDSP)
                // Bias
                //vDSP_vadd(qkvPtr, 1, attn_qkv_bias_ptr, 1, qkvPtr, 1, embedSize_vDSP * 3)
                
                //print("")
                //print("0 - Attention Query:")
                //print(qkv[0 ..< 3])
                //print(qkv[embedSize - 3 ..< embedSize])
                //print("")
                //print("0 - Attention Key:")
                //print(qkv[(embedSize + 0) ..< (embedSize + 3)])
                //print(qkv[(embedSize * 2 - 3) ..< (embedSize * 2)])
                //print("")
                //print("0 - Attention Value:")
                //print(qkv[(embedSize * 2 + 0) ..< (embedSize * 2 + 3)])
                //print(qkv[(embedSize * 3 - 3) ..< (embedSize * 3)])
                
                // MARK: Score each token
                // 1. query ⋅ keyT = score (contextSize, contextSize)
                // MARK: Attention amalgamation
                // 2. score ⋅ value = attention (contextSize, embedSize)
                Tasker.branch(headCount) { h in
                //for h in 0 ..< headCount {
                    
                    // Current source QKV - heads are simply concat
                    let hOffset = h * headEmbedSize
                    let currentQueryPtr = queryPtr.advanced(by: hOffset)
                    let currentKeyPtr = keyPtr.advanced(by: hOffset)
                    let currentValuePtr = valuePtr.advanced(by: hOffset)
                    
                    // Copy current token to last - after all reshape and transpose to minimize repeat operations
                    // Store both k and v as reshaped to (layer, head, context, headEmbed)
                    let pastContextOffset = j * headEmbedSize
                    let pastHeadSize = contextSize * headEmbedSize
                    let pastHeadOffset = h * pastHeadSize
                    let pastLayerSize = headCount * pastHeadSize
                    let pastLayerOffset = i * pastLayerSize
                    let pastCurrentOffset = pastLayerOffset + pastHeadOffset + pastContextOffset
                    let pastCurrentKeyPtr = pastKeyPtr.advanced(by: pastCurrentOffset)
                    let pastCurrentValuePtr = pastValuePtr.advanced(by: pastCurrentOffset)
                    cblas_scopy(headEmbedSize_vv, currentKeyPtr, 1, pastCurrentKeyPtr, 1)
                    cblas_scopy(headEmbedSize_vv, currentValuePtr, 1, pastCurrentValuePtr, 1)
                    
                    // The entire past key array for this head
                    let pastKeyPtr = pastKeyPtr.advanced(by: pastLayerOffset + pastHeadOffset)
                    // The transpose temp array is only for a single layer at a time, but has all heads for parallelism
                    let transposeKeyPtr = transposeKeyPtr.advanced(by: pastHeadOffset)
                    // Transpose keys to temp transpose buffer, currentSize instead of contextSize
                    // (currentSize, headEmbedSize) -> (headEmbedSize, currentSize)
                    // TODO: Is this okay if currentSize is not some n_power value? e.g. like 9 or 17?
                    vDSP_mtrans(pastKeyPtr, 1, transposeKeyPtr, 1, headEmbedSize_vDSP, currentSize_vDSP)
                    
                    // Score buffer, offset by the full contextSize in allocated memory
                    let scoreOffset = h * contextSize
                    let scorePtr = scorePtr.advanced(by: scoreOffset)
                    // Mat-mul the current query with the transposed keys, but only the currestSize # of keys
                    // Using currentSize is the equivalent of causal masking
                    // Weight: MP x PN = MN
                    vDSP_mmul(currentQueryPtr, 1, transposeKeyPtr, 1, scorePtr, 1, 1, currentSize_vDSP, headEmbedSize_vDSP)
                    
                    // Divide by 8 (sqrt of headEmbedSize) to assist numerical stability
                    // Ported from working coreml code so we will keep it for now even though softmax should produce the same results either way, just a trick to prevent nan/inf during e^n softmax I think
                    vDSP_vsdiv(scorePtr, 1, &headEmbedSizeSqrt, scorePtr, 1, currentSize_vDSP)
                    
                    // Softmax score rows (per token) for probs, only factor in current context so far
                    Math.softmax(scorePtr, currentSize)
                    
                    // Apply probs to values to get amalgamated headEmbedSize partial embedding
                    let pastValuePtr = pastValuePtr.advanced(by: pastLayerOffset + pastHeadOffset)
                    // Mat-mul directly into correct location of token embed
                    let embedPtr = embedPtr.advanced(by: hOffset)
                    // Weight: MP x PN = MN
                    vDSP_mmul(scorePtr, 1, pastValuePtr, 1, embedPtr, 1, 1, headEmbedSize_vDSP, currentSize_vDSP)
                }
                
                //print("")
                //print("0 - Attention merged:")
                //print(embed[0 ..< 3])
                //print(embed[(embed.count - 3) ..< embed.count])
                
                //print("0 - Attention proj Weight:")
                //print(attn_proj_weight[i][0 ..< 4])
                //print(attn_proj_weight[i][(attn_proj_weight[i].count - 4) ..< attn_proj_weight[i].count])
                //print("0 - Attention proj Bias:")
                //print(attn_proj_bias[i][0 ..< 4])
                //print(attn_proj_bias[i][(attn_proj_bias[i].count - 4) ..< attn_proj_bias[i].count])
                
                // MARK: Attention projection - project concatenated heads weights
                // embedSize to embedSize
                let attn_proj_weight_ptr = attn_proj_weight[i].mutPtr
                let attn_proj_bias_ptr = attn_proj_bias[i].mutPtr
                // Weight: MP x PN = MN
                vDSP_mmul(embedPtr, 1, attn_proj_weight_ptr, 1, embedPtr, 1, 1, embedSize_vDSP, embedSize_vDSP)
                // Bias
                vDSP_vadd(embedPtr, 1, attn_proj_bias_ptr, 1, embedPtr, 1, embedSize_vDSP)
                
                //print("")
                //print("0 - Attention proj:")
                //print(embed[0 ..< 3])
                //print(embed[(embed.count - 3) ..< embed.count])

                // MARK: Residual 1 (skip connection) - Element-wise add attention output and input of this decoder
                vDSP_vadd(residualPtr, 1, embedPtr, 1, residualPtr, 1, embedSize_vDSP)
                
                //print("")
                //print("0 - Residual 1:")
                //print(residual[0 ..< 3])
                //print(residual[(residual.count - 3) ..< residual.count])
                
                // TODO: CURRENT TOKEN APPROVED
                // MARK: Layer norm 2 - normalize per token and scale with ln_2 w&b
                let ln_2_weight_ptr = ln_2_weight[i].mutPtr
                let ln_2_bias_ptr = ln_2_bias[i].mutPtr
                // MVN (mean variance normalize) - per token in contextSize / all features (channels) together
                //var ln_2_mean: Float32 = 0
                //var ln_2_stdDev: Float32 = 0 // stdDev is square root of variance to match original units
                //vDSP_normalize(residualPtr, 1, embedPtr, 1, &ln_2_mean, &ln_2_stdDev, embedSize_vDSP)
                Math.meanVarianceNormalize(residualPtr, embedPtr, embedSize, layerNormEpsilon)
                vDSP_vmul(embedPtr, 1, ln_2_weight_ptr, 1, embedPtr, 1, embedSize_vDSP)
                vDSP_vadd(embedPtr, 1, ln_2_bias_ptr, 1, embedPtr, 1, embedSize_vDSP)
                
                print("0 - Layer norm 2:")
                print(embed[0 ..< 3])
                print(embed[(embed.count - 3) ..< embed.count])
                
                /*

                // MARK: FF MLP layer - Fully connected with mlp fc w&b
                let mlp_fc_weight_ptr = mlp_fc_weight[i].mutPtr
                let mlp_fc_bias_ptr = mlp_fc_bias[i].mutPtr
                // Weight: MP x PN = MN
                vDSP_mmul(embedPtr, 1, mlp_fc_weight_ptr, 1, mlpPtr, 1, 1, embedSize_vDSP * 4, embedSize_vDSP)
                // Bias
                vDSP_vadd(mlpPtr, 1, mlp_fc_bias_ptr, 1, mlpPtr, 1, embedSize_vDSP * 4)
                
                // TODO: CURRENT TOKEN APPROVED
                // MARK: Activation - gelu_new (gelu tanh appr)
                Math.geluApprTanh/*Sig*/(mlpPtr, embedSize * 4)
                
                // MARK: FF MLP layer - Project to correct dims with mlp proj w&b
                let mlp_proj_weight_ptr = mlp_proj_weight[i].mutPtr
                let mlp_proj_bias_ptr = mlp_proj_bias[i].mutPtr
                // Weight: MP x PN = MN
                vDSP_mmul(mlpPtr, 1, mlp_proj_weight_ptr, 1, embedPtr, 1, 1, embedSize_vDSP, embedSize_vDSP * 4)
                // Bias
                vDSP_vadd(embedPtr, 1, mlp_proj_bias_ptr, 1, embedPtr, 1, embedSize_vDSP)
                 
                 */

                // MARK: Residual 2 (skip connection) - Element-wise add mlp proj and attention output (layer output)
                vDSP_vadd(residualPtr, 1, embedPtr, 1, residualPtr, 1, embedSize_vDSP)
                
                // MARK: Output of layer is residualPtr
            }
            
            // Increment "j" generation index for next loop
            j += 1
            
            // TODO: Populate attention past KV before generation and start j generation index at current
            // Check if we are only populating attention, otherwise skip to next generation
            // Is this safe for generation with no input?
            if j < tokenIds.count {
                print("skip gen")
                continue
            }
            
            // MARK: Final layer - MVN (mean variance normalize) all dimensions together
            // embedSize ln_f_weight & ln_f_bias
            // Only need to calculate for the final token in the sequence to predict next
            //var mean: Float32 = 0
            //var stdDev: Float32 = 0 // stdDev is square root of variance to match original units
            //vDSP_normalize(residualPtr, 1, embedPtr, 1, &mean, &stdDev, embedSize_vDSP)
            Math.meanVarianceNormalize(residualPtr, embedPtr, embedSize, layerNormEpsilon)
            vDSP_vmul(embedPtr, 1, ln_f_weight_ptr, 1, embedPtr, 1, embedSize_vDSP)
            vDSP_vadd(embedPtr, 1, ln_f_bias_ptr, 1, embedPtr, 1, embedSize_vDSP)
            
            // MARK: LM head weights - embedSize -> vocabSize
            vDSP_mmul(embedPtr, 1, lm_head_weight_ptr, 1, probsPtr, 1, 1, vocabSize_vDSP, embedSize_vDSP)
            
            // Softmax vocab probs - all prediction algorithms will expect this
            //Math.softmax(probsPtr, vocabSize)
            
            // Layer time
            //time = CFAbsoluteTimeGetCurrent() - time
            //print("\(generated.count) \(time)")
            
            // Predict next token based on probs
            // Predict next - the shape of transformer output should be (vocabSize, 1) a simple 1d array of probs
            let ((tokenId, tokenProb), tokenTime) = Utils.time {
                return predict(probsPtr)
            }
            
            // Keep generating until EOS token prob is higher than 50% (hyperparam) or until max output length is reached as a fallback
            if tokenId == eosToken {
                print("ModelGPT2 generate EOS")
                generating = false
                break
            }
            
            // Shift/append predicted token to sequence, rinse and repeat
            generated.append(tokenId)
            tokenIds.append(tokenId)
            /*if tokens.count > contextSize {
                inputTokens.removeFirst()
            }*/
            
            // Decode tokens at each token, is this really necessary? Kind of cool though
            let decodedToken = tokenizer.decode([tokenId])
            decoded.append(decodedToken)
            print("🤖 \(decodedToken) \(tokenId) \(tokenProb)")
            // Callback each token
            if callback(decodedToken, tokenTime) {
                // Early exit if requested by callback
                print("💀 Early exit")
                generating = false
                break
            }
            
            // Check to see if context is full
            if tokenIds.count > contextSize {
                //inputTokens.removeFirst()
                print("ModelGPT2 contextSize full")
                generating = false
                break
            }
            
            // Prevent runaway with maximum output count
            /*if generated.count >= maxOutputCount {
                print("ModelGPT2 generate maxOutputCount")
                generating = false
                break
            }*/
        }
        
        // TODO: Free=ing of previous allocations needed? Or already handled auto at end of this function?
        //residualPtr.deallocate()
        //qkvPtr.deallocate()
        //mlpPtr.deallocate()
        //probsPtr.deallocate()
        
        // MARK: End - Decode and return full stream of tokens
        return tokenizer.decode(generated)
        
    }
}





