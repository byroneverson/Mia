//
//  Model.swift
//  Mia
//
//  Created by Byron Everson on 12/25/22.
//

import Foundation

enum AIModel {
    case StackLLama
    case OpenAssistant
}

class AI {
    
    var aiQueue = DispatchQueue(label: "Mia-Main", qos: .userInitiated, attributes: .concurrent, autoreleaseFrequency: .inherit, target: nil)
    
    //var model: Model!
    var model: Model!
    
    var flagExit = false
    private(set) var flagResponding = false
    
    init() {
        // MARK: OLD GPT2 CODE
        //let modelName = "gpt2-large"
        //model = ModelGPT2(modelName)
        //if model == nil {
        //    print("AI init error model load")
        //}
        
        print("AI init")
        
        // TODO: Need to renew apple developer membership to include necessary capability in provisioning profile
        // There is apparently no way around paying $107 to use this capability... wtf
        // com.apple.developer.kernel.extended-virtual-addressing
        // This will show available memory, not sure if mmap uses this pool, but nice to know anyway
        // Increased memory limit capability (not Extended Virtual Addressing) actually crashes mac catalyst on load...
        // This command returns 0 in mac catalyst
        print("AI init available process memory: \(os_proc_available_memory())")
        
        // MARK: LLaMa
        //let modelPath = Bundle.main.path(forResource: "ggml-llama-7b-se-rl-q4_0", ofType: "bin")!
        //model = try? LLaMa(path: modelPath)
        
        // MARK: GPTNeoX
        //let modelPath = Bundle.main.path(forResource: "ggml-GPT-NeoXT-Chat-Base-20B-q4_0", ofType: "bin")!
        let modelPath = Bundle.main.path(forResource: "ggml-oasst-sft-4-pythia-12b-epoch-3.5-q4_0", ofType: "bin")!
        //let modelPath = Bundle.main.path(forResource: "ggml-stablelm-tuned-alpha-7b-q4_0", ofType: "bin")!
        //let modelPath = Bundle.main.path(forResource: "ggml-stablelm-7b-sft-v7-epoch-3-q4_0", ofType: "bin")!
        //let modelPath = Bundle.main.path(forResource: "ggml-stablelm-base-alpha-3b-q4_0", ofType: "bin")!
        model = try? GPTNeoX(path: modelPath)
    }
    
    func loadModel(_ aiModel: AIModel) {
        switch aiModel {
        case .StackLLama:
            let modelPath = Bundle.main.path(forResource: "ggml-llama-7b-se-rl-q4_0", ofType: "bin")!
            model = try? LLaMa(path: modelPath)
        case .OpenAssistant:
            let modelPath = Bundle.main.path(forResource: "ggml-stablelm-7b-sft-v7-epoch-3-q4_0", ofType: "bin")!
            model = try? GPTNeoX(path: modelPath)
        }
    }
    
    func text(_ input: String, _ maxOutputCount: Int = 2048, _ tokenCallback: ((String, Double) -> ())?, _ completion: ((String) -> ())?) {
        flagResponding = true
        aiQueue.async {
            func mainCallback(_ str: String, _ time: Double) -> Bool {
                DispatchQueue.main.async {
                    tokenCallback?(str, time)
                }
                if self.flagExit {
                    // Reset flag
                    self.flagExit = false
                    // Alert model of exit flag
                    return true
                }
                return false
            }
            guard let completion = completion else { return }
            
            // Model output
            let output = try? self.model.predict(input, mainCallback)
            
            DispatchQueue.main.async {
                self.flagResponding = false
                completion(output ?? "[Error]")
            }
        }
    }
    
    
}




