# llama.swift

This package provides Swift bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp).

This package was hacked in an evening and is not production ready. This project is for educational purposes only.  

## Usage

```swift
import Llama

// 1. Open a model
let modelPath = Bundle.module.path(forResource: "gpt4all-lora-quantized", ofType: "bin")!
let llama = try Llama(path: modelPath)
        
// 2. Predict words based on input
let result = try llama.predict("Neil Armstrong: That's one small step for a man,")
print(result)

// 3. Get embeddings given input words 
let embeddings = try llama.embeddings("London bridge is falling down")
print(embeddings)
```

## Installation

The Swift Package Manager automates the distribution of Swift code. To use llama.swift with SPM, add a dependency to https://github.com/siuying/llama.swift.git

> **Note**: The models are not included in this repo. You must provide your own models when initializing the Llama object.

## License

The license for the Swift bindings is the same as the [license for the llama.cpp project](https://github.com/ggerganov/llama.cpp/blob/master/LICENSE), which is the MIT License.

See the LICENSE file for more details.
