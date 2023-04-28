//
//  PyTorchLoader.swift
//  Mia
//
//  Created by Byron Everson on 2/17/23.
//

import Foundation

// This class is able to process a pytorch_model.bin using a config.json
// This is mainly for loading transformers library models, but should work with any pytorch model
class PyTorchLoader {
    
    // Load a model dictionary (array of anys)
    // TODO: Update to allow for using snapshots directly from huggingface style directory structure (aliases)
    static func load(_ modelURL: URL) -> (Dictionary<String, Any>, Array<Any>)? {
        // Config file and model binary
        let modelConfigURL = modelURL.appendingPathComponent("config").appendingPathExtension("json")
        //let modelBinaryURL = modelURL.appendingPathComponent("pytorch_model").appendingPathExtension("bin")
        // Load json config
        guard let configData = NSData(contentsOf: modelConfigURL) else {
            print("PyTorchLoader init error configData")
            return nil
        }
        guard let configDict = try? JSONSerialization.jsonObject(with: configData as Data) as? Dictionary<String, Any> else {
            print("PyTorchLoader init error configDict")
            return nil
        }
        // Debug print config contents, we will use this to interpret binary file
        print("PyTorchLoader configDict")
        print(configDict)
        
        // Binary file should not be stored in memory, it should be streamed into the model
        // This dict/array will next be loaded to a model then released
        /*guard let modelDict = Pickle.load(modelBinaryURL) else {
            print("PyTorchLoader init error Pickle load")
            return nil
        }*/
        let modelDict = Array<Any>()
        return (configDict, modelDict)
    }

    
}
