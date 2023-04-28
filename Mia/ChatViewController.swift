//
//  ViewController.swift
//  Mia
//
//  Created by Byron Everson on 12/25/22.
//

import UIKit

// Metal debug
//import Metal
//var device: MTLDevice!
//var commandQueue: MTLCommandQueue!

class ChatViewController: UIViewController {
    
    // Storyboard
    @IBOutlet var outputText: AnimatedTextView!
    @IBOutlet var inputText: AnimatedTextField!
    @IBOutlet var sendButton: AnimatedButton!
    @IBOutlet var gestureLabel: AnimatedTextField!
    
    // Chat font
    private var inAttributes: AttributeContainer!
    private var outAttributes: AttributeContainer!
    let chatFontSize: CGFloat = 14.0
    
    // Mia
    var ai: AI!
    
    // Local vars
    private var generating = false
    
    override func viewDidLoad() {
        print("ChatViewController viewDidLoad")
        super.viewDidLoad()
        
        // Chat styling
        inAttributes = AttributeContainer()
        let inParagraph = NSMutableParagraphStyle()
        inParagraph.alignment = .right
        inAttributes[AttributeScopes.UIKitAttributes.ForegroundColorAttribute.self] = .systemBlue
        inAttributes[AttributeScopes.UIKitAttributes.FontAttribute.self] = UIFont.systemFont(ofSize: chatFontSize, weight: .light)
        inAttributes[AttributeScopes.UIKitAttributes.ParagraphStyleAttribute.self] = inParagraph
        
        outAttributes = AttributeContainer()
        let outParagraph = NSMutableParagraphStyle()
        outParagraph.alignment = .left
        outAttributes[AttributeScopes.UIKitAttributes.ForegroundColorAttribute.self] = .lightGray
        outAttributes[AttributeScopes.UIKitAttributes.FontAttribute.self] = UIFont.systemFont(ofSize: chatFontSize, weight: .light)
        outAttributes[AttributeScopes.UIKitAttributes.ParagraphStyleAttribute.self] = outParagraph
        
        // Input text change callback
        inputText.addTarget(self, action: #selector(textDidChange(_:)), for: .editingChanged)
        
        // Input return button pressed
        inputText.addTarget(self, action: #selector(sendTouched(_:)), for: .primaryActionTriggered)
        
        // Default to loading model, disable send button until loaded
        sendButton.isEnabled = false
        gestureLabel.text = "Loading model..."
    }
    
    override func viewDidAppear(_ animated: Bool) {
        // Load AI
        ai = AI()
        gestureLabel.text = "🤖" //"💤"
        sendButton.isEnabled = true
    }

    override func viewWillDisappear(_ animated: Bool) {
        print("ChatViewController viewWillDisappear")
        // Cancel any AI in progress
        //ai.flagExit = true
        
        /*
        // Alert app delegate of window removal
        let appDelegate = NSApplication.shared.delegate as! AppDelegate
        appDelegate.closeChat(self)
         */
    }
    
    
    @IBAction func sendTouched(_ sender: Any) {
        // Check for cancel pressed
        if generating {
            print("🧙 User cancelled generation!")
            // Temp disable button to prevent extra presses until ready
            sendButton.isEnabled = false
            // Set flag for exit generation
            ai.flagExit = true
            return
        }
        print("in: \(inputText.text!)")
        //sendButton.isEnabled = false
        sendButton.setTitle("Cancel", for: .normal)
        generating = true
        gestureLabel.text = "💭"
        // Input attributed string segment
        if inputText.text != " " {
            var inAttrString = AttributedString("\n\n" + inputText.text! + "\n\n")
            inAttrString.mergeAttributes(inAttributes, mergePolicy: .keepNew)
            outputText.textStorage.append(NSAttributedString(inAttrString))
            // Output attributed string segment
            var outAttrString = AttributedString(" ")
            outAttrString.mergeAttributes(outAttributes, mergePolicy: .keepNew)
            outputText.textStorage.append(NSAttributedString(outAttrString))
            // Scroll output text to bottom
            self.outputText.scrollToBottom()
        }
        // Generate output
        let maxOutputLength = 2048
        ai.text(inputText.text!, maxOutputLength, { str, time in
            // Add next token, animated, show a different emoji if cancelling
            self.gestureLabel.text = "💬" //🗯
            self.outputText.typeOn(str, {
                if self.generating {
                    self.gestureLabel.text = "💭"
                } else {
                    self.gestureLabel.text = "🤖" //"💤"
                }
            })
            // Scroll output text to bottom
            self.outputText.scrollToBottom()
        }, { str in
            self.generating = false
            self.inputText.isHighlighted = true
            self.inputText.isEnabled = true
            self.gestureLabel.text = "🤖" //"💤"
            // Reset button - do not enable, button will enable once new text is entered
            self.sendButton.setTitle("Send", for: .normal)
            //self.sendButton.isEnabled = true
        })
        inputText.text = ""
        inputText.isHighlighted = false
        inputText.isEnabled = false
        //print("out: \(outputText.stringValue)")
    }
    
    // Make sure there is at least a space character for the ability to send
    @objc func textDidChange(_ textField: Any) {
        //print("ChatVC textDidChange")
        // Basically just waits until you start typing to let you press send
        if inputText.text == "" {
            sendButton.isEnabled = false
        } else if (!ai.flagResponding) {
            sendButton.isEnabled = true
        }
    }

    

}

