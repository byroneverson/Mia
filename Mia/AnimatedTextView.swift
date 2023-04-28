//
//  AnimatedTextView.swift
//  Mia
//
//  Created by Byron Everson on 12/26/22.
//

import UIKit

class AnimatedTextView: UITextView {
    
    var characterArray: [Character] = []
    var typingSpeed = 0.02
    private var timer: Timer?
    
    func typeOn(_ string: String, _ completion: (() -> ())?) {
        /*if string.isEmpty {
            print("typeOn error empty string")
            return
        }*/
        characterArray.append(contentsOf: Array(string))
        if timer != nil { return }
        timer = Timer.scheduledTimer(withTimeInterval: typingSpeed, repeats: true) { (timer) in
            // Check before
            /*if self.characterArray.isEmpty {
                timer.invalidate()
                self.timer = nil
                completion?()
            }*/
            // Append character
            self.textStorage.mutableString.append("\(self.characterArray[0])")
            //self.string.append(self.characterArray[0])
            // Check after
            self.characterArray.removeFirst()
            if self.characterArray.isEmpty {
                timer.invalidate()
                self.timer = nil
                completion?()
            }
        }
    }

}
