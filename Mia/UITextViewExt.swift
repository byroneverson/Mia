//
//  UIScrollViewExt.swift
//  Mia
//
//  Created by Byron Everson on 12/26/22.
//

import UIKit

extension UITextView {
    
    func scrollToBottom() {
        let count = self.text.count
        if count > 0 {
            let location = count - 1
            let bottom = NSMakeRange(location, 1)
            self.scrollRangeToVisible(bottom)
        }
    }
    
}
