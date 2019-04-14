//
//  SpeechModelWrapper.swift
//  SpeechRecognitionDemo
//
//  Created by Daniel Picott on 2019-04-11.
//  Copyright © 2019 picottd. All rights reserved.
//

import Foundation
import CoreML

//Wrapper class for the FlatSpeech trained model
//FlatSpeech is only about 60% accurate but doesn't require special preprocessing besides normalization of audio samples
class FlatSpeechModelWrapper {
    
    //Load the model. Xcode automatically generates a class for it.
    let model = FlatSpeech()
    
    
    func Classify(audioQueue : AudioQueue) throws -> FlatSpeechOutput {
        let audio = audioQueue.data
        let audioArray = try MLMultiArray(shape: [16000], dataType: .double)
        
        //Process audio samples into array
        var max = 0.0
        var min = 0.0
        for i in 1..<16000{
            var sample = Double(audio![i])
            if (sample < min){
                min = sample
            }
            if (sample > max){
                max = sample
            }
            sample = sample / pow(2, 15) // Normalize
            audioArray[i] = NSNumber(value: sample)
        }
        print(min, max)
        //audioArray[0] = [NSNumber](repeating: 0, count: 16000)
        let input = FlatSpeechInput(input1: audioArray)
        let output = try model.prediction(input: input)
        
        return output
    }
    
}
