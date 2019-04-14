//
//  AudioQueue.swift
//  SpeechRecognitionDemo
//
//  Created by owner on 2019-04-13.
//  Copyright © 2019 picottd. All rights reserved.
//

import Foundation

//Class which stores audio in a queue while waiting for further processing
class AudioQueue {
    //Pointer to the raw audio sample data
    let data : UnsafeMutablePointer<Int16>!
    
    //Size of the blocks of audio that are provided from the Audio Engine tap.
    let blockSize : Int!
    
    // Number of samples in the queue. Must be evenly divisible by block size
    let samples : Int!
    
    init(size : Int, blockSize : Int) {
        data = UnsafeMutablePointer<Int16>.allocate(capacity: size)
        data.assign(repeating: 0, count: size) // Fill in initial data with 0s
        self.blockSize = blockSize
        samples = size
    }
    
    //Add a new block to the queue, dequeueing the top one
    func enqueueBlock(block : UnsafeMutablePointer<Int16>){
        let blockCount = samples / blockSize
        
        //Shift blocks
        for b in 0..<blockCount-1{
            for i in 0..<blockSize {
                data[(b*blockSize)+i] = data[((b+1)*blockSize)+i]
            }
        }
        
        //Add new data
        for i in 0..<blockSize {
            data[((blockCount-1)*blockSize)+i] = block[i]
        }
    }
    
}
