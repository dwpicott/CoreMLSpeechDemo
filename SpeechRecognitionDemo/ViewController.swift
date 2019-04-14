//
//  ViewController.swift
//  SpeechRecognitionDemo
//
//  Created by owner on 2019-04-02.
//  Copyright © 2019 picottd. All rights reserved.
//

import UIKit
import AVFoundation
import Foundation

class ViewController: UIViewController {
    
    @IBOutlet weak var commandLabel: UILabel!
    @IBOutlet weak var commandImageView: UIImageView!
    @IBOutlet weak var currentClassLabel: UILabel!
    @IBOutlet weak var confidenceSlider: UISlider!
    @IBOutlet weak var confidenceLabel: UILabel!
    
    var recorder:AVAudioRecorder!
    var recordingSession:AVAudioSession!
    var updateTimer: Timer?
    
    //Time in seconds between each attempt to identify a command
    let updatePeriod = 0.1
    
    //Required confidence before an action is triggered by a speech command
    var confidenceThreshold = 0.8
    
    let model = FlatSpeechModelWrapper()

    // Recording audio:
    var engine = AVAudioEngine()
    
    //Queue stores buffered audio in
    var audioQueue = AudioQueue(size: 16000, blockSize: 1600)
    
    @IBAction func confidenceSliderChanged(_ sender: Any) {
        confidenceThreshold = Double(confidenceSlider.value)
        confidenceLabel.text = String(format: "Confidence threshold: %.0f%%", confidenceThreshold*100)
    }
    
    func startRecording() {
        engine.stop()
        engine.reset()
        engine = AVAudioEngine()
        
        recordingSession = AVAudioSession.sharedInstance()
            
        do {
            try recordingSession.setPreferredSampleRate(16000.0)
            try recordingSession.setPreferredInputNumberOfChannels(1)
            try recordingSession.setCategory(.record, mode: .default)
            try recordingSession.setMode(.measurement)
            try recordingSession.setPreferredIOBufferDuration(0.1)
            try recordingSession.setActive(true)
            recordingSession.requestRecordPermission() { [unowned self] allowed in
                DispatchQueue.main.async {
                    if allowed {
                        print("Gained microphone permission.")
                    } else {
                        print("Microphone permission was denied.")
                    }
                }
            }

        } catch {
            
            assertionFailure("AVAudioSession setup error: \(error)")
        }
        
        let input = engine.inputNode
        
        let downmixer = AVAudioMixerNode()
        engine.attach(downmixer)
        
        /*
        input.installTap(onBus: 0, bufferSize: 4410, format: input.inputFormat(forBus: 0)) { (buffer, time) in
            print(buffer.frameLength, buffer.frameCapacity)
            print(buffer.floatChannelData?.pointee.pointee)
        }*/
        
        downmixer.installTap(onBus: 0, bufferSize: 1600, format: downmixer.outputFormat(forBus: 0)) { (buffer, time) in
            print(buffer.int16ChannelData!.pointee.pointee)
            print(buffer.frameLength, buffer.frameCapacity)
            print(buffer.format)
            self.handleAudioInput(buffer: buffer, time: time)
        }
        
        let inputFormat = input.inputFormat(forBus: 0)
        let format16KHzMono = AVAudioFormat(commonFormat: AVAudioCommonFormat.pcmFormatInt16, sampleRate: 11025.0, channels: 1, interleaved: true)
        
        engine.connect(input, to: downmixer, format: inputFormat)
        engine.connect(downmixer, to: engine.mainMixerNode, format: format16KHzMono)
        
        //Prevent audio from being sent to the output
        engine.mainMixerNode.outputVolume = 0
        
        engine.prepare()
        try! engine.start()
    }
    
    func stopRecording() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
    }
    
    //Process the audio received from the input node
    func handleAudioInput(buffer: AVAudioPCMBuffer, time: AVAudioTime){
        if (buffer.int16ChannelData != nil && buffer.frameLength == 1600){
            audioQueue.enqueueBlock(block: buffer.int16ChannelData!.pointee)
        }
        else{
            print("Error: Buffer not filled.")
            print(buffer.frameLength)
            print(buffer.int16ChannelData)
        }
    }
    
    //Attempt to classify the currently buffered audio
    func classifyAudio(){
        //return
        do{
            let output = try model.Classify(audioQueue: audioQueue)
            
            let confidence = (output.output1[output.classLabel] ?? 0)
            print(String(format: "Hearing: \(output.classLabel) with confidence: %.4f", confidence))
            currentClassLabel.text = String(format: "Hearing \"\(output.classLabel)\" with confidence: %.1f%%", confidence * 100)
            
            if (confidence >= confidenceThreshold && output.classLabel != "silence" && output.classLabel != "unknown"){
                commandLabel.text = "Last command: \(output.classLabel)"
                
                //Take an action based on the command (in this case, show an image)
                switch (output.classLabel){
                case "yes":
                    commandImageView.image = UIImage(named: "thumb-up")
                    break
                case "no":
                    commandImageView.image = UIImage(named: "thumb-down")
                    break
                case "up":
                    commandImageView.image = UIImage(named: "plain-arrow-up")
                    break
                case "down":
                    commandImageView.image = UIImage(named: "plain-arrow-down")
                    break
                case "left":
                    commandImageView.image = UIImage(named: "plain-arrow-left")
                    break
                case "right":
                    commandImageView.image = UIImage(named: "plain-arrow-right")
                    break
                case "on":
                    commandImageView.image = UIImage(named: "light-bulb-on")
                    break
                case "off":
                    commandImageView.image = UIImage(named: "light-bulb-off")
                    break
                case "stop":
                    commandImageView.image = UIImage(named: "traffic-lights-red")
                    break
                case "go":
                    commandImageView.image = UIImage(named: "traffic-lights-green")
                    break
                default:
                    print("Unexpected classification: \(output.classLabel)")
                    break
                }
            }
            
        }
        catch let err{
            print(err)
        }
    }
    
    //Register a timer for classifying the Audio
    func startClassification(){
        guard updateTimer == nil else {return}
        self.updateTimer = Timer(timeInterval: self.updatePeriod, repeats: true, block: { (timer) in
            self.classifyAudio()
        })
        RunLoop.current.add(updateTimer!, forMode: .default)
    }
    
    //Stop the classification timer
    func stopClassification(){
        guard updateTimer != nil else {return}
        updateTimer?.invalidate()
        updateTimer = nil
    }
 
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        confidenceThreshold = Double(confidenceSlider.value)
        confidenceLabel.text = String(format: "Confidence threshold: %.0f%%", confidenceThreshold*100)
        startRecording()
        startClassification()
    }


}

