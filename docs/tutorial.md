# CoreML Custom Speech Recognition Tutorial

This tutorial will walk you through the steps to integrate a custom CoreML speech recognition model into an iOS app and have it process input from the microphone.

## Table of Contents
1. [The Speech Model](#the-speech-model)
2. [Converting to CoreML](#converting-to-coreml)
3. [Accessing the Model](#accessing-the-model)

## The Speech Model

The speech recognition model we will be using is a simple 1-dimensional convolutional neural network.
It will take in a second worth of audio input at 16 kHz and classify it as one of 12 different categories. [See the main page for a list of categories.](index)

The data used to train this model is from the Tensorflow Speech Commands Dataset. This dataset was recently featured in a kaggle competition ([https://www.kaggle.com/c/tensorflow-speech-recognition-challenge]) and our model roughly follows the parameters of that challenge.

## Converting to CoreML

There are a few key things to keep in mind when converting a Keras model to CoreML:
- Keras must be version 1.2.2+
- The model must be defined using layers from the standalone keras module. Using the tensorflow.keras module will confuse the converter.
- ...

## Accessing the Model
