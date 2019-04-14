# CoreML Custom Speech Recognition Tutorial

This tutorial will walk you through the steps to integrate a custom CoreML speech recognition model into an iOS app and have it process input from the microphone.

## Table of Contents
1. [The Speech Model](#the-speech-model)
2. [Converting to CoreML](#converting-to-coreml)
3. [Accessing the Model](#accessing-the-model)

## The Speech Model

The speech recognition model we will be using is a simple 1-dimensional convolutional neural network.
It will take in a second worth of audio input at 16 kHz and classify it as one of 12 different categories. [See the main page for a list of categories.](index)

The data used to train this model is from the Tensorflow Speech Commands Dataset. This dataset was recently featured in a kaggle competition (<https://www.kaggle.com/c/tensorflow-speech-recognition-challenge>) and our model roughly follows the parameters of that challenge. 

The training data consists of 1-second clips of various people saying the command. The format of the clips is in 16kHz, 16-bit mono wave file files. This means that for each clip, we load 16000 audio samples as a list of 16-bit integers. To make training a bit easier for our network, we'll normalize the input to a range of -1 to 1 by dividing the samples by 2^15. This is the only preprocessing we'll do and, depending on the audio format used by the microphone, may or may not be required in our iOS app.
'samples = samples / (2**15)

```python
input_shape = (16000,)

# NVIDIA's Formant analysis network
model = keras.models.Sequential()

# Formant Analysis network
# Autocorrelation layer
#keras.layers.Lambda(AutoCorrelationLayer, input_shape=input_shape, output_shape=(124,32,1))

model.add(keras.layers.Reshape((16000,1), input_shape=input_shape))
model.add(keras.layers.Conv1D(2, kernel_size=4, strides=2, activation='relu', padding='valid'))
model.add(keras.layers.Conv1D(4, kernel_size=4, strides=2, activation='relu', padding='valid'))
model.add(keras.layers.Conv1D(8, kernel_size=4, strides=2, activation='relu', padding='valid'))
model.add(keras.layers.Conv1D(16, kernel_size=4, strides=2, activation='relu', padding='valid'))
model.add(keras.layers.Conv1D(32, kernel_size=4, strides=2, activation='relu', padding='valid'))
model.add(keras.layers.Conv1D(64, kernel_size=4, strides=2, activation='relu', padding='valid'))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(150, activation='relu'))
model.add(keras.layers.Dense(len(labelNames), activation='softmax'))


#Compile model with loss and optimizer functions
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()
```

## Converting to CoreML

There are a few key things to keep in mind when converting a Keras model to CoreML:
- Keras must be version 1.2.2+
- The model must be defined using layers from the standalone keras module. Using the tensorflow.keras module will confuse the converter.
- CoreML only supports input of 1, 3 or 5 dimensions. If your model needs a different format, you must include a Reshape layer to transform it into that format.
- Not all Keras layers are supported by Apple's converter. If your model features more advanced layers, you can look into writing a [custom conversion tool](https://developer.apple.com/documentation/coreml/converting_trained_models_to_core_ml) or adding them as [custom layers](https://developer.apple.com/documentation/coreml/core_ml_api/creating_a_custom_layer)

## Accessing the Model
