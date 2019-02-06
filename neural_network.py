import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils    

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

MODEL_PATH = 'neural_network/mnist.h5'

class NeuralNetwork:

    def __init__(self):
        self.active_model = self.model()
        self.active_model.load_weights(MODEL_PATH)
    
    def model(self):
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        return model

    def train():
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        X_train /= 255
        X_test /= 255

        Y_train = np_utils.to_categorical(y_train, 10)
        Y_test = np_utils.to_categorical(y_test, 10)

        model = self.model();
        model.fit(X_train, Y_train, batch_size=128, epochs=20,verbose=2,validation_data=(X_test, Y_test))

        model.save(MODEL_PATH)
        active_model = model

    def predict_number(self,number_image):
        predictions = self.active_model.predict(number_image, steps=1)[0]
        return predictions
        