# %%
from PIL import Image
from cv2 import cvtColor
import numpy as np                   # advanced math library
import matplotlib.pyplot as plt      # MATLAB like plotting routines
import random                        # for generating random numbers
import cv2
from pandas import array
from Box import Box

import tensorflow as tf
# MNIST dataset is included in Keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential  # Model type to be used

# Types of layers to be used in our model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical


class DigitRecognizer():
    def __init__(self):
       self.model = Sequential()
       (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
       self.nb_classes = 10  # number of unique digits

       self.Y_train = to_categorical(self.y_train, self.nb_classes)
       self.Y_test = to_categorical(self.y_test, self.nb_classes)
       print("Digit Recognizer")
       self.loadModel('my_model')

    def loadModel(self, filename):
        self.model = tf.keras.models.load_model(filename)
        print("Model loaded from \'" + filename)

    def train(self):
        # The MNIST data is split between 60,000 28 x 28 pixel training images and 10,000 28 x 28 pixel images

        print("X_train shape", self.X_train.shape)
        print("y_train shape", self.y_train.shape)
        print("X_test shape", self.X_test.shape)
        print("y_test shape", self.y_test.shape)

        # Make the figures a bit bigger
        plt.rcParams['figure.figsize'] = (9, 9)
        # reshape 60,000 28 x 28 matrices into 60,000 784-length vectors.
        self.X_train = self.X_train.reshape(60000, 784)
        # reshape 10,000 28 x 28 matrices into 10,000 784-length vectors.
        self.X_test = self.X_test.reshape(10000, 784)

        # change integers to 32-bit floating point numbers
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')

        # normalize each value for each pixel for the entire vector for each input
        self.X_train /= 255
        self.X_test /= 255

        print("Training matrix shape", self.X_train.shape)
        print("Testing matrix shape", self.X_test.shape)

        self.model = Sequential()
        self.model.add(Dense(512, input_shape=(784,)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))
        self.model.summary()

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

        self.model.fit(self.X_train, self.Y_train,
                       batch_size=128, epochs=5, verbose=1)

    def test(self):
        score = self.model.evaluate(self.X_test, self.Y_test)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        predicted_classes = self.model.predict_classes(self.X_test)

        correct_indices = np.nonzero(predicted_classes == self.y_test)[0]

        incorrect_indices = np.nonzero(predicted_classes != self.y_test)[0]

        plt.figure()
        for i, correct in enumerate(correct_indices[:9]):
            plt.subplot(3, 3, i+1)
            plt.imshow(self.X_test[correct].reshape(
                28, 28), cmap='gray', interpolation='none')
            plt.title("Predicted {}, Class {}".format(
                predicted_classes[correct], self.y_test[correct]))

        plt.tight_layout()
        plt.show()

        plt.figure()
        for i, incorrect in enumerate(incorrect_indices[:9]):
            plt.subplot(3, 3, i+1)
            plt.imshow(self.X_test[incorrect].reshape(
                28, 28), cmap='gray', interpolation='none')
            plt.title("Predicted {}, Class {}".format(
                predicted_classes[incorrect], self.y_test[incorrect]))

        plt.tight_layout()
        plt.show()

    def prepare_image(self, im: Image.Image) -> np.array:
        img = np.array(im)
        img = cv2.resize(img, (28, 28))
        if (len(img.shape) == 3):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # to grayscale

        # [optional] my input was white bg, I turned it to black - {bitwise_not} turns 1's into 0's and 0's into 1's
        # img = cv2.bitwise_not(img)
        # img = img / 255                    # normalize
        cv2.imshow('', img)
        cv2.waitKey(0)
        img = img.reshape((1, 784))
        return img

    def recognize(self, input):
        # new_list = []
        # res = None
        # if(type(input) == list):
        #     for i in input:
        #         new_list.append(self.prepare_image(i))

        #     res = self.model(new_list)
            
        
        # else:
        res = self.model(self.prepare_image(input))

        return res

        return np.argmax(res), max(res)

# %%
