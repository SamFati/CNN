from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model

import scipy.io
from pathlib import Path

import os
fileDir = os.path.dirname(os.path.realpath('__file__'))
print(fileDir)
filenameTrain = os.path.join(fileDir, 'train_32x32.mat')
filenameTest = os.path.join(fileDir, 'test_32x32.mat')

train_data = scipy.io.loadmat(filenameTrain, variable_names='X').get('X')
train_labels = scipy.io.loadmat(filenameTrain, variable_names='y').get('y')
test_data = scipy.io.loadmat(filenameTest, variable_names='X').get('X')
test_labels = scipy.io.loadmat(filenameTest, variable_names='y').get('y')

train_data = np.transpose(train_data, (3, 0, 1, 2))
test_data = np.transpose(test_data, (3, 0, 1, 2))

train_data = train_data / 255.0
test_data = test_data / 255.0

classes = np.unique(train_labels)
nClasses = len(classes)


_model = Sequential()
_model.add(Conv2D(32, kernel_size=(3, 3),padding='same',input_shape=(32,32,3)))
_model.add(LeakyReLU(alpha=0.1))
_model.add(MaxPooling2D((2, 2),padding='same'))
_model.add(Dropout(0.25))
_model.add(Conv2D(64, (3, 3),padding='same'))
_model.add(LeakyReLU(alpha=0.1))
_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
_model.add(Dropout(0.25))
_model.add(Conv2D(128, (3, 3),padding='same'))
_model.add(LeakyReLU(alpha=0.1))                  
_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
_model.add(Dropout(0.4))
_model.add(Flatten())
_model.add(Dense(128))
_model.add(LeakyReLU(alpha=0.1))           
_model.add(Dropout(0.3))
_model.add(Dense(nClasses+1, activation='softmax'))

_model.compile(optimizer='adam', 
             loss='sparse_categorical_crossentropy',              
             metrics=['accuracy'])




_model.fit(train_data, train_labels, epochs=20, batch_size=64)

test_loss, test_acc = _model.evaluate(test_data, test_labels)

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

filenameHDF5 = os.path.join(fileDir, 'model_hdf5.h5')
_model.save(filenameHDF5)
