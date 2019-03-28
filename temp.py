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

import scipy.io
train_data = scipy.io.loadmat('E:\\Memorial\\Machine learning\\Assignment_4\\train_32x32.mat', variable_names='X').get('X')
train_labels = scipy.io.loadmat('E:\\Memorial\\Machine learning\\Assignment_4\\train_32x32.mat', variable_names='y').get('y')
test_data = scipy.io.loadmat('E:\\Memorial\\Machine learning\\Assignment_4\\test_32x32.mat', variable_names='X').get('X')
test_labels = scipy.io.loadmat('E:\\Memorial\\Machine learning\\Assignment_4\\test_32x32.mat', variable_names='y').get('y')

print(train_data.shape, train_labels.shape)
print(test_data.shape, test_labels.shape)

train_data = np.transpose(train_data, (3, 0, 1, 2))
test_data = np.transpose(test_data, (3, 0, 1, 2))

print(train_data.shape)
print(test_data.shape)

train_data = train_data / 255.0
test_data = test_data / 255.0

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),padding='same',input_shape=(32,32,3)))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(64, (3, 3),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128, (3, 3),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.4))
fashion_model.add(Flatten())
fashion_model.add(Dense(128))
fashion_model.add(LeakyReLU(alpha=0.1))           
fashion_model.add(Dropout(0.3))
fashion_model.add(Dense(11, activation='softmax'))

fashion_model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


fashion_model.fit(train_data, train_labels, epochs=20, batch_size=64)

test_loss, test_acc = fashion_model.evaluate(test_data, test_labels)

print('Test accuracy:', test_acc)

predictions = fashion_model.predict(test_data)
print(np.argmax(predictions[0]))

from keras.models import load_model
fashion_model.save('E:\\Memorial\\Machine learning\\Assignment_4\\model_file.h5')
