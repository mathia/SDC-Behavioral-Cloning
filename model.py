#!/usr/bin/env python

# Basic LeNet architecture in Keras for training with the Behavioral Clone project data
# from https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/46a70500-493e-4057-a78e-b3075933709d/concepts/db4960f9-c8d3-4baf-9c7e-678341c1cf3e

import csv
import cv2
import numpy as np

## Load the training data from the CSV file into Numpy arrays
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    filename = line[0]
    current_path = 'data/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

print(len(images))
print(len(measurements))


## Load Keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

# Perform data normalization, zero mean and unit variance assuming image data values with normal distibution from 0-255.
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

# Build a classic LeNet model
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model-lenet.h5')
