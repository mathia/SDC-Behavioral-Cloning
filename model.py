#!/usr/bin/env python

import csv
import cv2
import numpy as np

def load_data():
    ## Load the training data from the CSV file into Numpy arrays
    lines = []
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # Filter out all data the data points where the steering angle is 0.
            # There are alot of examples in the supplied training data set where it is 0 and this biases the model.
            if float(line[3]) != 0:
                lines.append(line)
    
    images = []
    measurements = []
    for line in lines:
        try:
            filename = line[0].split('/')[-1]
            current_path = 'data/IMG/' + filename
            image = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB)
            images.append(image)
            measurement = float(line[3])
            measurements.append(measurement)
            # Augmenting the data set by adding horizontally flipped images with labeled with inverted steering angles.
            # This evens out the number of left and right steering data points in the data set.
            images.append(cv2.flip(image,1))
            measurements.append(-measurement)
        except Exception as err:
            print("Couldn't process the following line.")
            print(line)
            print('Skipping it.')
        
    print('Loaded {} images with {} steering angle measurements.'.format(len(images), len(measurements)))
    return np.array(images), np.array(measurements)

X_train, y_train = load_data()


## Load Keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Building a sequential Keras model as LeNet is a linear stack of layers.
# https://keras.io/getting-started/sequential-model-guide/
model = Sequential()

# Crop the images.  Just need the model to focus on the image of the road, the rest is extraneous.
# Removes 55 lines from the top to get rid of things above the horizon and 25 line from the bottom to remove the car's hood.  Nothing removed from left or right sides.
model.add(Cropping2D(cropping=((55, 25), (0, 0)), input_shape=(160, 320, 3)))

# Perform data normalization, zero mean and unit variance assuming the images' pixel values have a normal distibution from 0-255.
model.add(Lambda(lambda x: x / 255.0 - 0.5))

# Build a classic LeNet model
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

# Using the Adam optimizer with a Mean Square Error loss function.
model.compile(loss='mse', optimizer='adam')
# Found performance did not improve with more than 6 epochs.
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=6)

# Save the model architecture and its learned weights so it can "drive" the simulated car with drive.py
model.save('model.h5')
