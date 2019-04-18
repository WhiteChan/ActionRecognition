# import csv 
# import numpy as np 
# import random
# import glob
# import os.path 
# import pandas as pd 
# import sys 
# import operator
# from processor import process_image
# from keras.utils import np_utils

# seq_length = 40

# # get data
# with open('./data/data_file.csv', 'r') as fin:
#     reader = csv.reader(fin)
#     data = list(reader)
# print(np.shape(data))

# # get classes
# classes = []
# for item in data:
#     if item[1] not in classes:
#         classes.append(item[1])

# classes = sorted(classes)

# # clean data
# data_clean = []
# for item in data:
#     if int(item[3]) >= seq_length and int(item[3]) <= 300 and item[1] in classes:
#         data_clean.append(item)

# data = data_clean
# print(np.shape(data))

# # split train test
# train = []
# test = []
# for item in data:
#     if item[0] == 'train':
#         train.append(item)
#     else:
#         test.append(item)

# print('train shape: ', np.shape(train))
# print('test shape', np.shape(test))
# print(train[0])
# print(test[0])

from UCFdata import DataSet 
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D, Conv3D
from keras.utils import np_utils
import cv2 as cv 

data = DataSet()

train_data = []
train_label = []
all_labels = range(101)

def read_video_data(cap):
    data = []
    for _ in range(100):
        ret, frame = cap.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            data.append(frame)
        else:
            data.append(np.zeros(shape=(240, 320)))
    return data

all_labels = range(101)

for i in range(10):
    filename = 'data/' + data.data[i][0] + '/' + data.data[i][1] + '/' + data.data[i][2] + '.avi'
    cap = cv.VideoCapture(filename)
    if data.data[i][0] == 'train':
        train_data.append(read_video_data(cap))
        train_label.append(all_labels[data.classes.index(data.data[i][1])])

print(np.shape(train_data))

train_data = np.array(train_data)
train_data = train_data.reshape([10, 100, 240, 320, 1])
train_label = np.array(train_label)
train_label_OneHot = np_utils.to_categorical(train_label, num_classes=101)

model = Sequential()
model.add(Conv3D(filters=1, kernel_size=[9, 9, 9], strides=[3, 3, 3], input_shape=(100, 240, 320, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(units=101, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

train_history = model.fit(x=train_data, y=train_label_OneHot, validation_split=0.1, epochs=1)

# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         horizontal_flip=True,
#         rotation_range=10.,
#         width_shift_range=0.2,
#         height_shift_range=0.2)

# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     './data/train/',
#     target_size=(299, 299),
#     batch_size=32,
#     classes=data.classes,
#     class_mode='categorical')

# validation_generator = test_datagen.flow_from_directory(
#     './data/test/',
#     target_size=(299, 299),
#     batch_size=32,
#     classes=data.classes,
#     class_mode='categorical')

# model = Sequential()
# model.add(Conv2D(filters=8, kernel_size=(5, 5), padding='valid', input_shape=(299, 299, 3), activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation = 'relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Flatten())
# model.add(Dense(101, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# try:
#     model.load_weights("SaveModel/UCF101CNNModel.h5")
#     print("Model loaded successfully! Continuing training model")
# except:
#     print("Failed to load model! Start training a new model")

# print(model.summary())

# train_history = model.fit_generator(train_generator, steps_per_epoch=5500, validation_data=validation_generator, validation_steps=10, epochs=10)

# model.save_weights("SaveModel/cifarCnnModel.h5")
# print("Saved model to disk")