# -*- coding: utf-8 -*-

import os
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Convolution2D, Dense, Dropout, Flatten

BASEDIR = os.path.expanduser('~/tmp/datasets/sdcnd/behavioral-cloning')
BATCH_SIZE = 32
EPOCHS = 5


def samples(basedir, split=0.2):
    """ Load samples from driving log
    :param basedir: Driving log directory
    :param split: Train/Validation split percentage
    :return: Driving log samples
    """
    items = []
    with open('{}/driving_log.csv'.format(basedir)) as file:
        reader = csv.reader(file)
        next(reader)
        for line in reader:
            items.append(line)
    return train_test_split(items, test_size=split)


def load(basedir, sample):
    """ Load driving log sample data
    :param basedir: Image directory
    :param sample:
    :return: Image, Steering angle
    """
    x = cv2.imread('{}/{}'.format(basedir, sample[0]))
    y = float(sample[3])
    return x, y


def batch(items, basedir, size=32):
    """ Batch generator
    :param items: Driving log samples
    :param basedir: Image directory
    :param size: Batch size
    :return: X, y
    """
    n = len(items)
    while 1:
        shuffle(items)
        for offset in range(0, n, size):
            batch_samples = items[offset:offset + size]
            X, y = zip(*[load(basedir, sample) for sample in batch_samples])
            yield shuffle(np.array(X), np.array(y))


def model(input_shape):
    """ Steering angle prediction model
    CNN based on 'End to End Learning for Self-Driving Cars' arXiv:1604.07316
    :param input_shape: Image shape
    :return: Compiled model
    """
    mdl = Sequential()

    mdl.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    mdl.add(Cropping2D(cropping=((60, 20), (0, 0))))
    mdl.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    mdl.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    mdl.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))

    mdl.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
    mdl.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
    mdl.add(Flatten())

    mdl.add(Dense(100, activation='relu'))
    mdl.add(Dense(50, activation='relu'))
    mdl.add(Dense(10, activation='relu'))
    mdl.add(Dense(1))

    mdl.compile(loss='mse', optimizer='adam')
    return mdl


if __name__ == '__main__':
    """ 
    Train and store the steering angle prediction model 
    """
    train_samples, validation_samples = samples(BASEDIR, 0.1)

    m = model(input_shape=(160, 320, 3))
    m.fit_generator(batch(train_samples, basedir=BASEDIR, size=BATCH_SIZE), samples_per_epoch=len(train_samples),
                    validation_data=batch(validation_samples, basedir=BASEDIR, size=BATCH_SIZE),
                    nb_val_samples=len(validation_samples), nb_epoch=EPOCHS)

    m.save('model.h5')
