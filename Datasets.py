import tensorflow as tf
import numpy as np
import h5py
ROWS = 64
COLS = 64
batch_size = 8
path = "/Volumes/My Passport/programming/x_test_60Ma.csv"
path_train = "/Volumes/My Passport/programming/Dataset.mat"
path_test = "/Volumes/My Passport/programming/Dataset.mat"


def dataset():
    data = h5py.File(path_train)
    x, y = np.transpose(data['x_train_s']), np.transpose(data['y_train'])
    x = 2.0 * (np.array(x)+11000.0)/20000.0-1.0
    y = 2.0 * (np.array(y)+11000.0)/20000.0-1.0
    x_train, y_train = [], []
    for i, j in zip(x, y):
        img = np.reshape(i, (ROWS, COLS))
        x_train.append(np.array(img))
        img1 = np.reshape(j, (ROWS * 6, COLS * 6))
        y_train.append(np.array(img1))
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train, y_train


def dataset_test():
    data = h5py.File(path_test)
    x, y = np.transpose(data['x_test_s']), np.transpose(data['y_test'])
    x = 2.0 * (np.array(x)+11000.0)/20000.0-1.0
    y = 2.0 * (np.array(y)+11000.0)/20000.0-1.0
    x_test, y_test = [], []
    for i, j in zip(x, y):
        img = np.reshape(i, (ROWS, COLS))
        x_test.append(np.array(img))
        img1 = np.reshape(j, (ROWS * 6, COLS * 6))
        y_test.append(np.array(img1))
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_test, y_test


def dataset_predict():
    x_test1 = np.loadtxt(path, dtype=np.float64, delimiter=',')
    x_test1 = 2.0 * (np.array(x_test1)+11000.0)/20000.0-1.0
    x_test = []
    for i in x_test1:
        img = np.reshape(i, (ROWS, COLS))
        x_test.append(np.array(img))
    x_test = np.array(x_test)
    return x_test
