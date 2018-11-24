import numpy as np
import os
import tensorflow as tf

def get_data(base_dir):
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    np.save(os.path.join(base_dir, 'train', 'x_train.npy'), x_train)
    np.save(os.path.join(base_dir, 'train', 'y_train.npy'), y_train)
    np.save(os.path.join(base_dir, 'test', 'x_test.npy'), x_test)
    np.save(os.path.join(base_dir, 'test', 'y_test.npy'), y_test)

def load_training_data(base_dir):
    x_train = np.load(os.path.join(base_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(base_dir, 'y_train.npy'))
    return x_train, y_train

def load_testing_data(base_dir):
    x_test = np.load(os.path.join(base_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(base_dir, 'y_test.npy'))
    return x_test, y_test