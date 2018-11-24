import tensorflow as tf
import sys
import os

from tf_model import get_model
from tf_util import load_training_data, load_testing_data, get_data

data_location = '../data'
train_data_location = data_location + '/train'
test_data_location = data_location + '/test'

epochs = 1
model_save_location = '../models'

get_data(data_location)
x_train, y_train = load_training_data(train_data_location)
x_test, y_test = load_testing_data(test_data_location)

model = get_model()

model.fit(x_train, y_train, epochs=epochs)
model.evaluate(x_test, y_test)
model.save(os.path.join(model_save_location, 'my_model.h5'))
