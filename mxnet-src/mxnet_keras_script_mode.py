'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function

import argparse
import os

import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from mxnet.contrib import onnx as onnx_mxnet
import numpy as np

from mxnet_util import get_data
from mxnet_model import get_model

def main(batch_size, epochs, num_classes, training_channel, model_dir, dropout):
    x_train, y_train, x_test, y_test = get_data(num_classes, training_channel)
    
    input_shape = x_train[0].shape
        
    model = get_model(input_shape, num_classes, dropout)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print('Saving model in MXNet format')
    keras.models.save_mxnet_model(model=model, prefix=os.path.join(model_dir, 'model'), epoch=0)

    print('Exporting model to ONNX format')
    input_shape = (128, 28, 28, 1)
    sym = os.path.join(model_dir, 'model-symbol.json')
    params = os.path.join(model_dir, 'model-0000.params')
    onnx_file = os.path.join(model_dir, 'exported-model.onnx')
    onnx_mxnet.export_model(sym, params, [input_shape], np.float32, onnx_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--num_classes', type=float, default=12)
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    main(args.batch_size, args.epochs, args.num_classes, args.train, args.model_dir, args.dropout)

