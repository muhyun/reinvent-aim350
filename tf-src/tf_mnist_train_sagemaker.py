import tensorflow as tf
import argparse
import os
import numpy as np
import sys

from tf_util import load_training_data, load_testing_data
from tf_model import get_model

# Acquire hyperparameters and directory locations passed by SageMaker
def parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=1)
    
    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TESTING'])
    
    return parser.parse_known_args()

if __name__ == "__main__":
    
    args, unknown = parse_args()
    
    print(args)

    x_train, y_train = load_training_data(args.train)
    x_test, y_test = load_testing_data(args.test)
    
    model = get_model(x_train[0].shape)


    model.fit(x_train, y_train, epochs=args.epochs, verbose=2)
    model.evaluate(x_test, y_test, verbose=2)
    print('------ save model to {}'.format(os.path.join(args.model_dir, 'my_model.h5')))
    model.save(os.path.join(args.model_dir, 'my_model.h5'))
