import tensorflow as tf
from tensorflow import keras
from tensorflow.python.lib.io import file_io
from tensorflow import feature_column
from google.cloud import storage
import pathlib
from datetime import datetime

import sys
import json
import pandas as pd
import os
import argparse

from datetime import datetime
from uuid import uuid4

# Helper libraries
import numpy as np


ARGS = None

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name',
                        type=str,
                        default='gs://',
                        help='The bucket where the output has to be stored')
    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        help='Number of epochs for training the model')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='the batch size for each epoch')
    parser.add_argument('--output_folder',
                        type=str,
                        default='gs://',
                        help='the batch size for each epoch')
    parser.add_argument('--optimizer_name',
                        type=str,
                        default='SGD',
                        help='Number of epochs for training the model')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.01,
                        help='the batch size for each epoch')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.01,
                        help='the batch size for each epoch')

    return parser


def get_categorical_columns(df):
    unique_vals_list = []
    cat_columns = []
    for col in df.columns:
        if df[col].dtype.name != 'object':
            continue
        unique_vals = df[col].unique()
        unique_vals_list.append(unique_vals)
        cat_columns.append(col)

    cat_cols = [tf.feature_column.categorical_column_with_vocabulary_list(col, vocabulary_list=list)
            for col, list in zip(cat_columns, unique_vals_list)]

    return cat_cols

def get_feature_cols(cat_cols):

    feature_columns = []
    cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    for col in cols:
        feature_columns.append(feature_column.numeric_column(col))

    for col in cat_cols:
        feature_columns.append(feature_column.indicator_column(col))

    return feature_columns


def input_fn(features, labels, training=True, batch_size=256):
    """An input function for training or evaluating"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(500).repeat()

    return dataset.batch(batch_size)

def generate_input_fn(feature_columns):
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec);
    return input_fn


def train(bucket_name, output_folder, epochs=10, batch_size=128, optimizer_name = 'Adam'):
    train_file = bucket_name + os.path.sep +  output_folder + '/train.csv'
    train_labels = bucket_name + os.path.sep + output_folder + '/train_labels.csv'
    test_file = bucket_name + os.path.sep + output_folder + '/test.csv'
    test_labels = bucket_name + os.path.sep + output_folder +  '/test_labels.csv'

    trainDS = pd.read_csv(train_file)
    train_labels = pd.read_csv(train_labels)
    testDS = pd.read_csv(test_file)
    test_labels = pd.read_csv(test_labels)

    exportPath = bucket_name + '/model/export'

    feature_columns = get_feature_cols(get_categorical_columns(trainDS))

    run_config = tf.estimator.RunConfig()

    print("The run config is " )
    print("*************************************" )
    print(run_config.cluster_spec)
    print("The task type is {}".format(run_config.task_type))
    if run_config.task_type == 'worker' : 
        print("The number of worker replies are {}".format(run_config.num_worker_replicas))
        print("The task id is {}".format(run_config.task_id))
    if run_config.task_type == 'ps' :
        print("The number of ps replicas are  {}".format(run_config.num_ps_replicas))

    print("Is Chief {}".format(run_config.is_chief))
    print("Global Id {}".format(run_config.global_id_in_cluster))
    print("*************************************" )


    optimizer = tf.keras.optimizers.get(ARGS.optimizer_name)

    dt = datetime.now()
    print(dt.microsecond)
    p_model_dir = exportPath + '/' + str(dt.microsecond) + '/' + str(run_config.global_id_in_cluster)
    print('Model dir is {}'.format(p_model_dir))

    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        model_dir = p_model_dir,
        hidden_units=[2056, 512, 128, 16], n_classes=2,
        optimizer=optimizer, config=run_config)

    print(classifier.config)

    tf.estimator.train_and_evaluate( classifier,
        train_spec=tf.estimator.TrainSpec(input_fn=lambda: input_fn(trainDS, train_labels, training=True, batch_size=256), max_steps = 100),
        eval_spec=tf.estimator.EvalSpec(input_fn=lambda: input_fn(testDS, test_labels, training=False, batch_size=256))
    )

#    classifier.train(
#        input_fn=lambda: input_fn(trainDS, train_labels, training=True, batch_size=64),
#        steps=500)

    path = exportPath + '/' + run_config.task_type + '/' + str(run_config.global_id_in_cluster)
    classifier.export_saved_model(path, generate_input_fn(feature_columns))


if __name__ == '__main__':
    print("The arguments are ", str(sys.argv))
    if len(sys.argv) < 1:
        print("Usage: train bucket-name epochs batch-size")
        sys.exit(-1)

    parser = parse_arguments()
    ARGS, unknown_args = parser.parse_known_args()

    print(ARGS)
    train(ARGS.bucket_name, ARGS.output_folder, int(ARGS.epochs), int(ARGS.batch_size))

