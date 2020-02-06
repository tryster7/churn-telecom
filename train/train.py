import tensorflow as tf
from tensorflow import keras
from tensorflow.python.lib.io import file_io
from tensorflow import feature_column
from google.cloud import storage
import pathlib

import sys
import json
import pandas as pd
import os
import argparse

from sklearn.metrics import confusion_matrix

from keras.datasets import fashion_mnist

from kubeflow.metadata import metadata
from datetime import datetime
from uuid import uuid4

# Helper libraries
import numpy as np

METADATA_STORE_HOST = "metadata-grpc-service.kubeflow"  # default DNS of Kubeflow Metadata gRPC serivce.
METADATA_STORE_PORT = 8080

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

def save_tfmodel_in_gcs(classifier, export_path, input_receiver_fn):
    classifier.export_saved_model(export_path, input_receiver_fn)


def create_tfmodel(feature_columns, optimizer):
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[2056, 512, 128, 16], n_classes=2,
        optimizer=optimizer)
    return classifier
  
def create_kf_visualization(bucket_name, df, test_acc):
    metrics = {
        'metrics': [{
            'name': 'accuracy-score',
            'numberValue': str(test_acc),
            'format': "PERCENTAGE"
        }]
    }

    with file_io.FileIO('/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)

    vocab = list(df['target'].unique())
    cm = confusion_matrix(df['target'], df['predicted'], labels=vocab)
    data = []
    for target_index, target_row in enumerate(cm):
        for predicted_index, count in enumerate(target_row):
            data.append((vocab[target_index], vocab[predicted_index], count))
    df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
    cm_file = bucket_name + '/metadata/cm.csv'

    with file_io.FileIO(cm_file, 'w') as f:
        df_cm.to_csv(f, columns=['target', 'predicted', 'count'], header=False, index=False)

    print("***************************************")
    print("Writing the confusion matrix to ", cm_file)
    metadata = {
        'outputs': [{
            'type': 'confusion_matrix',
            'format': 'csv',
            'schema': [
                {'name': 'target', 'type': 'CATEGORY'},
                {'name': 'predicted', 'type': 'CATEGORY'},
                {'name': 'count', 'type': 'NUMBER'},
            ],
            'source': cm_file,
            'labels': list(map(str, vocab)),
        }]
    }

    with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)


def train(bucket_name, output_folder, epochs=10, batch_size=128, optimizer_name = 'Adam'):
    train_file = bucket_name + os.path.sep +  output_folder + '/train.csv'
    train_labels = bucket_name + os.path.sep + output_folder + '/train_labels.csv'
    test_file = bucket_name + os.path.sep + output_folder + '/test.csv'
    test_labels = bucket_name + os.path.sep + output_folder +  '/test_labels.csv'

    trainDS = pd.read_csv(train_file)
    print(trainDS.head())
    print(train_labels)
    train_labels = pd.read_csv(train_labels)
    testDS = pd.read_csv(test_file)
    test_labels = pd.read_csv(test_labels)

    exportPath = bucket_name + '/model/export'

    feature_columns = get_feature_cols(get_categorical_columns(trainDS))

    optimizer = tf.keras.optimizers.get(ARGS.optimizer_name)
    classifier = create_tfmodel(feature_columns, optimizer)

    classifier.train(
        input_fn=lambda: input_fn(trainDS, train_labels, training=True, batch_size=64),
        steps=500)

    metrics = classifier.evaluate(
        input_fn=lambda: input_fn(testDS, test_labels, training=False),
        steps = 100)

    print(metrics)
    print("accuracy={}".format(metrics['accuracy']))

    save_tfmodel_in_gcs(classifier, exportPath, generate_input_fn(feature_columns))

    predictions = classifier.predict(input_fn=lambda: input_fn(testDS, test_labels, training=False))

    result = []
    for prediction in predictions :
        result.append(prediction['class_ids'][0])

    df1 = pd.DataFrame(data = result, columns=['predicted'])
    test_labels.rename(columns = {'Churn':'target'}, inplace = True)

    df = pd.concat([df1, test_labels], axis=1)
    print(df)
    create_kf_visualization(bucket_name, df, metrics['accuracy'])

if __name__ == '__main__':
    print("The arguments are ", str(sys.argv))
    if len(sys.argv) < 1:
        print("Usage: train bucket-name epochs batch-size")
        sys.exit(-1)

    parser = parse_arguments()
    ARGS, unknown_args = parser.parse_known_args()

    print(ARGS)
    train(ARGS.bucket_name, ARGS.output_folder, int(ARGS.epochs), int(ARGS.batch_size))

