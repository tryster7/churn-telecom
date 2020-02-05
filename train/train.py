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

    args = parser.parse_known_args()[0]
    return args


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
    tf.saved_model.save(classifier, export_dir=export_path)


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


def save_metric_metadata(exec, model, test_acc, test_loss):
    # Save evaluation
    metrics = exec.log_output(
        metadata.Metrics(
            name="MNIST-evaluation",
            description="validating the MNIST model to recognize images",
            owner="demo@kubeflow.org",
            uri="gs://a-kb-poc-262417/mnist/metadata/mnist-metric.csv",
            model_id=str(model.id),
            metrics_type=metadata.Metrics.VALIDATION,
            values={"accuracy": str(test_acc),
                    "test_loss": str(test_loss)},
            labels={"mylabel": "l1"}))
    print("Metrics id is %s" % metrics.id)


def save_model_metadata(exec, batch_size, epochs):
    # Save model;
    model_version = "model_version_" + str(uuid4())
    model = exec.log_output(
        metadata.Model(
            name="MNIST",
            description="model to recognize images",
            owner="demo@kubeflow.org",
            uri="gs://a-kb-poc-262417/mnist/export/model",
            model_type="CNN",
            training_framework={
                "name": "tensorflow",
                "version": "v2.0"
            },
            hyperparameters={
                "learning_rate": 0.5,
                "layers": [28, 28, 1],
                "epochs": str(epochs),
                "batch-size": str(batch_size),
                "early_stop": True
            },
            version=model_version,
            labels={"tag": "train"}))
    print(model)
    print("\nModel id is {0.id} and version is {0.version}".format(model))
    return model


def create_metadata_execution():
    global metadata
    # Create Metadata Workspace and a Exec to log details
    mnist_train_workspace = metadata.Workspace(
        # Connect to metadata service in namespace kubeflow in k8s cluster.
        store=metadata.Store(grpc_host=METADATA_STORE_HOST, grpc_port=METADATA_STORE_PORT),
        name="mnist train workspace",
        description="a workspace for training mnist",
        labels={"n1": "v1"})
    run1 = metadata.Run(
        workspace=mnist_train_workspace,
        name="run-" + datetime.utcnow().isoformat("T"),
        description="a run in ws_1")
    exec = metadata.Execution(
        name="execution" + datetime.utcnow().isoformat("T"),
        workspace=mnist_train_workspace,
        run=run1,
        description="execution example")
    print("An execution was created with id %s" % exec.id)
    return exec


def train(bucket_name, output_folder, epochs=10, batch_size=128):
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

    classifier = create_tfmodel(feature_columns, tf.keras.optimizers.RMSprop, )

    classifier.train(
        input_fn=lambda: input_fn(trainDS, train_labels, training=True, batch_size=64),
        steps=500)

    metrics = classifier.evaluate(
        input_fn=lambda: input_fn(testDS, test_labels, training=True),
        steps = 100)

    print(metrics)

    save_tfmodel_in_gcs(classifier, exportPath, generate_input_fn(feature_columns))

if __name__ == '__main__':
    print("The arguments are ", str(sys.argv))
    if len(sys.argv) < 1:
        print("Usage: train bucket-name epochs batch-size")
        sys.exit(-1)

    args = parse_arguments()
    print(args)
    train(args.bucket_name, args.output_folder, int(args.epochs), int(args.batch_size))

