
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import requests
import json
import os
import subprocess
import time

from tensorflow_serving.apis import prediction_service_pb2_grpc


def get_prediction(server_host='127.0.0.1', server_port=9000, name='demo'):

    channel = implementations.insecure_channel(server_host, server_port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    inputs = pd.read('gs://kube-1122/t-churn/output/test.csv')

    print(inputs)
    headers = {"content-type": "application/json"}
    json_response = requests.post("http://" + server_host + ":" + str(server_port) + "/v1/models/" + name + ":predict", data=data, headers=headers)
    print(json_response)


if __name__ == "__main__":
    print("Usage: client server_host server_port model_name ")
    server_host = sys.argv[1]
    server_port = int(sys.argv[2])
    model_name = sys.argv[3]

    print("***********************")
    print(server_host, server_port, model_name)
    print("***********************")

    get_prediction(server_host=server_host, server_port=server_port, name=model_name)
