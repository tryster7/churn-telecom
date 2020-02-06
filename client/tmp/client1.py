import sys
import json
import requests
from keras.datasets import fashion_mnist


def get_prediction(server_host='127.0.0.1', server_port=9000):
    (trainX, trainy), (testX, testy) = fashion_mnist.load_data()
    testX = testX / 255
    testX = testX.reshape(testX.shape[0], 28, 28, 1)
    data = json.dumps({"signature_name": "serving_default", "instances": testX[0:3].tolist()})
    print(data)
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://' + server_host + ':' + str(server_port) + '/v1/models/demo:predict', data=data, headers=headers)
    print(json_response)
    print(json.loads(json_response.text))
    #predictions = json.loads(json_response.text['predictions')


if __name__ == '__main__':
    if len(sys.argv) != 3:
       print("Usage: client server_host server_port ")
       sys.exit(-1)
    server_host = sys.argv[1]
    server_port = int(sys.argv[2])
    get_prediction(server_host=server_host, server_port=server_port)

