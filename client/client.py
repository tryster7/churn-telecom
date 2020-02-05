import sys
import json
import requests
from keras.datasets import fashion_mnist


def get_prediction(server_host='127.0.0.1', server_port=9000, name='demo'):
    #data = json.dumps({"signature_name": "serving_default", "instances": testX[0:3].tolist()})
        #"examples" : [{"tenure": 0.263889, "MonthlyCharges": 0.397512, "TotalCharges": 0.120354, "SeniorCitizen" : "No"}]
    my_json = {
            "examples" : [{
                "tenure": 0.263889, 
                "MonthlyCharges": 0.397512, 
                "TotalCharges": 0.120354, "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "No", 
                "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "DSL", "OnlineSecurity": "No", 
                "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", 
                "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check" }]
    }
    data = json.dumps(my_json)
    print(data)
    headers = {"content-type": "application/json"}
    json_response = requests.post("http://" + server_host + ":" + str(server_port) + "/v1/models/" + name + ":classify", data=data, headers=headers)
    print(json_response)
    print(json_response.text)


if __name__ == "__main__":
    print("Usage: client server_host server_port model_name ")
    server_host = sys.argv[1]
    server_port = int(sys.argv[2])
    model_name = sys.argv[3]

    print("***********************")
    print(server_host, server_port, model_name)
    print("***********************")

    get_prediction(server_host=server_host, server_port=server_port, name=model_name)
