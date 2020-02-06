#!/bin/bash -e


set -x

while (($#)); do
   case $1 in
     "--model_path")
       shift
       MODEL_PATH="$1"
       shift
       ;;
     "--model_name")
       shift
       MODEL_NAME="$1"
       shift
       ;;
     "--server_name")
       shift
       SERVER_NAME="$1"
       shift
       ;;
     *)
       echo "Unknown argument: '$1'"
       exit 1
       ;;
   esac
done

echo "About to run kubectl command"

echo "model name is $MODEL_NAME, model path is $MODEL_PATH and service name is  $SERVER_NAME"

sed -i "s/SERVER_NAME/$SERVER_NAME/g" /tmp/tfserve.yaml 
sed -i "s/MODEL_NAME/$MODEL_NAME/g" /tmp/tfserve.yaml 
sed -i "s+MODEL_PATH+$MODEL_PATH+g" /tmp/tfserve.yaml 

kubectl apply -f /tmp/tfserve.yaml

echo "After kubectl command"
