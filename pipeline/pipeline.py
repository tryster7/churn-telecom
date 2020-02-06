# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Kubeflow Pipelines MNIST example

Run this script to compile pipeline
"""


import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.onprem as onprem

platform = 'GCP'

@dsl.pipeline(
  name='Customer Churn',
  description='Customer Churn'
)
def pipeline(gs_bucket='gs://your-bucket/export', 
		   input_file_with_folder='input/churn.csv',
		   output_folder = 'output',
		   optimizer_name='RMSProp', 
		   learning_rate=0.003,
		   momentum=0.01,
		   model_dir='gs://your-bucket/export', 
		   model_name='dummy',
		   server_name='dummy'):
		   
  preprocess_args = [
  		'--bucket_name', gs_bucket, 
  		'--input_file_with_folder', input_file_with_folder, 
  		'--output', output_folder
  ]
  preprocess = dsl.ContainerOp(
      name='preprocess',
      image='gcr.io/kube-2020/churn/preprocess:latest',
      arguments= preprocess_args
  )


  train_args = [
  		'--bucket_name', gs_bucket, 
  		'--output_folder', output_folder, 
  		'--optimizer_name', optimizer_name,
  		'--learning_rate', learning_rate, 
  		'--momentum' , momentum
  ]
  train = dsl.ContainerOp(
      name='train',
      image='gcr.io/kube-2020/churn/train:latest',
      arguments= train_args
  )

  serve_args = [
      '--model_path', model_dir,
      '--model_name', model_name,
      '--server_name', server_name
  ]

  serve = dsl.ContainerOp(
      name='serve',
      image='gcr.io/kube-2020/churn/pipeline/deployer:latest',
      arguments=serve_args
  )

  steps = [preprocess, train, serve]
  for step in steps:
    if platform == 'GCP':
      step.apply(gcp.use_gcp_secret('user-gcp-sa'))
    else:
      step.apply(onprem.mount_pvc(pvc_name, 'local-storage', '/mnt'))

  train.after(preprocess)
  serve.after(train)

if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(mnist_pipeline, __file__ + '.tar.gz')
