# Copyright © 2023 ACCELR

import tvm
from tvm.contrib import graph_executor
import os
import numpy as np
from  urllib import request
from .evaluate import evaluator

labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
labels_path = "./data/synset.txt"

def pretty_print(input: str) -> None:
  pad_length = max(0, 100 - len(input))
  print(f"\n{'-' * (pad_length // 2) + input + '-' * (pad_length // 2 + pad_length % 2)}\n\n")

def softmax(input: np.ndarray) -> np.ndarray:
  shape = input.shape
  input = input.flatten()
  softmax_output = np.exp(input)/sum(np.exp(input))
  return softmax_output.reshape(shape)

def _postprocess(tvm_output: np.ndarray) -> None:
  if not os.path.exists(labels_path): request.urlretrieve(labels_url, labels_path)
  with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]
  # Open the output and read the output tensor
  scores = softmax(tvm_output)
  scores = np.squeeze(scores)
  ranks = np.argsort(scores)[::-1]
  top_five_output = [[scores[rank] for rank in ranks[0:5]], [labels[rank] for rank in ranks[0:5]]]
  return top_five_output

def resnet18_session(arch: str, num_steps: int) -> None:
  eval = evaluator(arch)
  lib_path = "./bin/resnet18_arch.tar".replace("arch", arch)
  pretty_print(f" TVM resnet18 inference session on {arch} ")
  print(f" dynamic loading compiled library from {lib_path}! \n")

  loaded_lib = tvm.runtime.load_module(lib_path)
  runtime_module = graph_executor.GraphModule(loaded_lib["default"](tvm.device(str("llvm"), 0)))
  tensor_dir = "./data/imagenet/imagetensors/"
  tensors = os.listdir(tensor_dir)
  tensors = [tensor for tensor in tensors if tensor != '.gitignore']

  for tensor_idx in range(num_steps if num_steps >= 0 else len(tensors)):
    tensor_path = os.path.join(tensor_dir, tensors[tensor_idx])
    data_tensor = np.load(tensor_path, allow_pickle=True)
    runtime_module.set_input("data", data_tensor)
    runtime_module.run()
    output = runtime_module.get_output(0)
    top_five_output = _postprocess(output.asnumpy())
    eval.log(tensors[tensor_idx], top_five_output)
  
  if arch == "riscv64": eval.process(num_steps)
  eval.end()
  pretty_print(f" End of TVM resnet18 inference session on {arch} ")
