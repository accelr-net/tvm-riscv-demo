# Copyright © 2023 ACCELR

import tvm
from tvm.contrib import graph_executor
import os
import time
import platform
import numpy as np
from tqdm import tqdm
from  urllib import request
from .evaluate import evaluator
from .dataloader import dataloader

platform_arch = platform.machine().lower()

if platform_arch == "x86_64":
  import torch
  import onnx
  from onnx2pytorch import ConvertModel

labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
labels_path = "./models/synset.txt"

class pytorch_session:
  def __init__(self, model_path: str="./models/resnet18-v2-7.onnx"):
    self.onnx_model = onnx.load(model_path)
    self.model = ConvertModel(self.onnx_model)
    self.model.eval()

  def infer(self, input: np.ndarray) -> np.ndarray:
    with torch.no_grad():
      start_time_pt = time.time_ns()
      output = self.model(torch.tensor(input))
      end_time_pt = time.time_ns()
    elapsed_time_ns = end_time_pt - start_time_pt
    return output.numpy(), elapsed_time_ns

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

def resnet18_session(num_steps: int, verbose_report: bool) -> None:
  eval = evaluator(platform_arch, "./imagenet/log.json")
  dl = dataloader()
  if platform_arch == "x86_64": pt_session = pytorch_session()
  lib_path = "./bin/resnet18_arch.tar".replace("arch", platform_arch)
  pretty_print(f" TVM resnet18 inference session on {platform_arch} ")
  print(f" dynamic loading compiled library from {lib_path}! \n")

  loaded_lib = tvm.runtime.load_module(lib_path)
  runtime_module = graph_executor.GraphModule(loaded_lib["default"](tvm.device("llvm", 0)))
  image_dir = "./data/imagenet/tiny-imagenet-200/test/images"
  images = os.listdir(image_dir)

  for image_idx in tqdm(range(num_steps if num_steps >= 0 else len(images))):
    image_path = os.path.join(image_dir, images[image_idx])
    data_tensor = dl.load(image_path)
    start_time_tvm = time.time_ns()
    runtime_module.set_input("data", data_tensor)
    runtime_module.run()
    output = runtime_module.get_output(0).asnumpy()
    end_time_tvm = time.time_ns()
    elapsed_time_ns_tvm = end_time_tvm - start_time_tvm
    top_five_output_tvm = _postprocess(output)
    eval.log(images[image_idx], top_five_output_tvm, elapsed_time_ns_tvm)

    if platform_arch == "x86_64":
      pytorch_output, elapsed_time_ns_pt = pt_session.infer(data_tensor)
      top_five_output_pytorch = _postprocess(pytorch_output)
      eval.log(images[image_idx], top_five_output_pytorch, elapsed_time_ns_pt, pt=True)
  
  if platform_arch == "riscv64": eval.process(num_steps, verbose_report)
  eval.end(platform_arch, num_steps)
  pretty_print(f" End of TVM resnet18 inference session on {platform_arch} ")
