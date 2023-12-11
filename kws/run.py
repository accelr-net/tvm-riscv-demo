# Copyright © 2023 ACCELR

import tvm
from tvm.contrib import graph_executor
import os
import numpy as np
from .evaluate import evaluator

def pretty_print(input: str) -> None:
  pad_length = max(0, 100 - len(input))
  print(f"\n{'-' * (pad_length // 2) + input + '-' * (pad_length // 2 + pad_length % 2)}\n\n")

def kws_session(arch: str, num_steps: int) -> None:
  eval = evaluator(arch)
  lib_path = "./bin/kws_arch.tar".replace("arch", arch)
  pretty_print(f" TVM kws inference session on {arch} ")
  print(f" dynamic loading compiled library from {lib_path}! \n")

  loaded_lib = tvm.runtime.load_module(lib_path)
  runtime_module = graph_executor.GraphModule(loaded_lib["default"](tvm.device(str("llvm"), 0)))
  tensor_dir = "./data/speechcommands/speechtensors/"
  tensors = os.listdir(tensor_dir)
  tensors = [tensor for tensor in tensors if tensor != '.gitignore']

  for tensor_idx in range(num_steps if num_steps >= 0 else len(tensors)):
    tensor_path = os.path.join(tensor_dir, tensors[tensor_idx])
    data_tensor = np.load(tensor_path, allow_pickle=True)
    runtime_module.set_input("data", data_tensor)
    runtime_module.run()
    output = runtime_module.get_output(0)
    eval.log(tensors[tensor_idx], output.numpy()[0])
  
  if arch == "riscv64": eval.process(num_steps)
  eval.end()
  pretty_print(f" End of TVM kws inference session on {arch} ")