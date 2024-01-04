import os

import torch
import onnx

import tvm
from tvm import relay

from typing import Dict


class Compiler:
  devices: Dict[str, tvm.target.Target] = {
    "riscv64" : tvm.target.Target("llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=generic-rv64 -mabi=lp64d -mattr=+64bit,+m,+f,+d"),
    "x86_64"  : tvm.target.Target("llvm")
  }

  def __init__(self, platform: str) -> None:
    self.platform = platform
    self.device = Compiler.devices[platform]
  
  @staticmethod
  def _imagenet_resnet18(target: tvm.target.Target) -> tvm.runtime.Module:
    try:
      model = onnx.load("./models/resnet18-v2-7.onnx")
    except Exception as e:
      print(f" error loading the model: {e} \n")

    input_shape = (1, 3, 224, 224)
    shape_dict = {"data": input_shape}
    mod, params = relay.frontend.from_onnx(model, shape_dict)
    with tvm.transform.PassContext(opt_level=3, config={}):
      lib = relay.build(mod, target=target, params=params)
    return lib

  @staticmethod
  def _kws_resnet18(target: tvm.target.Target) -> tvm.runtime.Module:
    try:
      model = torch.jit.load("./models/resnet18-kws-best-acc.pt")
    except Exception as e:
      print(f" error loading the model: {e} \n")

    input_shape = (1, 1, 128, 32)
    shape_list = [("data", input_shape)]
    mod, params = relay.frontend.from_pytorch(model, shape_list)
    with tvm.transform.PassContext(opt_level=3, config={}):
      lib = relay.build(mod, target=target, params=params)
    return lib

  def compile(self,  model: str) -> None:
    binary_path = f"./bin/{model}_{self.platform}.tar"
    if model == "imagenet":
      if not os.path.exists(binary_path):
        imagenet_module = Compiler._imagenet_resnet18(self.device)
        imagenet_module.export_library(binary_path)
    if model == "kws":
      if not os.path.exists(binary_path):
        kws_module = Compiler._kws_resnet18(self.device)
        kws_module.export_library(binary_path)
