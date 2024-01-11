import tvm
from tvm.contrib import graph_executor

import torch
import numpy as np


class Model:
  """
  Abstract base class for defining a model.
  """

  def __init__(self, model_path: str, architecture: str, pt: bool=False):
    self.model_path = model_path
    self.architecture = architecture
    self.ispytorch = pt

  def run(self, input: np.ndarray) -> np.ndarray:
    raise NotImplementedError


class Imagenet(Model):
  """
  Imagenet model, extends model.
  Contains inference pipeline for both tvm and pytorch sessions.
  """

  def __init__(self, model_path: str, architecture: str, pt: bool=False):
    super().__init__(model_path, architecture, pt)
    if not self.ispytorch:
      loaded_lib = tvm.runtime.load_module(self.model_path)
      self.runtime_module = graph_executor.GraphModule(loaded_lib["default"](tvm.device("llvm", 0)))
    else:
      import onnx
      from onnx2pytorch import ConvertModel
      self.onnx_model = onnx.load(self.model_path)
      self.model = ConvertModel(self.onnx_model)
      self.model.eval()

  def run(self, input: np.ndarray) -> np.ndarray:
    if not self.ispytorch:
      self.runtime_module.set_input("data", input)
      self.runtime_module.run()
      output = self.runtime_module.get_output(0).asnumpy()
    else:
      with torch.no_grad():
        output = self.model(torch.tensor(input)).numpy()
    return output


class KWS(Model):
  """
  KWS model, extends model.
  Contains inference pipeline for both tvm and pytorch sessions.
  """

  def __init__(self, model_path: str, architecture: str, pt: bool=False):
    super().__init__(model_path, architecture, pt)
    if not self.ispytorch:
      loaded_lib = tvm.runtime.load_module(self.model_path)
      self.runtime_module = graph_executor.GraphModule(loaded_lib["default"](tvm.device("llvm", 0)))
    else:
      self.model = torch.jit.load(self.model_path)
      self.model.eval()

  def run(self, input: np.ndarray) -> np.ndarray:
    if not self.ispytorch:
      self.runtime_module.set_input("data", input)
      self.runtime_module.run()
      output = self.runtime_module.get_output(0).asnumpy()
    else:
      with torch.no_grad():
        output = self.model(torch.tensor(input)).numpy()
    return output
