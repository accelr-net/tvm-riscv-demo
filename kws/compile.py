# Copyright © 2023 ACCELR

import torch
from .dataloader import get_shape
import tvm
from tvm import relay

def kwsrn18(target: tvm.target.Target) -> tvm.runtime.Module:
  try:
    model = torch.jit.load("./models/resnet18-kws-best-acc.pt")
  except Exception as e:
    print(f" error loading the model: {e} \n")

  shape_list = [("data", get_shape())]
  mod, params = relay.frontend.from_pytorch(model, shape_list)
  with tvm.transform.PassContext(opt_level=3, config={}):
    lib = relay.build(mod, target=target, params=params)
  return lib
