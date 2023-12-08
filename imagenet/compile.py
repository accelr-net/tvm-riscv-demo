# Copyright © 2023 ACCELR

import onnx
import os
from  urllib import request
import numpy as np

import tvm
import tvm.relay as relay


model_url = "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet18-v2-7.onnx"
model_path  = "./models/resnet18-v2-7.onnx"

labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
labels_path = "./models/synset.txt"


def resnet18(target: tvm.target.Target) -> tvm.runtime.Module:
  if not os.path.exists(model_path): request.urlretrieve(model_url, model_path)
  if not os.path.exists(labels_path): request.urlretrieve(labels_url, labels_path)
  onnx_model = onnx.load(model_path)

  image_data = np.expand_dims(np.transpose(np.random.rand(224, 224, 3).astype(np.float32) * 255.0, (2, 0, 1)), axis=0)
  shape_dict = {"data": image_data.shape}

  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
  with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

  return lib
