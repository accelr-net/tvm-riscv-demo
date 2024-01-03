# Copyright © 2023 ACCELR

import os
import zipfile
from  urllib import request
import cv2
import numpy as np

class dataloader:
  tinyimagenet_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
  tinyimagenet_path = "./data/imagenet/tiny-imagenet-200.zip"

  def __init__(self):
    pass
    # if not os.path.exists(dataloader.tinyimagenet_path):
    #   request.urlretrieve(dataloader.tinyimagenet_url, dataloader.tinyimagenet_path)

    # if not os.path.exists("./data/imagenet/tiny-imagenet-200"):
    #   with zipfile.ZipFile(dataloader.tinyimagenet_path, 'r') as zip_ref:
    #     zip_ref.extractall("./data/imagenet/")

  def _imagenet_transforms(image: "cv2.typing.MatLike") -> np.ndarray:
    resized_image = cv2.resize(image, (224, 224))
    image_data = np.asarray(resized_image).astype(np.float32)
    if len(image_data.shape) < 3: image_data = np.repeat(image_data[:, :, np.newaxis], 3, axis=-1)
    image_data = np.transpose(image_data, (2, 0, 1))
    imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    norm_image_data = (image_data / 255 - imagenet_mean) / imagenet_stddev
    image_data = np.expand_dims(norm_image_data, axis=0).astype(np.float32)
    return image_data

  def load(self, image_path: str)  -> np.ndarray:
    image_file = cv2.imread(image_path)
    image_data = dataloader._imagenet_transforms(image_file)
    return image_data
