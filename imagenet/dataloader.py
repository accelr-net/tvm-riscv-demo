# Copyright © 2023 ACCELR

import os
import zipfile
from  urllib import request
from PIL import Image
import numpy as np


tinyimagenet_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
tinyimagenet_path = "./data/imagenet/tiny-imagenet-200.zip"


def _imagenet_transforms(image: Image.Image) -> np.ndarray:
  resized_image = image.resize((224, 224))
  image_data = np.asarray(resized_image).astype(np.float32)
  if len(image_data.shape) < 3: image_data = np.repeat(image_data[:, :, np.newaxis], 3, axis=-1)
  image_data = np.transpose(image_data, (2, 0, 1))
  
  imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
  imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
  norm_image_data = (image_data / 255 - imagenet_mean) / imagenet_stddev
  
  image_data = np.expand_dims(norm_image_data, axis=0)

  return image_data  


def preprocess_imagenet_data(num_steps: int) -> None:
  if not os.path.exists(tinyimagenet_path): request.urlretrieve(tinyimagenet_url, tinyimagenet_path)
  
  with zipfile.ZipFile(tinyimagenet_path, 'r') as zip_ref:
    zip_ref.extractall("./data/imagenet/")

  image_dir = "./data/imagenet/tiny-imagenet-200/test/images"
  data_dir = "./data/imagenet/imagetensors/"

  images = os.listdir(image_dir)
  for image_idx in range(num_steps if num_steps >= 0 else len(images)):
    image_path = os.path.join(image_dir, images[image_idx])
    image_file = Image.open(image_path)

    image_data = _imagenet_transforms(image_file)
    
    np.save(data_dir + images[image_idx], image_data)
