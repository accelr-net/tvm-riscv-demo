import os
import tarfile
import glob
import cv2
import numpy as np

from ..utils.helpers import softmax

from typing import List, Tuple


class ImagenetDataLoader:
  def __init__(self, num_steps: int) ->  None:
    if not os.path.exists("./data/imagenet/valset"):
      with tarfile.open("./data/imagenet/valset.tar.xz", 'r:xz') as tar:
        tar.extractall("./data/imagenet/valset")
    image_dir = "./data/imagenet/valset/val"
    image_titles = glob.glob(image_dir + '/**/*.JPEG', recursive=True)
    self.data = []
    for i in range(num_steps if num_steps < len(image_titles) else len(image_titles)):
      self.data.append((ImagenetDataLoader._load(image_titles[i]), 0, image_titles[i].split('/')[5], 0, image_titles[i].split('/')[6]))

  @staticmethod
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

  @staticmethod
  def _load(image_path: str)  -> np.ndarray:
    image_file = cv2.imread(image_path)
    image_data = ImagenetDataLoader._imagenet_transforms(image_file)
    return image_data

  def postprocess(self, tvm_output: np.ndarray) -> Tuple[List[float], List[str]]:
    labels_path = "./models/synset.txt"
    with open(labels_path, "r") as f:
      labels = [l.rstrip() for l in f]
    scores = softmax(tvm_output)
    scores = np.squeeze(scores)
    ranks = np.argsort(scores)[::-1]
    top_five_output = ([scores[rank] for rank in ranks[0:5]], [labels[rank] for rank in ranks[0:5]])
    return top_five_output

  def get_data(self) -> List[tuple]:
    return self.data
