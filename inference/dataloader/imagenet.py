import os
import tarfile
import glob
import cv2
import numpy as np

from ..utils.helpers import softmax

from typing import List, Tuple


class ImagenetDataLoader:
  def __init__(self, num_steps: int):
    if not os.path.exists("./data/imagenet/imagenet10"):
      with tarfile.open("./data/imagenet/imagenet10.tar.gz", 'r:gz') as tar:
        tar.extractall("./data/imagenet/imagenet10")
    image_dir = "./data/imagenet/imagenet10/val"
    titles = glob.glob(image_dir + '/**/*.JPEG', recursive=True)
    self.processed_dataset = []
    for i in range(num_steps if num_steps < len(titles) else len(titles)):
      title = titles[i].split('/')
      self.processed_dataset.append((ImagenetDataLoader._load(titles[i]), 0, title[-2], 0, title[-1]))

  @staticmethod
  def _imagenet_transforms(image: "cv2.typing.MatLike") -> np.ndarray:
    """
    Performs the standard transforms for imagenet.

    Args:
      - image(cv2.typing.MatLike): input image as a cv2.typing.MatLike object
    """

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

  def postprocess(self, model_output: np.ndarray) -> Tuple[List[float], List[str]]:
    """
    Performs the postprocessing for model outputs.

    Args:
      - model_output(np.ndarray): model output as a numpy array
    """

    labels_path = "./models/synset.txt"
    with open(labels_path, "r") as f:
      labels = [l.rstrip() for l in f]
    scores = softmax(model_output)
    scores = np.squeeze(scores)
    ranks = np.argsort(scores)[::-1]
    top_five_output = ([scores[rank] for rank in ranks[0:5]], [labels[rank] for rank in ranks[0:5]])
    return top_five_output

  def get_data(self) -> List[tuple]:
    return self.processed_dataset
