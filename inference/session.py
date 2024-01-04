import time
from tqdm import tqdm

from .models import Imagenet as imagene_tmodel
from .models import KWS as kws_model

from .dataloader.imagenet import ImagenetDataLoader as imagenet_dataloader
from .dataloader.kws import KWSDataLoader as kws_dataloader

from .utils.evaluate import evaluator


class InferenceSession:
  def __init__(self, model: str, arch: str, num_steps: int, pt: bool=False) -> None:
    self.model_name = model
    self.arch = arch
    self.ispytorch = pt
    self.correct = 0
    self.accuracy = 0
    self.evaluator = evaluator(self.arch, self.model_name, self.ispytorch)

    if self.model_name == "imagenet":
      self.dataloader = imagenet_dataloader(int(num_steps))
      if not self.ispytorch:
        self.model = imagene_tmodel(f"./bin/{model}_{arch}.tar", arch)
      else:
        self.model = imagene_tmodel(f"./models/resnet18-v2-7.onnx", arch, pt=self.ispytorch)
    if self.model_name == "kws":
      self.dataloader = kws_dataloader(int(num_steps))
      if not self.ispytorch:
        self.model = kws_model(f"./bin/{model}_{arch}.tar", arch)
      else:
        self.model = kws_model(f"./models/resnet18-kws-best-acc.pt", arch, pt=self.ispytorch)
    
    self.data = self.dataloader.get_data()
    self.num_steps = len(self.data)

  def run(self) -> None:
    for data_idx in tqdm(range(self.num_steps)):
      data_tensor, sample_rate, label, speaker_id, utterance_number = self.data[data_idx]

      start_time = time.time_ns()
      output = self.model.run(data_tensor)
      end_time = time.time_ns()
      elapsed_time_ns = end_time - start_time
      top_five_output = self.dataloader.postprocess(output)

      self.correct = self.correct + 1 if label == top_five_output[1][0].split(" ")[0] else self.correct
      self.evaluator.log(f"{label}_{str(speaker_id)}_{str(utterance_number)}", top_five_output, elapsed_time_ns)

    self.accuracy = self.correct/self.num_steps

  def benchmark(self) -> None:
    self.evaluator.dump(self.num_steps, self.accuracy)

  def evaluate(self) -> None:
    self.evaluator.process(self.num_steps)
