import time
from tqdm import tqdm

from .models import Imagenet as imagenet_model
from .models import KWS as kws_model

from .dataloader.imagenet import ImagenetDataLoader as imagenet_dataloader
from .dataloader.kws import KWSDataLoader as kws_dataloader

from .utils.evaluate import evaluator


class InferenceSession:
  """
  Inference session:
  - Provides a factory to create inference sessions from models.
  - Creates corresponding dataloader bojects.
  - Executes the models using a unified method.
  - Calls the evaluator to evaluate the resilts.
  """

  def __init__(self, model: str, arch: str, num_steps: int, pt: bool=False):
    """
    initiates the inference session.

    Args:
      model(str)      : model name
      arch(str)       : processor architecture
      num_steps(int)  : number of test cases the session is planning to be executed
      pt(bool)        : is this a pytorch inference session
    """

    self.model_name = model
    self.arch = arch
    self.ispytorch = pt
    self.correct = 0
    self.accuracy = 0
    self.evaluator = evaluator(self.arch, self.model_name, self.ispytorch)

    if self.model_name == "imagenet":
      self.dataloader = imagenet_dataloader(int(num_steps))
      if not self.ispytorch:
        self.model = imagenet_model(f"./bin/{model}_{arch}.tar", arch)
      else:
        self.model = imagenet_model(f"./models/resnet18-v2-7.onnx", arch, pt=self.ispytorch)
    if self.model_name == "kws":
      self.dataloader = kws_dataloader(int(num_steps))
      if not self.ispytorch:
        self.model = kws_model(f"./bin/{model}_{arch}.tar", arch)
      else:
        self.model = kws_model(f"./models/resnet18-kws-best-acc.pt", arch, pt=self.ispytorch)
    
    self.data = self.dataloader.get_data()
    self.num_steps = len(self.data)

  def run(self) -> None:
    """
    Executes the inference session for {self.num_steps}/(length of the dataset) times.
    """

    for data_idx in tqdm(range(self.num_steps)):
      # both data loaders returns a tuple of five containing data and other attributes
      data_tensor, attribute2, label, attribute4, attribute5 = self.data[data_idx]

      # measuring the time around the API call to the inference library
      start_time = time.time_ns()
      output = self.model.run(data_tensor)
      end_time = time.time_ns()
      elapsed_time_ns = end_time - start_time
      top_five_output = self.dataloader.postprocess(output)

      # asserting the output with the label
      self.correct = self.correct + 1 if label == top_five_output[1][0].split(" ")[0] else self.correct
      # logging the accuracy and elapsed time for each test case
      self.evaluator.log(f"{label}_{str(attribute4)}_{str(attribute5)}", top_five_output, elapsed_time_ns)

    self.accuracy = self.correct/self.num_steps

  def benchmark(self) -> None:
    """
    calculates the bench mark data after each inference session.
    """
    self.evaluator.dump(self.num_steps, self.accuracy)

  def evaluate(self) -> None:
    """
    evalutes the bench mark data after the end of all sessions.
    """
    self.evaluator.process(self.num_steps)
