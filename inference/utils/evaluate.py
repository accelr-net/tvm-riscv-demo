import json
import os
import numpy as np
from typing import Tuple


class evaluator:
  """
  Evaluator class to log the data for each test case and evaluate the
  functional and performance results at the end to create a unified report.
  """

  def __init__(self, arch: str, model: str, pt: bool=False):
    """
    Initiates an evaluator object.

    Args:
      - arch(str)  : processor architecture
      - model(str) : model name
      - pt(bool)   : is this evaluator is created for a pytorch inference session
    """

    name_tag = "pytorch" if pt else ""
    self.arch = arch
    self.model = model
    self.ispytorch = pt

    # a json file to log results of each test case in the ./tmp directory
    self.log_path = f"./tmp/log_{model}_{self.arch}_{name_tag}.json"
    self.log_json = {}

    # a json file to log the final evaluation report after analyzing all the data logged in to ./tmp directory
    self.report_path = f"./evaluation_report_{self.model}.json"
    self.report_json = {}
    self.step_counter = 0
    self.elapsed_time_ns = 0
    os.makedirs("./tmp", exist_ok=True)

    # creating the files in x86_64 and read the files in riscv64
    if arch == "x86_64":
      json_file = open(self.log_path, 'w')
      json.dump(self.log_json, json_file)
    else:
      json_file = open(self.log_path, 'r')
      self.log_json = json.load(json_file)

  def log(self, key: str, output: Tuple[float, str], elapsed_time_ns: int) -> None:
    """
    Logs the data after performing inference for each test case.

    Args:
      - key(str)                  : a unique key identify each test case
      - output(Tuple[float, str]) : top five output from the model as a tuple of floats and strings
                                    floats  : probability after parsing through a softmax layer
                                    strings : labels
      - elapsed_time_ns(int)      : elapsed time for inference in nanoseconds
    """

    self.step_counter += 1
    if key not in self.log_json:
      self.log_json[key] = {"test_no" : self.step_counter}
    self.log_json[key]["scores"] = [str(i)for i in output[0]]
    self.log_json[key]["labels"] = [str(i)for i in output[1]]
    self.elapsed_time_ns += elapsed_time_ns

  def dump(self, num_steps: int, accuracy: float) -> None:
    """
    Logs the bench mark data (speed and accuracy) for at the end of the execution of each model.

    Args:
      - num_steps(int)  : number of test cases executed
      - accuracy(float) : model accuracy as a float
    """

    self.log_json["inference_speed(fps)"] = round(num_steps * 10e9 / self.elapsed_time_ns, 4),
    self.log_json["model_accuracy"] = round(accuracy * 100, 4)
    json_file = open(self.log_path, 'w')
    json.dump(self.log_json, json_file, indent=4)

  def process(self, num_steps: int) -> None:
    """
    Analizes the logs of each model and creates a final evaluation report after a thorough verification of model outputs.

    Args:
      - num_steps(int)  : number of test cases executed
    """

    # openning the log files for each model
    pytorch = json.load(open(f"./tmp/log_{self.model}_x86_64_pytorch.json", 'r'))
    x86_64 = json.load(open(f"./tmp/log_{self.model}_x86_64_.json", 'r'))
    riscv64 = json.load(open(f"./tmp/log_{self.model}_riscv64_.json", 'r'))

    x86_64_riscv64_passed_high_level_top_one = 0; x86_64_riscv64_failed_high_level_top_one = 0
    pytorch_riscv64_passed_high_level_top_one = 0; pytorch_riscv64_failed_high_level_top_one = 0
    x86_64_riscv64_passed_top_one = 0; x86_64_riscv64_failed_top_one = 0
    pytorch_riscv64_passed_top_one = 0; pytorch_riscv64_failed_top_one = 0
    x86_64_riscv64_passed_high_level_top_five = 0; x86_64_riscv64_failed_high_level_top_five = 0
    pytorch_riscv64_passed_high_level_top_five = 0; pytorch_riscv64_failed_high_level_top_five = 0
    x86_64_riscv64_passed_top_five = 0; x86_64_riscv64_failed_top_five = 0
    pytorch_riscv64_passed_top_five = 0; pytorch_riscv64_failed_top_five = 0

    # iterating through the results of all the model outputs and verifying them
    global_counter = 0
    for key in x86_64:
      if global_counter >= len(x86_64) - 2: break
      global_counter += 1

      x86_64_scores  = np.array(list(map(float, x86_64 [key]["scores"])))
      riscv64_scores = np.array(list(map(float, riscv64[key]["scores"])))
      pytorch_scores = np.array(list(map(float, pytorch[key]["scores"])))
      x86_64_labels  = np.array(list(map(str,   x86_64 [key]["labels"])))
      riscv64_labels = np.array(list(map(str,   riscv64[key]["labels"])))
      pytorch_labels = np.array(list(map(str,   pytorch[key]["labels"])))

      # top one accuracy(TVM on x86 vs TVM on RISC-V)
      if x86_64_labels[0] == riscv64_labels[0]:
        x86_64_riscv64_passed_high_level_top_one = x86_64_riscv64_passed_high_level_top_one + 1
        if np.allclose(np.array(x86_64_scores[0]), np.array(riscv64_scores[0]), rtol=1e-05):
          x86_64_riscv64_passed_top_one = x86_64_riscv64_passed_top_one + 1
        else:
          x86_64_riscv64_failed_top_one = x86_64_riscv64_failed_top_one + 1
      else:
        x86_64_riscv64_failed_high_level_top_one = x86_64_riscv64_failed_high_level_top_one + 1
        x86_64_riscv64_failed_top_one = x86_64_riscv64_failed_top_one + 1

      # top one accuracy(Pytorch on x86 vs TVM on RISC-V)
      if pytorch_labels[0] == riscv64_labels[0]:
        pytorch_riscv64_passed_high_level_top_one = pytorch_riscv64_passed_high_level_top_one + 1
        if np.allclose(np.array(pytorch_scores[0]), np.array(riscv64_scores[0]), rtol=1e-05):
          pytorch_riscv64_passed_top_one = pytorch_riscv64_passed_top_one + 1
        else:
          pytorch_riscv64_failed_top_one = pytorch_riscv64_failed_top_one + 1
      else:
        pytorch_riscv64_failed_high_level_top_one = pytorch_riscv64_failed_high_level_top_one + 1
        pytorch_riscv64_failed_top_one = pytorch_riscv64_failed_top_one + 1

      # top five accuracy(TVM on x86 vs TVM on RISC-V)
      if set(x86_64_labels) == set(riscv64_labels):
        x86_64_riscv64_passed_high_level_top_five = x86_64_riscv64_passed_high_level_top_five + 1
        if np.allclose(x86_64_scores, riscv64_scores, rtol=1e-05):
          x86_64_riscv64_passed_top_five = x86_64_riscv64_passed_top_five + 1
        else:
          x86_64_riscv64_failed_top_five = x86_64_riscv64_failed_top_five + 1
      else:
        x86_64_riscv64_failed_high_level_top_five = x86_64_riscv64_failed_high_level_top_five + 1
        x86_64_riscv64_failed_top_five = x86_64_riscv64_failed_top_five + 1

      # top five accuracy(Pytorch on x86 vs TVM on RISC-V)
      if set(pytorch_labels) == set(riscv64_labels):
        pytorch_riscv64_passed_high_level_top_five = pytorch_riscv64_passed_high_level_top_five + 1
        if np.allclose(pytorch_scores, riscv64_scores, rtol=1e-05):
          pytorch_riscv64_passed_top_five = pytorch_riscv64_passed_top_five + 1
        else:
          pytorch_riscv64_failed_top_five = pytorch_riscv64_failed_top_five + 1
      else:
        pytorch_riscv64_failed_high_level_top_five = pytorch_riscv64_failed_high_level_top_five + 1
        pytorch_riscv64_failed_top_five = pytorch_riscv64_failed_top_five + 1

    x86_64_riscv64_passed_high_level_top_one_accuracy = x86_64_riscv64_passed_high_level_top_one*100/num_steps
    pytorch_riscv64_passed_high_level_top_one_accuracy = pytorch_riscv64_passed_high_level_top_one*100/num_steps
    x86_64_riscv64_passed_high_level_top_five_accuracy = x86_64_riscv64_passed_high_level_top_five*100/num_steps
    pytorch_riscv64_passed_high_level_top_five_accuracy = pytorch_riscv64_passed_high_level_top_five*100/num_steps
    x86_64_riscv64_passed_top_one_accuracy = x86_64_riscv64_passed_top_one*100/num_steps
    pytorch_riscv64_passed_top_one_accuracy = pytorch_riscv64_passed_top_one*100/num_steps
    x86_64_riscv64_passed_top_five_accuracy = x86_64_riscv64_passed_top_five*100/num_steps
    pytorch_riscv64_passed_top_five_accuracy = pytorch_riscv64_passed_top_five*100/num_steps

    self.report_json["model"] = self.model
    self.report_json["test_cases"] = global_counter
    self.report_json["x86_64_riscv64_passed_high_level_top_one_accuracy"] = x86_64_riscv64_passed_high_level_top_one_accuracy
    self.report_json["x86_64_riscv64_passed_top_one_accuracy"] = x86_64_riscv64_passed_top_one_accuracy
    self.report_json["pytorch_riscv64_passed_high_level_top_one_accuracy"] = pytorch_riscv64_passed_high_level_top_one_accuracy
    self.report_json["pytorch_riscv64_passed_top_one_accuracy"] = pytorch_riscv64_passed_top_one_accuracy
    self.report_json["x86_64_riscv64_passed_high_level_top_five_accuracy"] = x86_64_riscv64_passed_high_level_top_five_accuracy
    self.report_json["x86_64_riscv64_passed_top_five_accuracy"] = x86_64_riscv64_passed_top_five_accuracy
    self.report_json["pytorch_riscv64_passed_high_level_top_five_accuracy"] = pytorch_riscv64_passed_high_level_top_five_accuracy
    self.report_json["pytorch_riscv64_passed_top_five_accuracy"] = pytorch_riscv64_passed_top_five_accuracy

    self.report_json["inference_speed(fps)"] = {
      "pytorch": pytorch["inference_speed(fps)"][0],
      "x86_64" : x86_64 ["inference_speed(fps)"][0],
      "riscv64": riscv64["inference_speed(fps)"][0],
    }
    self.report_json["model_accuracy"] = {
      "pytorch": pytorch["model_accuracy"],
      "x86_64" : x86_64 ["model_accuracy"],
      "riscv64": riscv64["model_accuracy"],
    }

    report_file = open(self.report_path, 'w')
    json.dump(self.report_json, report_file, indent=4)
