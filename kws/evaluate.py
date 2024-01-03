# Copyright © 2023 ACCELR

import json
import numpy as np
from typing import Tuple

class evaluator:
  def __init__(self, arch: str, json_path: str):
    self.arch = arch; self.output_json = {}
    self.json_path = json_path
    self.step_counter = 0
    self.elapsed_time_ns_tvm = 0
    self.elapsed_time_ns_pt = 0
    try:
      if arch == "x86_64":
        json_file = open(self.json_path, 'w')
        json.dump({}, json_file)
      else:
        json_file = open(self.json_path, 'r')
        self.output_json = json.load(json_file)
    except FileNotFoundError:
      json_file = open(self.json_path, 'w')
      json.dump({}, json_file)

  def log(self, key: str, output: Tuple[float, str], elapsed_time_ns , pt: bool=False) -> None:
    if self.arch == "x86_64" and pt is False:
      self.step_counter += 1
    if key not in self.output_json:
      self.output_json[key] = {
        "test_no" : self.step_counter,
        self.arch: {
          "scores" : [str(i)for i in output[0]],
          "labels" : [str(i)for i in output[1]]
        }
      }
    else:
      self.output_json[key]["pytorch" if pt else self.arch] = {
        "scores" : [str(i)for i in output[0]],
        "labels" : [str(i)for i in output[1]]
      }
    if pt:
      self.elapsed_time_ns_pt += elapsed_time_ns
    else:
      self.elapsed_time_ns_tvm += elapsed_time_ns

  def process(self, num_steps: int, verbose_report: bool) -> None:
    print(" starting verification ... \n")
    print(f" {num_steps} verification test case(s) detected ... \n")

    x86_64_riscv64_passed_high_level_top_one = 0; x86_64_riscv64_failed_high_level_top_one = 0
    pytorch_riscv64_passed_high_level_top_one = 0; pytorch_riscv64_failed_high_level_top_one = 0
    x86_64_riscv64_failed_high_level_top_one_items = []; pytorch_riscv64_failed_high_level_top_one_items = [];

    x86_64_riscv64_passed_top_one = 0; x86_64_riscv64_failed_top_one = 0
    pytorch_riscv64_passed_top_one = 0; pytorch_riscv64_failed_top_one = 0
    x86_64_riscv64_failed_top_one_items = []; pytorch_riscv64_failed_top_one_items = [];

    x86_64_riscv64_passed_high_level_top_five = 0; x86_64_riscv64_failed_high_level_top_five = 0
    pytorch_riscv64_passed_high_level_top_five = 0; pytorch_riscv64_failed_high_level_top_five = 0
    x86_64_riscv64_failed_high_level_top_five_items = []; pytorch_riscv64_failed_high_level_top_five_items = [];

    x86_64_riscv64_passed_top_five = 0; x86_64_riscv64_failed_top_five = 0
    pytorch_riscv64_passed_top_five = 0; pytorch_riscv64_failed_top_five = 0
    x86_64_riscv64_failed_top_five_items = []; pytorch_riscv64_failed_top_five_items = [];

    local_step_counter = 0
    for key in self.output_json:
      local_step_counter += 1
      if local_step_counter > num_steps: break
      if self.output_json[key].get("x86_64") is None:
        print(" run the test in x86_64 first ... \n")
        return

      x86_64_scores = np.array(list(map(float, self.output_json[key]["x86_64"]["scores"])))
      riscv64_scores = np.array(list(map(float, self.output_json[key]["riscv64"]["scores"])))
      pytorch_scores = np.array(list(map(float, self.output_json[key]["pytorch"]["scores"])))
      x86_64_labels = np.array(list(map(str, self.output_json[key]["x86_64"]["labels"])))
      riscv64_labels = np.array(list(map(str, self.output_json[key]["riscv64"]["labels"])))
      pytorch_labels = np.array(list(map(str, self.output_json[key]["pytorch"]["labels"])))

      if x86_64_labels[0] == riscv64_labels[0]:
        self.output_json[key]["x86_64_riscv64_passed_high_level_top_one"] = "passed"
        x86_64_riscv64_passed_high_level_top_one = x86_64_riscv64_passed_high_level_top_one + 1
        if np.allclose(np.array(x86_64_scores[0]), np.array(riscv64_scores[0]), rtol=1e-05):
          self.output_json[key]["x86_64_riscv64_test_top_one"] = "passed"
          x86_64_riscv64_passed_top_one = x86_64_riscv64_passed_top_one + 1
        else:
          self.output_json[key]["x86_64_riscv64_test_top_one"] = "failed"
          x86_64_riscv64_failed_top_one = x86_64_riscv64_failed_top_one + 1
          x86_64_riscv64_failed_top_one_items.append({key: self.output_json[key]})
      else:
        self.output_json[key]["x86_64_riscv64_failed_high_level_top_one"] = "failed"
        x86_64_riscv64_failed_high_level_top_one = x86_64_riscv64_failed_high_level_top_one + 1
        x86_64_riscv64_failed_high_level_top_one_items.append({key: self.output_json[key]})
        self.output_json[key]["x86_64_riscv64_test_top_one"] = "failed"
        x86_64_riscv64_failed_top_one = x86_64_riscv64_failed_top_one + 1
        x86_64_riscv64_failed_top_one_items.append({key: self.output_json[key]})

      if pytorch_labels[0] == riscv64_labels[0]:
        self.output_json[key]["pytorch_riscv64_passed_high_level_top_one"] = "passed"
        pytorch_riscv64_passed_high_level_top_one = pytorch_riscv64_passed_high_level_top_one + 1
        if np.allclose(np.array(pytorch_scores[0]), np.array(riscv64_scores[0]), rtol=1e-05):
          self.output_json[key]["pytorch_riscv64_test_top_one"] = "passed"
          pytorch_riscv64_passed_top_one = pytorch_riscv64_passed_top_one + 1
        else:
          self.output_json[key]["pytorch_riscv64_test_top_one"] = "failed"
          pytorch_riscv64_failed_top_one = pytorch_riscv64_failed_top_one + 1
          pytorch_riscv64_failed_top_one_items.append({key: self.output_json[key]})
      else:
        self.output_json[key]["pytorch_riscv64_failed_high_level_top_one"] = "failed"
        pytorch_riscv64_failed_high_level_top_one = pytorch_riscv64_failed_high_level_top_one + 1
        pytorch_riscv64_failed_high_level_top_one_items.append({key: self.output_json[key]})
        self.output_json[key]["pytorch_riscv64_test_top_one"] = "failed"
        pytorch_riscv64_failed_top_one = pytorch_riscv64_failed_top_one + 1
        pytorch_riscv64_failed_top_one_items.append({key: self.output_json[key]})

      if set(x86_64_labels) == set(riscv64_labels):
        self.output_json[key]["x86_64_riscv64_passed_high_level_top_five"] = "passed"
        x86_64_riscv64_passed_high_level_top_five = x86_64_riscv64_passed_high_level_top_five + 1
        if np.allclose(x86_64_scores, riscv64_scores, rtol=1e-05):
          self.output_json[key]["x86_64_riscv64_test_top_five"] = "passed"
          x86_64_riscv64_passed_top_five = x86_64_riscv64_passed_top_five + 1
        else:
          self.output_json[key]["x86_64_riscv64_test_top_five"] = "failed"
          x86_64_riscv64_failed_top_five = x86_64_riscv64_failed_top_five + 1
          x86_64_riscv64_failed_top_five_items.append({key: self.output_json[key]})
      else:
        self.output_json[key]["x86_64_riscv64_failed_high_level_top_five"] = "failed"
        x86_64_riscv64_failed_high_level_top_five = x86_64_riscv64_failed_high_level_top_five + 1
        x86_64_riscv64_failed_high_level_top_five_items.append({key: self.output_json[key]})
        self.output_json[key]["x86_64_riscv64_test_top_five"] = "failed"
        x86_64_riscv64_failed_top_five = x86_64_riscv64_failed_top_five + 1
        x86_64_riscv64_failed_top_five_items.append({key: self.output_json[key]})

      if set(pytorch_labels) == set(riscv64_labels):
        self.output_json[key]["pytorch_riscv64_passed_high_level_top_five"] = "passed"
        pytorch_riscv64_passed_high_level_top_five = pytorch_riscv64_passed_high_level_top_five + 1
        if np.allclose(pytorch_scores, riscv64_scores, rtol=1e-05):
          self.output_json[key]["pytorch_riscv64_test_top_five"] = "passed"
          pytorch_riscv64_passed_top_five = pytorch_riscv64_passed_top_five + 1
        else:
          self.output_json[key]["pytorch_riscv64_test_top_five"] = "failed"
          pytorch_riscv64_failed_top_five = pytorch_riscv64_failed_top_five + 1
          pytorch_riscv64_failed_top_five_items.append({key: self.output_json[key]})
      else:
        self.output_json[key]["pytorch_riscv64_failed_high_level_top_five"] = "failed"
        pytorch_riscv64_failed_high_level_top_five = pytorch_riscv64_failed_high_level_top_five + 1
        pytorch_riscv64_failed_high_level_top_five_items.append({key: self.output_json[key]})
        self.output_json[key]["pytorch_riscv64_test_top_five"] = "failed"
        pytorch_riscv64_failed_top_five = pytorch_riscv64_failed_top_five + 1
        pytorch_riscv64_failed_top_five_items.append({key: self.output_json[key]})

    x86_64_riscv64_passed_high_level_top_one_accuracy = x86_64_riscv64_passed_high_level_top_one*100/num_steps
    pytorch_riscv64_passed_high_level_top_one_accuracy = pytorch_riscv64_passed_high_level_top_one*100/num_steps
    x86_64_riscv64_passed_high_level_top_five_accuracy = x86_64_riscv64_passed_high_level_top_five*100/num_steps
    pytorch_riscv64_passed_high_level_top_five_accuracy = pytorch_riscv64_passed_high_level_top_five*100/num_steps
    x86_64_riscv64_passed_top_one_accuracy = x86_64_riscv64_passed_top_one*100/num_steps
    pytorch_riscv64_passed_top_one_accuracy = pytorch_riscv64_passed_top_one*100/num_steps
    x86_64_riscv64_passed_top_five_accuracy = x86_64_riscv64_passed_top_five*100/num_steps
    pytorch_riscv64_passed_top_five_accuracy = pytorch_riscv64_passed_top_five*100/num_steps

    self.output_json["x86_64_riscv64_passed_high_level_top_one_accuracy"] = x86_64_riscv64_passed_high_level_top_one_accuracy
    self.output_json["x86_64_riscv64_passed_top_one_accuracy"] = x86_64_riscv64_passed_top_one_accuracy
    self.output_json["pytorch_riscv64_passed_high_level_top_one_accuracy"] = pytorch_riscv64_passed_high_level_top_one_accuracy
    self.output_json["pytorch_riscv64_passed_top_one_accuracy"] = pytorch_riscv64_passed_top_one_accuracy
    self.output_json["x86_64_riscv64_passed_high_level_top_five_accuracy"] = x86_64_riscv64_passed_high_level_top_five_accuracy
    self.output_json["x86_64_riscv64_passed_top_five_accuracy"] = x86_64_riscv64_passed_top_five_accuracy
    self.output_json["pytorch_riscv64_passed_high_level_top_five_accuracy"] = pytorch_riscv64_passed_high_level_top_five_accuracy
    self.output_json["pytorch_riscv64_passed_top_five_accuracy"] = pytorch_riscv64_passed_top_five_accuracy

    print(" \t# x86_64 vs riscv64 high level top one test summary ... \n")
    print(f"\t  - passed cases: {x86_64_riscv64_passed_high_level_top_one}")
    print(f"\t  - failed cases: {x86_64_riscv64_failed_high_level_top_one}")
    print(f"\t  - accuracy    : {x86_64_riscv64_passed_high_level_top_one_accuracy} %\n")
    if verbose_report:
      if x86_64_riscv64_failed_high_level_top_one > 0: print("\t  - summary report for failed test cases ...\n")
      for item in x86_64_riscv64_failed_high_level_top_one_items:
        key = list(item.keys())[0]
        print(f"\titem: {key}")
        print(f"\tx86_64 score  : {item[key]['x86_64']['scores'][0]}")
        print(f"\triscv64 score : {item[key]['riscv64']['scores'][0]}")
        print()

    print(" \t# x86_64 vs riscv64 top one test summary ... \n")
    print(f"\t  - passed cases: {x86_64_riscv64_passed_top_one}")
    print(f"\t  - failed cases: {x86_64_riscv64_failed_top_one}")
    print(f"\t  - accuracy    : {x86_64_riscv64_passed_top_one_accuracy} %\n")
    if verbose_report:
      if x86_64_riscv64_failed_top_one > 0: print("\t  - summary report for failed test cases ...\n")
      for item in x86_64_riscv64_failed_top_one_items:
        key = list(item.keys())[0]
        print(f"\titem: {key}")
        print(f"\tx86_64 score  : {item[key]['x86_64']['scores'][0]}")
        print(f"\triscv64 score : {item[key]['riscv64']['scores'][0]}")
        print()

    print(" \t# pytorch vs riscv64 high level top one test summary ... \n")
    print(f"\t  - passed cases: {pytorch_riscv64_passed_high_level_top_one}")
    print(f"\t  - failed cases: {pytorch_riscv64_failed_high_level_top_one}")
    print(f"\t  - accuracy    : {pytorch_riscv64_passed_high_level_top_one_accuracy} %\n")
    if verbose_report:
      if pytorch_riscv64_failed_high_level_top_one > 0: print("\t  - summary report for failed test cases ...\n")
      for item in pytorch_riscv64_failed_high_level_top_one_items:
        key = list(item.keys())[0]
        print(f"\titem: {key}")
        print(f"\tpytorch score  : {item[key]['pytorch']['scores'][0]}")
        print(f"\triscv64 score : {item[key]['riscv64']['scores'][0]}")
        print()

    print(" \t# pytorch vs riscv64 top one test summary ... \n")
    print(f"\t  - passed cases: {pytorch_riscv64_passed_top_one}")
    print(f"\t  - failed cases: {pytorch_riscv64_failed_top_one}")
    print(f"\t  - accuracy    : {pytorch_riscv64_passed_top_one_accuracy} %\n")
    if verbose_report:
      if pytorch_riscv64_failed_top_one > 0: print("\t  - summary report for failed test cases ...\n")
      for item in pytorch_riscv64_failed_top_one_items:
        key = list(item.keys())[0]
        print(f"\titem: {key}")
        print(f"\tpytorch score  : {item[key]['pytorch']['scores'][0]}")
        print(f"\triscv64 score : {item[key]['riscv64']['scores'][0]}")
        print()

    print(" \t# x86_64 vs riscv64 high level top five test summary ... \n")
    print(f"\t  - passed cases: {x86_64_riscv64_passed_high_level_top_five}")
    print(f"\t  - failed cases: {x86_64_riscv64_failed_high_level_top_five}")
    print(f"\t  - accuracy    : {x86_64_riscv64_passed_high_level_top_five_accuracy} %\n")
    if verbose_report:
      if x86_64_riscv64_failed_high_level_top_five > 0: print("\t  - summary report for failed test cases ...\n")
      for item in x86_64_riscv64_failed_high_level_top_five_items:
        key = list(item.keys())[0]
        print(f"\titem: {key}")
        print(f"\tx86_64 score  : {item[key]['x86_64']['scores'][0]}")
        print(f"\triscv64 score : {item[key]['riscv64']['scores'][0]}")
        print()

    print(" \t# x86_64 vs riscv64 top five test summary ... \n")
    print(f"\t  - passed cases: {x86_64_riscv64_passed_top_five}")
    print(f"\t  - failed cases: {x86_64_riscv64_failed_top_five}")
    print(f"\t  - accuracy    : {x86_64_riscv64_passed_top_five_accuracy} %\n")
    if verbose_report:
      if x86_64_riscv64_failed_top_five > 0: print("\t  - summary report for failed test cases ...\n")
      for item in x86_64_riscv64_failed_top_five_items:
        key = list(item.keys())[0]
        print(f"\titem: {key}")
        print(f"\tx86_64 scores  : {item[key]['x86_64']['scores']}")
        print(f"\triscv64 scores : {item[key]['riscv64']['scores']}")
        print()

    print(" \t# pytorch vs riscv64 high level top five test summary ... \n")
    print(f"\t  - passed cases: {pytorch_riscv64_passed_high_level_top_five}")
    print(f"\t  - failed cases: {pytorch_riscv64_failed_high_level_top_five}")
    print(f"\t  - accuracy    : {pytorch_riscv64_passed_high_level_top_five_accuracy} %\n")
    if verbose_report:
      if pytorch_riscv64_failed_high_level_top_five > 0: print("\t  - summary report for failed test cases ...\n")
      for item in pytorch_riscv64_failed_high_level_top_five_items:
        key = list(item.keys())[0]
        print(f"\titem: {key}")
        print(f"\tpytorch score  : {item[key]['pytorch']['scores'][0]}")
        print(f"\triscv64 score : {item[key]['riscv64']['scores'][0]}")
        print()

    print(" \t# pytorch vs riscv64 top five test summary ... \n")
    print(f"\t  - passed cases: {pytorch_riscv64_passed_top_five}")
    print(f"\t  - failed cases: {pytorch_riscv64_failed_top_five}")
    print(f"\t  - accuracy    : {pytorch_riscv64_passed_top_five_accuracy} %\n")
    if verbose_report:
      if pytorch_riscv64_failed_top_five > 0: print("\t  - summary report for failed test cases ...\n")
      for item in pytorch_riscv64_failed_top_five_items:
        key = list(item.keys())[0]
        print(f"\titem: {key}")
        print(f"\tpytorch scores  : {item[key]['pytorch']['scores']}")
        print(f"\triscv64 scores : {item[key]['riscv64']['scores']}")
        print()

    print(f" verification log can be found at {self.json_path} ...\n")
    print(" verification completed ... \n")

  def end(self, arch: str, num_steps: int, accuracy_tvm: float, accuracy_pt: float) -> None:
    if arch == "x86_64":
      self.output_json["inference_speed(FPS)"] = {
        "tvm_x86_64": int(num_steps * 10e9 / self.elapsed_time_ns_tvm),
        "pytorch_x86_64": int(num_steps * 10e9 / self.elapsed_time_ns_pt)
      }
      self.output_json["model_accuracy"] = {
        "tvm_x86_64": round(accuracy_tvm * 100, 4),
        "pytorch_x86_64": round(accuracy_pt * 100, 4)
      }
    else:
      self.output_json["inference_speed(FPS)"]["tvm_riscv64"] = round(num_steps * 10e9 / self.elapsed_time_ns_tvm, 4)
      self.output_json["model_accuracy"]["tvm_riscv64"] = round(accuracy_tvm * 100, 4)
    json_file = open(self.json_path, 'w')
    json.dump(self.output_json, json_file, indent=4)
