# Copyright © 2023 ACCELR

import json
import numpy as np
from typing import Tuple

class evaluator:
  def __init__(self, arch: str, json_path: str):
    self.arch = arch; self.output_json = {}
    self.json_path = json_path
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

  def log(self, key: str, output: Tuple[float, str], pt: bool=False) -> None:
    if key not in self.output_json:
      self.output_json[key] = {
        "pytorch" if pt else self.arch: {
          "scores" : [str(i)for i in output[0]],
          "labels" : [str(i)for i in output[1]]
        }
      }
    else:
      self.output_json[key]["pytorch" if pt else self.arch] = {
        "scores" : [str(i)for i in output[0]],
        "labels" : [str(i)for i in output[1]]
      }

  def process(self, num_steps: int) -> None:
    print(" starting verification ... \n")
    print(f" {num_steps} verification test case(s) detected ... \n")

    x86_64_riscv64_passed_top_five = 0; x86_64_riscv64_failed_top_five = 0
    pytorch_riscv64_passed_top_five = 0; pytorch_riscv64_failed_top_five = 0
    x86_64_riscv64_failed_top_five_items = []; pytorch_riscv64_failed_top_five_items = [];

    x86_64_riscv64_passed_top_one = 0; x86_64_riscv64_failed_top_one = 0
    pytorch_riscv64_passed_top_one = 0; pytorch_riscv64_failed_top_one = 0
    x86_64_riscv64_failed_top_one_items = []; pytorch_riscv64_failed_top_one_items = [];

    step_counter = 0
    for key in self.output_json:
      step_counter += 1
      if step_counter > num_steps: break
      if self.output_json[key].get("x86_64") is None:
        print(" run the test in x86_64 first ... \n")
        return

      self.output_json[key]["test no"] = step_counter

      x86_64_scores = np.array(list(map(float, self.output_json[key]["x86_64"]["scores"])))
      riscv64_scores = np.array(list(map(float, self.output_json[key]["riscv64"]["scores"])))
      pytorch_scores = np.array(list(map(float, self.output_json[key]["pytorch"]["scores"])))

      if np.allclose(x86_64_scores, riscv64_scores, rtol=1e-05):
        self.output_json[key]["x86_64_riscv64_test_top_five"] = "passed"
        x86_64_riscv64_passed_top_five = x86_64_riscv64_passed_top_five + 1
      else:
        self.output_json[key]["x86_64_riscv64_test_top_five"] = "failed"
        x86_64_riscv64_failed_top_five = x86_64_riscv64_failed_top_five + 1
        x86_64_riscv64_failed_top_five_items.append({key: self.output_json[key]})

      if np.allclose(pytorch_scores, riscv64_scores, rtol=1e-05):
        self.output_json[key]["pytorch_riscv64_test_top_five"] = "passed"
        pytorch_riscv64_passed_top_five = pytorch_riscv64_passed_top_five + 1
      else:
        self.output_json[key]["pytorch_riscv64_test_top_five"] = "failed"
        pytorch_riscv64_failed_top_five = pytorch_riscv64_failed_top_five + 1
        pytorch_riscv64_failed_top_five_items.append({key: self.output_json[key]})

      if np.allclose(np.array(x86_64_scores[0]), np.array(riscv64_scores[0]), rtol=1e-05):
        self.output_json[key]["x86_64_riscv64_test_top_one"] = "passed"
        x86_64_riscv64_passed_top_one = x86_64_riscv64_passed_top_one + 1
      else:
        self.output_json[key]["x86_64_riscv64_test_top_one"] = "failed"
        x86_64_riscv64_failed_top_one = x86_64_riscv64_failed_top_one + 1
        x86_64_riscv64_failed_top_one_items.append({key: self.output_json[key]})

      if np.allclose(np.array(pytorch_scores[0]), np.array(riscv64_scores[0]), rtol=1e-05):
        self.output_json[key]["pytorch_riscv64_test_top_one"] = "passed"
        pytorch_riscv64_passed_top_one = pytorch_riscv64_passed_top_one + 1
      else:
        self.output_json[key]["pytorch_riscv64_test_top_one"] = "failed"
        pytorch_riscv64_failed_top_one = pytorch_riscv64_failed_top_one + 1
        pytorch_riscv64_failed_top_one_items.append({key: self.output_json[key]})

    x86_64_riscv64_passed_top_five_accuracy = x86_64_riscv64_passed_top_five*100/num_steps
    pytorch_riscv64_passed_top_five_accuracy = pytorch_riscv64_passed_top_five*100/num_steps
    x86_64_riscv64_passed_top_one_accuracy = x86_64_riscv64_passed_top_one*100/num_steps
    pytorch_riscv64_passed_top_one_accuracy = pytorch_riscv64_passed_top_one*100/num_steps

    self.output_json["x86_64_riscv64_passed_top_five_accuracy"] = x86_64_riscv64_passed_top_five_accuracy
    self.output_json["pytorch_riscv64_passed_top_five_accuracy"] = pytorch_riscv64_passed_top_five_accuracy
    self.output_json["x86_64_riscv64_passed_top_one_accuracy"] = x86_64_riscv64_passed_top_one_accuracy
    self.output_json["pytorch_riscv64_passed_top_one_accuracy"] = pytorch_riscv64_passed_top_one_accuracy

    print(" \t# x86_64 vs riscv64 top five test summary ... \n")
    print(f"\t  - passed cases: {x86_64_riscv64_passed_top_five}")
    print(f"\t  - failed cases: {x86_64_riscv64_failed_top_five}")
    print(f"\t  - accuracy    : {x86_64_riscv64_passed_top_five_accuracy} %\n")
    if x86_64_riscv64_failed_top_five > 0: print("\t  - summary report for failed test cases ...\n")
    for item in x86_64_riscv64_failed_top_five_items:
      key = list(item.keys())[0]
      print(f"\titem: {key}")
      print(f"\tx86_64 scores  : {item[key]['x86_64']['scores']}")
      print(f"\triscv64 scores : {item[key]['riscv64']['scores']}")
      print()

    print(" \t# pytorch vs riscv64 top five test summary ... \n")
    print(f"\t  - passed cases: {pytorch_riscv64_passed_top_five}")
    print(f"\t  - failed cases: {pytorch_riscv64_failed_top_five}")
    print(f"\t  - accuracy    : {pytorch_riscv64_passed_top_five_accuracy} %\n")
    if pytorch_riscv64_failed_top_five > 0: print("\t  - summary report for failed test cases ...\n")
    for item in pytorch_riscv64_failed_top_five_items:
      key = list(item.keys())[0]
      print(f"\titem: {key}")
      print(f"\tx86_64 scores  : {item[key]['x86_64']['scores']}")
      print(f"\triscv64 scores : {item[key]['riscv64']['scores']}")
      print()

    print(" \t# x86_64 vs riscv64 top one test summary ... \n")
    print(f"\t  - passed cases: {x86_64_riscv64_passed_top_one}")
    print(f"\t  - failed cases: {x86_64_riscv64_failed_top_one}")
    print(f"\t  - accuracy    : {x86_64_riscv64_passed_top_one_accuracy} %\n")
    if x86_64_riscv64_failed_top_one > 0: print("\t  - summary report for failed test cases ...\n")
    for item in x86_64_riscv64_failed_top_one_items:
      key = list(item.keys())[0]
      print(f"\titem: {key}")
      print(f"\tx86_64 score  : {item[key]['x86_64']['scores'][0]}")
      print(f"\triscv64 score : {item[key]['riscv64']['scores'][0]}")
      print()

    print(" \t# pytorch vs riscv64 top one test summary ... \n")
    print(f"\t  - passed cases: {pytorch_riscv64_passed_top_one}")
    print(f"\t  - failed cases: {pytorch_riscv64_failed_top_one}")
    print(f"\t  - accuracy    : {pytorch_riscv64_passed_top_one_accuracy} %\n")
    if pytorch_riscv64_failed_top_one > 0: print("\t  - summary report for failed test cases ...\n")
    for item in pytorch_riscv64_failed_top_one_items:
      key = list(item.keys())[0]
      print(f"\titem: {key}")
      print(f"\tx86_64 score  : {item[key]['x86_64']['scores'][0]}")
      print(f"\triscv64 score : {item[key]['riscv64']['scores'][0]}")
      print()

    print(f" verification log can be found at {self.json_path} ...\n")
    print(" verification completed ... \n")

  def end(self) -> None:
    json_file = open(self.json_path, 'w')
    json.dump(self.output_json, json_file, indent=4)
