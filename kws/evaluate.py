# Copyright © 2023 ACCELR

import json
import pickle
import numpy as np
from typing import Tuple

class evaluator:
  json_path: str = "./kws/log.json"

  def __init__(self, arch: str):
    self.arch = arch; self.output_json = {}
    try:
      json_file = open(evaluator.json_path, 'r')
      self.output_json = json.load(json_file)
    except FileNotFoundError:
      json_file = open(evaluator.json_path, 'w')
      json.dump({}, json_file)

    labels = pickle.load(open("./models/lable.pickle", 'rb'))
    self.class_to_idx = {c: i for i, c in enumerate(labels)}
    self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

  def log(self, key: str, output: np.ndarray) -> None:
    top_output = np.argmax(output)
    label = self.idx_to_class[top_output]
    if key not in self.output_json:
      self.output_json[key] = { self.arch : { "score" : int(top_output), "label" :  label} }
    else:
      self.output_json[key][self.arch] = { "score" : int(top_output), "label" :  label}

  def process(self, num_steps: int) -> None:
    print(" starting verification ... \n")
    print(f" {num_steps} verification test case(s) detected ... \n")

    passed = 0; failed = 0; failed_items = []; step_counter = 0
    for key in self.output_json:
      step_counter += 1
      if step_counter > num_steps: break
      if self.output_json[key].get("x86_64") is None:
        print(" run the test in x86_64 first ... \n")
        return

      x86_64_score = self.output_json[key]["x86_64"]["score"]
      riscv64_score = self.output_json[key]["riscv64"]["score"]
      x86_64_label = self.output_json[key]["x86_64"]["label"]
      riscv64_label = self.output_json[key]["riscv64"]["label"]
      if  x86_64_score == riscv64_score and x86_64_label == riscv64_label:
        self.output_json[key]["status"] = "passed"
        passed = passed + 1
      else:
        self.output_json[key]["status"] = "failed"
        failed = failed + 1
        failed_items.append({key: self.output_json[key]})

    print(f"  - passed cases: {passed}")
    print(f"  - failed cases: {failed}")
    print(f"\n verification log can be found at {evaluator.json_path}\n ")
    if failed > 0: print("  - summary report for failed test cases ...\n")
    for item in failed_items:
      key = list(item.keys())[0]
      print(f"\titem: {key}")
      print(f"\tx86_64 score  : {item[key]['x86_64']['score']}")
      print(f"\tx86_64 label  : {item[key]['x86_64']['label']}")
      print(f"\triscv64 score : {item[key]['riscv64']['score']}")
      print(f"\triscv64 label : {item[key]['riscv64']['label']}")
      print()
    print(" verification completed ... \n")

  def end(self) -> None:
    json_file = open(evaluator.json_path, 'w')
    json.dump(self.output_json, json_file, indent=4)
