# Copyright © 2023 ACCELR

import json
import numpy as np
from typing import Tuple

class evaluator:
  json_path: str = "./imagenet/log.json"

  def __init__(self, arch: str):
    self.arch = arch; self.output_json = {}
    try:
      json_file = open(evaluator.json_path, 'r')
      self.output_json = json.load(json_file)
    except FileNotFoundError:
      json_file = open(evaluator.json_path, 'w')
      json.dump({}, json_file)

  def log(self, key: str, output: Tuple[float, str]) -> None:
    if key not in self.output_json:
      self.output_json[key] = { self.arch : { "scores" : [str(i)for i in output[0]], "labels" : [str(i)for i in output[1]] } }
    else:
      self.output_json[key][self.arch] = { "scores" : [str(i)for i in output[0]], "labels" : [str(i)for i in output[1]] }

  def process(self, num_steps: int) -> None:
    print(" starting verification ... \n")
    print(f" {num_steps} verification test cases detected ... \n")

    passed = 0; failed = 0; failed_items = []; step_counter = 0
    for key in self.output_json:
      step_counter += 1
      if step_counter > num_steps: break
      if self.output_json[key].get("x86_64") is None:
        print(" run the test in x86_64 first ... \n")
        return

      x86_64_scores = np.array(list(map(float, self.output_json[key]["x86_64"]["scores"])))
      riscv64_scores = np.array(list(map(float, self.output_json[key]["riscv64"]["scores"])))
      if np.testing.assert_allclose(x86_64_scores, riscv64_scores, rtol=1e-05) is None:
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
      print(f"\tx86_64 scores  : {item[key]['x86_64']['scores']}")
      print(f"\triscv64 scores : {item[key]['riscv64']['scores']}")
      print()
    print(" verification completed ... \n")

  def end(self) -> None:
    json_file = open(evaluator.json_path, 'w')
    json.dump(self.output_json, json_file, indent=4)
