# Copyright © 2023 ACCELR

from typing import Tuple

class evaluator:
  def __init__(self, arch: str):
    self.arch = arch

  def log(self, output: Tuple[float, str]) -> None:
    print(output)

  
