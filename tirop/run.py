# Copyright © 2023 ACCELR

import tvm
import numpy as np

def pretty_print(input: str) -> None:
  pad_length = max(0, 100 - len(input))
  print(f"\n{'-' * (pad_length // 2) + input + '-' * (pad_length // 2 + pad_length % 2)}\n\n")

def teop_session(arch:str, N: int=10) -> None:
  lib_path = "./bin/teop_arch.tar".replace("arch", arch)
  pretty_print(f" TVM custom op inference session on {arch} ")
  print(f" dynamic loading compiled library from {lib_path}! \n")

  mod = tvm.runtime.load_module(lib_path)
  faddone = mod.get_function("fadd")
  x = tvm.nd.array(np.arange(N, dtype=np.float32))
  y = tvm.nd.array(np.zeros(N, dtype=np.float32))

  print(" starting verification ... \n")
  faddone(x, y); np_x = x.numpy(); np_y = y.numpy()
  assert np.all([xi + 1 == yi for xi, yi in zip(np_x, np_y)])
  print(" verification completed ... \n")
  pretty_print(f" End of TVM custom op inference session on {arch} ")
