# Copyright © 2023 ACCELR

import tvm
import numpy as np

def teop_session(arch:str, N: int=10) -> None:
  lib_path = "./bin/teop_arch.tar".replace("arch", arch)
  print(f"\n~ TVM custom op inference session on {arch} ~\n\n" )
  print(f" dynamic loading compiled library from {lib_path}! \n")

  mod = tvm.runtime.load_module(lib_path)
  faddone = mod.get_function("fadd")
  x = tvm.nd.array(np.arange(N, dtype=np.float32))
  y = tvm.nd.array(np.zeros(N, dtype=np.float32))

  print(" starting verification ... \n")
  faddone(x, y); np_x = x.numpy(); np_y = y.numpy()
  assert np.all([xi + 1 == yi for xi, yi in zip(np_x, np_y)])
  print(" verification completed ... \n")
  print(f"\n~ End of TVM custom op inference session on {arch} ~\n\n")
