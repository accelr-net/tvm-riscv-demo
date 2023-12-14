# Copyright © 2023 ACCELR

import tvm
import platform
import numpy as np

platform_arch = platform.machine().lower()

def pretty_print(input: str) -> None:
  pad_length = max(0, 100 - len(input))
  print(f"\n{'-' * (pad_length // 2) + input + '-' * (pad_length // 2 + pad_length % 2)}\n\n")

def teop_session(M: int=1024, N: int=1024, K: int=1024) -> None:
  lib_path = "./bin/teop_arch.tar".replace("arch", platform_arch)
  pretty_print(f" TVM custom op inference session on {platform_arch} ")
  print(f" dynamic loading compiled library from {lib_path}! \n")

  mod = tvm.runtime.load_module(lib_path)
  mmult = mod.get_function("mmult")
  x = tvm.nd.array(np.random.rand(M, K).astype(np.float32))
  y = tvm.nd.array(np.random.rand(K, N).astype(np.float32))
  z = tvm.nd.array(np.zeros((M, N), dtype=np.float32))

  print(" starting verification ... \n")
  mmult(x, y, z)
  np_x = x.numpy(); np_y = y.numpy()
  np_z = np_x @ np_y
  np.testing.assert_allclose(np_z, z.numpy(), rtol=1e-5)
  print(" verification completed ... \n")
  pretty_print(f" End of TVM custom op inference session on {platform_arch} ")
