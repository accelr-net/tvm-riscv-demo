# Copyright © 2023 ACCELR

import tvm
import numpy as np
  

def teop_session(arch:str, N: int=10, lib_path: str="./bin/teop_arch.tar") -> None:
  lib_path = lib_path.replace("arch", arch)

  print("\n")
  print( "-----------------------------------------------" )
  print(f" ~ TVM custom op inference session on {arch} ~ " )
  print( "-----------------------------------------------" )
  print("\n")

  print(f" dynamic loading compiled library from {lib_path} ")
  
  mod = tvm.runtime.load_module(lib_path)
  faddone = mod.get_function("fadd")

  x = tvm.nd.array(np.arange(N, dtype=np.float32))
  y = tvm.nd.array(np.zeros(N, dtype=np.float32))
  
  faddone(x, y); np_x = x.numpy(); np_y = y.numpy()
  assert np.all([xi + 1 == yi for xi, yi in zip(np_x, np_y)])

  print(" verification completed... \n")
