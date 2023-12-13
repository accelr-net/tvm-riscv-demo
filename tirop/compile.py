# Copyright © 2023 ACCELR

import tvm
from tvm import te

def teop(target: tvm.target.Target, M: int=1024, N: int=1024, K: int=1024) -> tvm.runtime.Module:
  k = te.reduce_axis((0, K), "k")
  A = te.placeholder((M, K), name="A")
  B = te.placeholder((K, N), name="B")
  C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
  s = te.create_schedule(C.op)
  lib = tvm.build(s, [A, B, C], target=target, name="mmult")
  return lib
