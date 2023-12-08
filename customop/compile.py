# Copyright © 2023 ACCELR

import tvm
from tvm import te


def teop(target: tvm.target.Target) -> tvm.runtime.Module:
  n = te.var("n")

  A = te.placeholder((n,), name="A")
  B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
  s = te.create_schedule(B.op)

  lib = tvm.build(s, [A, B], target=target, name="fadd")

  return lib
