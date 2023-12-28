# Copyright © 2023 ACCELR

import os
import sys
import platform
import argparse

if __name__ == "__main__":
  platform_arch = platform.machine().lower()
  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--tirop", help="activate tensor ir operation test", default=False)
  parser.add_argument("-i", "--imagenet", help="activate imagenet test", default=False)
  parser.add_argument("-k", "--kws", help="activate kws test", default=False)
  parser.add_argument("-a", "--all", help="activate all tests", default=False)
  parser.add_argument("-s", "--numsteps", help = "number of examples from the testset which the test should be done", default=10)
  parser.add_argument("-v", "--verbose", help = "verbose report for failed test cases in console", default=False)
  args = parser.parse_args()

  print("\n")
  print("--------------------------")
  print("~ ACCELR RISC V TVM Demo ~")
  print("--------------------------")
  print("\n")
  print(f" {platform_arch} system detected ... \n\n")

  if len(sys.argv) == 1:
    parser.print_help()
    print("\n")

  if platform_arch == "x86_64":
    from tirop.compile import teop
    from imagenet.compile import resnet18
    from kws.compile import kwsrn18
    import tvm

    devices = {
      "riscv64" : tvm.target.Target("llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=generic-rv64 -mabi=lp64d -mattr=+64bit,+m,+f,+d"),
      "x86_64"  : tvm.target.Target("llvm")
    }

    # compile for x86
    if not os.path.exists("./bin/teop_x86_64.tar") or not os.path.exists("./bin/resnet18_x86_64.tar") or not os.path.exists("./bin/kws_x86_64.tar"):
      print(" compiling models for x86_64 ... \n")
      if args.tirop or args.all:
        teop_module = teop(devices["x86_64"])
        teop_module.export_library("./bin/teop_x86_64.tar")
      if args.imagenet or args.all:
        resnet18_module = resnet18(devices["x86_64"])
        resnet18_module.export_library("./bin/resnet18_x86_64.tar")
      if args.kws or args.all:
        kwsrn18_module = kwsrn18(devices["x86_64"])
        kwsrn18_module.export_library("./bin/kws_x86_64.tar")   

    # compile for riscv64
    if not os.path.exists("./bin/teop_riscv64.tar") or not os.path.exists("./bin/resnet18_riscv64.tar") or not os.path.exists("./bin/kws_riscv64.tar"):
      print(" compiling models for riscv64 ... \n")
      if args.tirop or args.all:
        teop_module = teop(devices["riscv64"])
        teop_module.export_library("./bin/teop_riscv64.tar")
      if args.imagenet or args.all:
        resnet18_module = resnet18(devices["riscv64"])
        resnet18_module.export_library("./bin/resnet18_riscv64.tar")
      if args.kws or args.all:
        kwsrn18_module = kwsrn18(devices["riscv64"])
        kwsrn18_module.export_library("./bin/kws_riscv64.tar")

  elif platform_arch != "riscv64":
    raise NotImplementedError

  # inference sessions
  from tirop.run import teop_session
  from imagenet.run import resnet18_session
  from kws.run import kws_session

  if args.tirop: teop_session()
  if args.imagenet: resnet18_session(int(args.numsteps), bool(args.tirop))
  if args.kws: kws_session(int(args.numsteps), bool(args.tirop))
  if args.all:
    teop_session()
    resnet18_session(int(args.numsteps), bool(args.tirop))
    kws_session(int(args.numsteps), bool(args.tirop))

  print("-------------------")
  print("~ End of the Demo ~")
  print("-------------------")
  print("\n")
