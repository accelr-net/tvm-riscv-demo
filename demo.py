# Copyright © 2023 ACCELR

import os
import platform

from customop.compile import teop
from customop.run import teop_session

from imagenet.compile import resnet18
from imagenet.run import resnet18_session

from imagenet.dataloader import preprocess_imagenet_data

import tvm



devices = {
  "riscv64" : tvm.target.Target("llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=generic-rv64 -mabi=lp64d -mattr=+64bit,+m,+f,+d"),
  "x86_64"  : tvm.target.Target("llvm")
}

platform_arch = platform.machine().lower()
num_setps = os.environ.get("STEPS", 10)


if platform_arch == "x86_64": pass
  # ********************** compile for x86 **********************
  # teop_module = teop(devices["x86_64"])
  # teop_module.export_library("./bin/teop_x86_64.tar")

  # resnet18_module = resnet18(devices["x86_64"])
  # resnet18_module.export_library("./bin/resnet18_x86_64.tar")

  # ******************* compile for riscv64 *********************
  # teop_module = teop(devices["riscv64"])
  # teop_module.export_library("./bin/teop_rv64.tar")
  
  # resnet18_module = resnet18(devices["riscv64"])
  # resnet18_module.export_library("./bin/resnet18_rv64.tar")      
  
  # ************** dave input data tensors to disk ****************
  # preprocess_imagenet_data(num_setps)

# *********************** inference session ***********************
teop_session(platform_arch)
resnet18_session(platform_arch, num_setps)
