# Copyright © 2023 ACCELR

import unittest
import numpy as np

from customop.compile import teop
from customop.run import teop_session

from imagenet.compile import resnet18
from imagenet.run import resnet18_session

from imagenet.dataloader import preprocess_imagenet_data

import tvm


num_steps = 10 # check 10 test cases for smoketest verification!

class test(unittest.TestCase):
    
  def test_customop(self):
    teop_module = teop(tvm.target.Target("llvm"))
    teop_module.export_library("teop_x86_64.tar")    

    teop_session("x86_64", lib_path="teop_x86_64.tar")

  # def test_dataloader(self):
  #   preprocess_imagenet_data(num_steps)

  def test_imagenet(self):
    resnet18_module = resnet18(tvm.target.Target("llvm"))
    resnet18_module.export_library("resnet18_x86_64.tar")

    resnet18_session("x86_64", num_steps, lib_path="resnet18_arch.tar")

if __name__ == '__main__':
  unittest.main()
