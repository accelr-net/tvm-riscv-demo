# Copyright © 2023 ACCELR

import tvm
from tvm.contrib import graph_executor
import pickle
import platform
import numpy as np
from .evaluate import evaluator
from .dataloader import SubsetSC, preprocess_speechcommands_data
from tqdm import tqdm

platform_arch = platform.machine().lower()

if platform_arch == "x86_64":
  import torch

labels = pickle.load(open("./models/lable.pickle", 'rb'))

class pytorch_session:
  def __init__(self, model_path: str="./models/resnet18-kws-best-acc.pt"):
    self.model = torch.jit.load(model_path)
    self.model.eval()

  def infer(self, input: np.ndarray) -> np.ndarray:
    with torch.no_grad():
      output = self.model(torch.tensor(input))
    return output.numpy()

def pretty_print(input: str) -> None:
  pad_length = max(0, 100 - len(input))
  print(f"\n{'-' * (pad_length // 2) + input + '-' * (pad_length // 2 + pad_length % 2)}\n\n")

def softmax(input: np.ndarray) -> np.ndarray:
  shape = input.shape
  input = input.flatten()
  softmax_output = np.exp(input)/sum(np.exp(input))
  return softmax_output.reshape(shape)

def _postprocess(output: np.ndarray) -> None:
  scores = softmax(output)
  scores = np.squeeze(scores)
  ranks = np.argsort(scores)[::-1]
  top_five_output = [[scores[rank] for rank in ranks[0:5]], [labels[rank] for rank in ranks[0:5]]]
  return top_five_output

def kws_session(num_steps: int) -> None:
  eval = evaluator(platform_arch, "./kws/log.json")
  if platform_arch == "x86_64": pt_session = pytorch_session()
  lib_path = "./bin/kws_arch.tar".replace("arch", platform_arch)
  pretty_print(f" TVM kws inference session on {platform_arch} ")
  print(f" dynamic loading compiled library from {lib_path}! \n")

  loaded_lib = tvm.runtime.load_module(lib_path)
  runtime_module = graph_executor.GraphModule(loaded_lib["default"](tvm.device("llvm", 0)))
  dataset = SubsetSC("testing")

  for data_idx in tqdm(range(num_steps if num_steps >= 0 else len(dataset))):
    waveform, sample_rate, label, speaker_id, utterance_number = dataset[data_idx]
    data_tensor = preprocess_speechcommands_data(waveform, sample_rate)
    runtime_module.set_input("data", data_tensor)
    runtime_module.run()
    output = runtime_module.get_output(0).asnumpy()
    top_five_output_tvm = _postprocess(output)
    eval.log(label + str(speaker_id) + str(utterance_number), top_five_output_tvm)

    if platform_arch == "x86_64":
      pytorch_output = pt_session.infer(data_tensor)
      top_five_output_pytorch = _postprocess(pytorch_output)
      eval.log(label + str(speaker_id) + str(utterance_number), top_five_output_pytorch, pt=True)
  
  if platform_arch == "riscv64": eval.process(num_steps)
  eval.end()
  pretty_print(f" End of TVM kws inference session on {platform_arch} ")
