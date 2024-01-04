import numpy as np

def softmax(input: np.ndarray) -> np.ndarray:
  shape = input.shape
  input = input.flatten()
  softmax_output = np.exp(input)/sum(np.exp(input))
  return softmax_output.reshape(shape)

def pretty_print(input: str) -> None:
  pad_length = max(0, 100 - len(input))
  print(f"\n{'-' * (pad_length // 2) + input + '-' * (pad_length // 2 + pad_length % 2)}\n\n")
