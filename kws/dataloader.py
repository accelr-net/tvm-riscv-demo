# Copyright © 2023 ACCELR

import os
import torch
import torchaudio
import numpy as np
from torchaudio.datasets import SPEECHCOMMANDS

class SubsetSC(SPEECHCOMMANDS):
  def __init__(self, subset: str=None):
    super().__init__("./data/speechcommands", download=True if not os.path.exists("./data/speechcommands/speech_commands_v0.02.tar.gz") else False)

    def load_list(filename):
      filepath = os.path.join(self._path, filename)
      with open(filepath) as fileobj:
        return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

    if subset == "validation":
      self._walker = load_list("validation_list.txt")
    elif subset == "testing":
      self._walker = load_list("testing_list.txt")
    elif subset == "training":
      excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
      excludes = set(excludes)
      self._walker = [w for w in self._walker if w not in excludes]

def _transform_audio(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
  transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, win_length=None, hop_length=512, n_mels=128, power=2.0)
  transformed = transform(audio)
  return transformed

def _reshape(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
  if waveform.shape[-1] < sample_rate :
    waveform = torch.cat([waveform, torch.zeros((1, sample_rate - waveform.shape[-1]))], dim=-1)
  elif waveform.shape[-1] > sample_rate:
    waveform = waveform[:,:sample_rate]
  waveform = _transform_audio(waveform, sample_rate)
  return waveform

def _power_to_db(s: float, amin: float=1e-16, top_db: float=80.0) -> np.ndarray:
  # https://gist.github.com/dschwertfeger/f9746bc62871c736e47d5ec3ff4230f7
  _log10 = lambda x: np.log(x) / np.log(10).astype(np.float32)
  log_spec = 10.0 * _log10(np.maximum(amin, s))
  log_spec -= 10.0 * _log10(np.maximum(amin, np.max(s)))
  log_spec = np.maximum(log_spec, np.max(log_spec) - top_db)
  return log_spec

def _mel_conversion(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
  waveform = _reshape(waveform, sample_rate)
  spectogram = torch.tensor(_power_to_db(waveform.squeeze().numpy())).unsqueeze(0)
  spectogram = spectogram.unsqueeze(0)
  return spectogram

def _get_shape() -> torch.Size:
  set = SubsetSC("testing")
  waveform, sample_rate, *_ = set[0]
  spectogram = _mel_conversion(waveform,sample_rate)
  return spectogram.shape

def preprocess_speechcommands_data(waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
  spectogram_t = _mel_conversion(waveform,sample_rate).numpy()
  return spectogram_t
