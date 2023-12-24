# Copyright © 2023 ACCELR

import os
import torch
import torchaudio
import librosa
from torchvision.transforms import ToTensor
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

def _mel_conversion(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
  waveform = _reshape(waveform, sample_rate)
  spectogram = ToTensor()(librosa.power_to_db(waveform.squeeze().numpy(), ref=np.max))
  spectogram = spectogram.unsqueeze(0)
  return spectogram

def _get_shape() -> torch.Size:
  set = SubsetSC("testing")
  waveform, sample_rate, *_ = set[0]
  spectogram = _mel_conversion(waveform,sample_rate)
  return spectogram.shape

def preprocess_speechcommands_data(num_steps: int, data_dir: str) -> None:
  dataset = SubsetSC("testing")
  counter = 0
  for data in dataset:
    counter = counter + 1
    if counter > num_steps and num_steps >= 0: break
    waveform_t, sample_rate_t, label, speaker_id_t, utterance_number_t  = data
    spectogram_t = _mel_conversion(waveform_t,sample_rate_t).numpy()
    np.save(data_dir + label + speaker_id_t + str(utterance_number_t), spectogram_t)
