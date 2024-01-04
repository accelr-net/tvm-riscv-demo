import os
import pickle

import torch
import torchaudio
import numpy as np
from torchaudio.datasets import SPEECHCOMMANDS

from ..utils.helpers import softmax

from typing import List, Tuple


class KWSDataLoader:
  class SpeechCommandsSubset(SPEECHCOMMANDS):
    def __init__(self, subset: str=None):
      dir = "./data/speechcommands"
      super().__init__(dir, download=True if not os.path.exists(dir + "/speech_commands_v0.02.tar.gz") else False)

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

  def __init__(self, num_steps: int) -> None:
    scsubset = KWSDataLoader.SpeechCommandsSubset("validation")
    self.processed_dataset = []
    for i in range(num_steps if num_steps < len(scsubset) else len(scsubset)):
      wfm, sr, lbl, spkr_id, utr_no = scsubset[i]
      self.processed_dataset.append((KWSDataLoader._preprocess(wfm, sr), sr, lbl, spkr_id, utr_no))

  @staticmethod
  def _transform_audio(audio: torch.Tensor, sr: int) -> torch.Tensor:
    trnsfrm = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, win_length=None, hop_length=512, n_mels=128, power=2.0)
    transformed = trnsfrm(audio)
    return transformed

  @staticmethod
  def _reshape(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if waveform.shape[-1] < sample_rate :
      waveform = torch.cat([waveform, torch.zeros((1, sample_rate - waveform.shape[-1]))], dim=-1)
    elif waveform.shape[-1] > sample_rate:
      waveform = waveform[:,:sample_rate]
    waveform = KWSDataLoader._transform_audio(waveform, sample_rate)
    return waveform

  @staticmethod
  def _power_to_db(s: float, amin: float=1e-16, top_db: float=80.0) -> np.ndarray:
    # https://gist.github.com/dschwertfeger/f9746bc62871c736e47d5ec3ff4230f7
    log10_ = lambda x: np.log(x) / np.log(10).astype(np.float32)
    log_spec = 10.0 * log10_(np.maximum(amin, s))
    log_spec -= 10.0 * log10_(np.maximum(amin, np.max(s)))
    log_spec = np.maximum(log_spec, np.max(log_spec) - top_db)
    return log_spec

  @staticmethod
  def _mel_conversion(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    waveform = KWSDataLoader._reshape(waveform, sample_rate)
    spectogram = torch.tensor(KWSDataLoader._power_to_db(waveform.squeeze().numpy())).unsqueeze(0)
    spectogram = spectogram.unsqueeze(0)
    return spectogram

  @staticmethod
  def _preprocess(waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
    spectogram_t = KWSDataLoader._mel_conversion(waveform,sample_rate).numpy()
    return spectogram_t

  def postprocess(self, output: np.ndarray) -> Tuple[List[float], List[str]]:
    labels = pickle.load(open("./models/lable.pickle", 'rb'))
    scores = softmax(output)
    scores = np.squeeze(scores)
    ranks = np.argsort(scores)[::-1]
    top_five_output = ([scores[rank] for rank in ranks[0:5]], [labels[rank] for rank in ranks[0:5]])
    return top_five_output

  def get_data(self) -> List[tuple]:
    return self.processed_dataset
