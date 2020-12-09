import torch
import torchaudio
import torch.utils.data
import torch.nn.functional as F
from librosa.core import load, resample
from librosa.util import normalize
from pathlib import Path
import numpy as np
import random
from modules.helper_functions import files_to_list
from dataset.audio_augs import AudioAugs


class AudioDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, training_files, segment_length, sampling_rate, augment=True):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.audio_files = files_to_list(training_files)
        self.audio_files = [Path(training_files).parent / x for x in self.audio_files]
        random.seed(1234)
        random.shuffle(self.audio_files)
        self.augment = AudioAugs(sampling_rate) if augment else None

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]

        audio, sampling_rate = self.load_wav_to_torch(filename)
        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data
        
        # audio = audio / 32768.0
        return audio.unsqueeze(0)

    def __len__(self):
        return len(self.audio_files)

    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        data, sampling_rate = load(full_path, sr=self.sampling_rate)
        data = 0.95 * normalize(data)
        data =  torch.from_numpy(data).float()
        if self.augment:            
            data = self.augment(data)            
        return data, sampling_rate