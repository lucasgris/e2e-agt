import os
import warnings
import librosa
import nnAudio
import numpy as np
import torch
import torchaudio
from scipy.io import wavfile
from torchaudio.functional import pitch_shift
from omegaconf import DictConfig

import logging
logger = logging.getLogger(__name__)

DATA_PATH = "./data"

STR_MIDI_DICT = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
STR_NOTE_DICT = {0: "E2", 1: "A2", 2: "D3", 3: "G3", 4: "B3", 5: "E4"}
NOTE_OCTAVES = ['E2', 'F2', 'F#2', 'G2', 'G#2', 'A2', 'A#2', 'B2', 'C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4', 'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5', 'C6', 'C#6', 'D6']
CHROMATIC_SCALE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def fret_to_note(string, fret):
    note_idx = (NOTE_OCTAVES.index(string) + fret) 
    return NOTE_OCTAVES[note_idx]        

def note_to_onehot(note_octave):
    note_index = NOTE_OCTAVES.index(note_octave)
    onehot = [0] * len(NOTE_OCTAVES)
    onehot[note_index] = 1
    return onehot

def frets_to_notes(frets):
    num_strings = frets.shape[0]
    num_frames = frets.shape[1]
    notes = torch.zeros((num_frames, len(NOTE_OCTAVES)+1), dtype=torch.float32)
    notes[:, 0] = 1 # silence
    for s in range(num_strings):
        for t in range(num_frames):
            fret_number = frets[s, t].argmax().item()-1
            if fret_number < 0:
                continue
            note = fret_to_note(STR_NOTE_DICT[s], fret_number)
            note_index = NOTE_OCTAVES.index(note)+1
            notes[t, 0] = 0  
            notes[t, note_index] = 1
    return notes

class FeatureExtractor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = cfg.audio.device
        if cfg.audio.feature == "raw":
            self.feature_extractor = None
        else:
            if cfg.audio.feature_extractor == "nnAudio":
                if cfg.audio.feature == "chroma_stft":
                    self.feature_extractor = nnAudio.features.ChromaSTFT(
                        sr=cfg.audio.chroma_stft_params.sr,
                        n_fft=cfg.audio.chroma_stft_params.n_fft,
                        n_chroma=cfg.audio.chroma_stft_params.n_chroma,
                        hop_length=cfg.audio.chroma_stft_params.hop_length,
                    ).to(self.device)
                elif cfg.audio.feature == "cqt":
                    self.feature_extractor = nnAudio.features.CQT1992v2(
                        sr=cfg.audio.cqt_params.sr,
                        hop_length=cfg.audio.cqt_params.hop_length,
                        n_bins=cfg.audio.cqt_params.n_bins,
                        bins_per_octave=cfg.audio.cqt_params.bins_per_octave,
                    ).to(self.device)
                elif cfg.audio.feature == "spectrogram":
                    self.feature_extractor = nnAudio.features.STFT(
                        n_fft=cfg.audio.spectrogram_params.n_fft,
                        win_length=cfg.audio.spectrogram_params.win_length,
                        hop_length=cfg.audio.spectrogram_params.hop_length,
                    ).to(self.device)
                elif cfg.audio.feature == "melspectrogram":
                    self.feature_extractor = nnAudio.Spectrogram.MelSpectrogram(
                        sample_rate=cfg.audio.sr,
                        n_fft=cfg.audio.melspectrogram_params.n_fft,
                        win_length=cfg.audio.melspectrogram_params.win_length,
                        hop_length=cfg.audio.melspectrogram_params.hop_length,
                        n_mels=cfg.audio.melspectrogram_params.n_mels,
                        f_min=cfg.audio.melspectrogram_params.f_min,
                        f_max=cfg.audio.melspectrogram_params.f_max,
                    ).to(self.device)
                else:
                    raise ValueError(f"Invalid feature: {cfg.audio.feature}")
            elif cfg.audio.feature_extractor == "torchaudio":
                if cfg.audio.feature == "chroma_stft":
                    self.feature_extractor = torchaudio.transforms.ChromaSTFT(
                        sr=cfg.audio.chroma_stft_params.sr,
                        n_fft=cfg.audio.chroma_stft_params.n_fft,
                        n_chroma=cfg.audio.chroma_stft_params.n_chroma,
                        hop_length=cfg.audio.chroma_stft_params.hop_length,
                    ).to(self.device)
                elif cfg.audio.feature == "cqt":
                    self.feature_extractor = torchaudio.transforms.CQT(
                        sr=cfg.audio.cqt_params.sr,
                        hop_length=cfg.audio.cqt_params.hop_length,
                        n_bins=cfg.audio.cqt_params.n_bins,
                        bins_per_octave=cfg.audio.cqt_params.bins_per_octave,
                    ).to(self.device)
                elif cfg.audio.feature == "spectrogram":
                    self.feature_extractor = torchaudio.transforms.Spectrogram(
                        n_fft=cfg.audio.spectrogram_params.n_fft,
                        pad_mode="reflect",
                        center=True,
                        win_length=cfg.audio.spectrogram_params.win_length,
                        hop_length=cfg.audio.spectrogram_params.hop_length,
                    ).to(self.device)
                elif cfg.audio.feature == "melspectrogram":
                    self.feature_extractor = torchaudio.transforms.MelSpectrogram(
                        sample_rate=cfg.audio.sr,
                        center=True,
                        n_fft=cfg.audio.melspectrogram_params.n_fft,
                        win_length=cfg.audio.melspectrogram_params.win_length,
                        hop_length=cfg.audio.melspectrogram_params.hop_length,
                        n_mels=cfg.audio.melspectrogram_params.n_mels,
                        f_min=cfg.audio.melspectrogram_params.f_min,
                        f_max=cfg.audio.melspectrogram_params.f_max,
                    ).to(self.device)
                else:
                    raise ValueError(f"Invalid feature: {cfg.audio.feature}")
            elif cfg.audio.feature_extractor == "librosa" or cfg.audio.feature_extractor == "raw":
                self.feature_extractor = None
            else:
                raise ValueError(
                    f"Invalid feature extractor: {cfg.audio.feature_extractor}"
                )

    def __call__(self, x):
        if self.cfg.audio.feature_extractor == "raw":
            return
        if self.cfg.audio.feature_extractor == "librosa":
            if self.cfg.audio.feature == "chroma_stft":
                feature = np.abs(
                    librosa.feature.chroma_stft(
                        y=np.array(x),
                        sr=self.cfg.audio.chroma_stft_params.sr,
                        n_fft=self.cfg.audio.chroma_stft_params.n_fft,
                        n_chroma=self.cfg.audio.chroma_stft_params.n_chroma,
                        hop_length=self.cfg.audio.chroma_stft_params.hop_length,
                    )
                )
            elif self.cfg.audio.feature == "cqt":
                feature = np.abs(
                    librosa.cqt(
                        y=np.array(x),
                        sr=self.cfg.audio.cqt_params.sr,
                        hop_length=self.cfg.audio.cqt_params.hop_length,
                        n_bins=self.cfg.audio.cqt_params.n_bins,
                        bins_per_octave=self.cfg.audio.cqt_params.bins_per_octave,
                    )
                )
            elif self.cfg.audio.feature == "spectrogram":
                feature = np.abs(
                    librosa.stft(
                        y=np.array(x),
                        n_fft=self.cfg.audio.spectrogram_params.n_fft,
                        win_length=self.cfg.audio.spectrogram_params.win_length,
                        hop_length=self.cfg.audio.spectrogram_params.hop_length,
                    )
                )
            elif self.cfg.audio.feature == "melspectrogram":
                feature = np.abs(
                    librosa.feature.melspectrogram(
                        y=np.array(x),
                        sr=self.cfg.audio.sr,
                        n_fft=self.cfg.audio.melspectrogram_params.n_fft,
                        win_length=self.cfg.audio.melspectrogram_params.win_length,
                        hop_length=self.cfg.audio.melspectrogram_params.hop_length,
                        n_mels=self.cfg.audio.melspectrogram_params.n_mels,
                        f_min=self.cfg.audio.melspectrogram_params.f_min,
                        f_max=self.cfg.audio.melspectrogram_params.f_max,
                    )
                )
            else:
                raise ValueError(f"Invalid feature: {self.cfg.audio.feature}")
            feature = torch.from_numpy(feature)
            feature = feature.unsqueeze(0)
        else:
            feature = self.feature_extractor(x.to(self.device))
        return feature

def extract_features(
    cfg,
    feature_extractor,
    audio_path,
    feature_path,  
    start_sec=None,
    audio_load_sec=None,
    pitch_step=0,
    onset_feature_path=None,
    segment_path=None,
    force_reprocess=False,
    device="cpu",
):
    if not force_reprocess:
        if cfg.audio.feature_extractor == "raw":
            if os.path.isfile(segment_path):
                logger.info(
                    f"Segment already processed: audio_path={audio_path} start_sec={start_sec} audio_load_sec={audio_load_sec} segment_path={segment_path}"
                )
                return None
        else:
            if os.path.isfile(feature_path):
                try:
                    feature = torch.load(feature_path)
                    feature_shape = feature.shape
                    logger.info(
                        f"Segment already processed: audio_path={audio_path} start_sec={start_sec} audio_load_sec={audio_load_sec} feature_path={feature_path} feature_shape={feature_shape}"
                    )
                    return feature_shape
                except:
                    logger.warning(f"Failed to load feature: {feature_path}")

    if cfg.audio.loader == "torchaudio":
        sr = torchaudio.info(audio_path).sample_rate
        x, sr = torchaudio.load(
            audio_path,
            normalize=True,
            frame_offset=int(start_sec * sr) if start_sec is not None else 0,
            num_frames=int(audio_load_sec * sr) if audio_load_sec is not None else -1,
        )

        if cfg.audio.resample and sr != cfg.audio.sr:
            if cfg.audio.loader == "torchaudio":
                x = torchaudio.transforms.Resample(orig_freq=sr, new_freq=cfg.audio.sr)(
                    x
                )
                sr = cfg.audio.sr

        x = torch.mean(x, dim=0)
        x = x.to(device)
    elif cfg.audio.loader == "librosa":
        if segment_path is not None and os.path.isfile(segment_path):
            x, sr = librosa.load(
                segment_path, mono=True, sr=cfg.audio.sr if cfg.audio.resample else None
            )
        else:
            x, sr = librosa.load(
                audio_path,
                mono=True,
                sr=cfg.audio.sr if cfg.audio.resample else None,
                offset=start_sec if start_sec is not None else 0,
                duration=audio_load_sec if audio_load_sec is not None else None,
            )
        x = torch.from_numpy(x).to(device)
    else:
        raise ValueError(f"Invalid loader: {cfg.audio.loader}")

    if pitch_step != 0:
        x = pitch_shift(x.cuda(), sr, pitch_step).cpu()

    if segment_path and not os.path.isfile(segment_path):
        wavfile.write(segment_path, sr, np.array(x.cpu()))

    if onset_feature_path is not None:
        warnings.warn("Onset feature extraction is really slow")
        x, sr = librosa.load(audio_path, sr=cfg.audio.sr)
        onset_frames = librosa.onset.onset_detect(
            y=x, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1
        )
        onset_frames = np.array(onset_frames)
        onset_times = librosa.frames_to_time(onset_frames)
        onset_times = np.array(onset_times)
        np.save(onset_feature_path, onset_times)

    if cfg.audio.feature_extractor != "raw":
        feature = feature_extractor(x)
        torch.save(feature, feature_path)
        return feature.shape