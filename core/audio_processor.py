from locale import normalize
from typing import Union
import torchaudio
import torch
import librosa
from omegaconf import DictConfig, OmegaConf
import logging
import os
import numpy as np
# import pyloudnorm as pyln # TODO: implement loudness normalization

logger = logging.getLogger(__name__)


class AudioProcessor:

    def __init__(self, config: DictConfig, custom_feature_extractor: callable = None):
        self.config = config
        logger.info(
            "Init AudioProcessor with parameters: "
            + str(OmegaConf.to_container(self.config))
        )

        if custom_feature_extractor is not None:
            assert callable(custom_feature_extractor)
            logger.info("Overriding configuration with custom feature extractor")
            self.feature_extractor = custom_feature_extractor

        valid_features = [
            "spectrogram",
            "melspectrogram",
            "cqt",
            "chroma_stft",
            "mert",
            "raw",
        ]
        if self.config.feature not in valid_features:
            raise ValueError("Invalid Feature: " + str(self.config.feature))

        if self.config.feature == "raw":
            self.feature_extractor = lambda x: x
        elif self.config.feature == "mert":  # Not working
            from transformers import Wav2Vec2FeatureExtractor

            self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
                "m-a-p/MERT-v0", trust_remote_code=True
            )
            self.feature_extractor = self._mert_feature_extractor
        elif self.config.feature == "spectrogram":
            if self.config.loader == "torchaudio":
                self.feature_extractor = torchaudio.transforms.Spectrogram(
                    n_fft=self.config.spectrogram_params.n_fft,
                    pad_mode="reflect",  # TODO: add config
                    center=True,  # TODO: add config
                    win_length=self.config.spectrogram_params.win_length,
                    hop_length=self.config.spectrogram_params.hop_length,
                )
        elif self.config.feature == "melspectrogram":
            if self.config.loader == "torchaudio":
                self.feature_extractor = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.config.sr,
                    center=True,  # TODO: add config
                    n_fft=self.config.melspectrogram_params.n_fft,
                    win_length=self.config.melspectrogram_params.win_length,
                    hop_length=self.config.melspectrogram_params.hop_length,
                    n_mels=self.config.melspectrogram_params.n_mels,
                    f_min=self.config.melspectrogram_params.f_min,
                    f_max=self.config.melspectrogram_params.f_max,
                )
        elif self.config.feature == "cqt":

            def _cqt(x):
                return np.abs(
                    librosa.cqt(
                        np.array(x),
                        sr=self.config.cqt_params.sr,
                        hop_length=self.config.cqt_params.hop_length,
                        n_bins=self.config.cqt_params.n_bins,
                        bins_per_octave=self.config.cqt_params.bins_per_octave,
                    )
                )

            self.feature_extractor = _cqt
        elif self.config.feature == "chroma_stft":

            def _chroma_stft(x):
                return np.abs(
                    librosa.feature.chroma_stft(
                        np.array(x),
                        sr=self.config.chroma_stft_params.sr,
                        n_fft=self.config.chroma_stft_params.n_fft,
                        n_chroma=self.config.chroma_stft_params.n_chroma,
                        hop_length=self.config.chroma_stft_params.hop_length,
                    )
                )

            self.feature_extractor = _chroma_stft
        else:
            raise NotImplementedError

    def _mert_feature_extractor(
        self, x: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x = x.cpu().squeeze(1).numpy()
        x = self.mert_processor(x, sampling_rate=self.config.sr, return_tensors="pt")
        return x.input_values, x.attention_mask

    def wav2feature(self, x) -> Union[np.ndarray, torch.Tensor]:
        feature = self.feature_extractor(x)
        return feature

    def get_feature_from_audio_path(
        self, path: Union[os.PathLike, str]
    ) -> Union[np.ndarray, torch.Tensor]:
        return self.wav2feature(self.load_wav(path))

    def get_feature_from_audio(
        self, wav: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        return self.wav2feature(wav)

    def load_wav(
        self,
        path: Union[os.PathLike, str],
        start_sec: float = None,
        duration_sec: float = None,
    ) -> torch.Tensor:
        if os.path.exists(path) is False:
            raise FileNotFoundError(f"File not found: {path}")
        if self.config.loader == "torchaudio":
            if start_sec is not None or duration_sec is not None:
                sr = torchaudio.info(path).sample_rate
                wav, sr = torchaudio.load(
                    path,
                    frame_offset=int(start_sec * sr),
                    num_frames=int(duration_sec * sr),
                )
            else:
                wav, sr = torchaudio.load(
                    path
                )
            if self.config.mono:
                wav = torch.mean(wav, dim=0, keepdim=True)
            if self.config.resample and sr != self.config.sr:
                resample = torchaudio.transforms.Resample(
                    sr, self.config.sr
                )  # TODO: add config and move to transforms
                wav = resample(wav)
                sr = self.config.sr
        elif self.config.loader == "librosa":
            wav, sr = librosa.load(
                path,
                mono=self.config.mono,
                sr=self.config.sr if self.config.resample else None,
                offset=start_sec,
                duration=duration_sec,
            )
            if sr != self.config.sr:
                raise ValueError(
                    f"Invalid Sample Rate {sr} != {self.config.sr}. Set resample=True in config."
                )
            wav = torch.from_numpy(wav).unsqueeze(0)
        else:
            raise NotImplementedError
        if sr != self.config.sr:
            logger.warning(
                f"Audio sample rate {sr} is different from configuration: {self.config.sr} - cfg.resample={self.config.resample}"
            )
        return wav
