import ast
import os
import random
import logging
import warnings

warnings.filterwarnings("ignore")
from multiprocessing import Manager

import audiomentations
import torch
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig, OmegaConf
from torchaudio.functional import pitch_shift

from core.audio_processor import AudioProcessor
from core.guitar_effects import GuitarEffects
from utils.preprocess import *

logger = logging.getLogger(__name__)

# TODO refactor
class AGTDataset(Dataset):

    def __init__(
        self,
        cfg: DictConfig,
        audio_processor: AudioProcessor,
        train=False,
        valid=False,
        test=False,
    ):
        super().__init__()
        logger.info("Init Dataset with parameters: " + str(OmegaConf.to_container(cfg)))
        self.audio_processor = audio_processor

        self.train = train
        self.valid = valid
        self.test = test
        self.feature_size = cfg.feature_size
        self.augment_audio = cfg.audio_augmentation

        if not self.train:
            self.augment_audio = False
        if cfg.audio_augmentation:
            self.guitar_effects_prob = cfg.audio_augmentation.guitar_effects_prob or 0.5
            self.audio_augmentations_prob = (
                cfg.audio_augmentation.audio_augmentations_prob or 0.5
            )
            self.pitch_shift_steps = cfg.audio_augmentation.pitch_shift_steps

            self.guitar_effects = GuitarEffects(**cfg.audio_augmentation.GuitarEffects)
            audiomentations_list = []
            for cls_name in cfg.audio_augmentation.AudioAugmentations:
                Cls = getattr(audiomentations, cls_name)
                audiomentations_list.append(
                    Cls(**cfg.audio_augmentation.AudioAugmentations[cls_name])
                )
            self.audiomentations = audiomentations.Compose(audiomentations_list)

        self.seed = cfg.random_seed
        if self.seed is not None:
            # fix random seeds for reproducibility
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(self.seed)

        if self.train:
            logger.info("Setting up train dataset")
            assert os.path.isfile(
                cfg.train_csv_file
            ), f"{cfg.train_csv_file} does not exist"
            self.data_csv = cfg.train_csv_file
        elif self.valid:
            logger.info("Setting up valid dataset")
            assert os.path.isfile(
                cfg.valid_csv_file
            ), f"{cfg.valid_csv_file} does not exist"
            self.data_csv = cfg.valid_csv_file
        elif self.test:
            logger.info(
                "Setting up test dataset"
            ), f"{cfg.test_csv_file} does not exist"
            assert os.path.isfile(cfg.test_csv_file)
            self.data_csv = cfg.test_csv_file
        else:
            logger.info("Setting up dummy dataset")
            assert os.path.isfile(
                cfg.dummy_csv_file
            ), f"{cfg.dummy_csv_file} does not exist"
            self.data_csv = cfg.dummy_csv_file

        logger.info(f"Reading metadata from csv: {self.data_csv}")
        self.metadata = pd.read_csv(self.data_csv, sep=";", on_bad_lines='warn')
        self.data_list = self.metadata.values
        logger.debug(self.metadata)
        logger.info(f"Succesfuly loaded metadata: data length is {len(self.data_list)}")
        
        self.num_strings = cfg.num_strings
        self.fret_size = cfg.fret_size

        self.insert_ffm = cfg.insert_ffm
        if self.insert_ffm:
            self.ffm_delete_frames_prob = cfg.ffm.delete_frames_prob
            self.ffm_delete_prob = cfg.ffm.delete_prob
            self.ffm_noise_factor = cfg.ffm.noise_factor
            self.ffm_perturb = cfg.ffm.perturb_on_test
            self.ffm_map_kernel = (
                torch.tensor(list(cfg.ffm.map_kernel))
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
            )
            if self.ffm_map_kernel.shape[-1] % 2 == 0:
                raise ValueError(
                    "ffm_map_kernel should be an odd-sized kernel, got shape: {}".format(
                        self.ffm_map_kernel.shape
                    )
                )
            assert (
                len(self.ffm_map_kernel.shape) == 4
            ), "ffm_map_kernel should be a 4D kernel"

        self.cache_data_mem = cfg.cache_data_mem
        self.features_dir = self.audio_dir = self.segments_dir = None

        if self.cache_data_mem:
            logger.info(f"Data caching is set to true")
            logger.warning(f"Data caching in memory is an experimental feature")
            # self._cache_manager = Manager()
            # self._cache = self._cache_manager.dict()
            self._cache = {}
        if cfg.skip_cache:
            logger.info(
                f"Disabling data cache: cfg.cache_data_mem = cfg.features_dir = None"
            )
            cfg.cache_data_mem = False
            cfg.features_dir = None

        if cfg.segments_dir is not None:
            self.segments_dir = cfg.segments_dir
            logger.info(
                f"Will generate features dynamicaly from segments at {self.segments_dir}"
            )
        elif cfg.audio_dir is not None:
            self.audio_dir = cfg.audio_dir
            logger.info(
                f"Will generate features dynamicaly from raw audio at {self.audio_dir}"
            )
        else:
            raise ValueError(
                "Should provide either cfg.segments_dir or cfg.audio_dir to load data"
            )

    def _get_ffm_map(self, frets_ohe):
        # We have to discard the silence and first string class to create the map of teh hand in the fretboard
        frets_map_raw = frets_ohe[:, :, 2:].clone().mean(0).unsqueeze(0)
        frets_map_binarized = (frets_map_raw > 0).float()
        
        if self.ffm_delete_prob > 0 and random.random() < self.ffm_delete_prob:
            frets_map = torch.zeros_like(frets_map_raw).squeeze(0)
            return frets_map
        
        frets_map = torch.nn.functional.conv2d(
            frets_map_binarized.unsqueeze(1),
            self.ffm_map_kernel,
            padding=[
                self.ffm_map_kernel.size(2) // 2,
                self.ffm_map_kernel.size(3) // 2,
            ],
            stride=self.ffm_map_kernel.size(2),
        ).squeeze(1)
        
        if self.ffm_noise_factor > 0:
            frets_map = self.add_noise(frets_map, self.ffm_noise_factor)
        frets_map = frets_map.squeeze()
        if self.ffm_delete_frames_prob > 0:
            # mask in the time dimension
            mask = torch.rand(frets_map.shape[0]) > self.ffm_delete_frames_prob
            frets_map = frets_map * mask.unsqueeze(-1)
        return frets_map

    def _resize_tensor(tensor, target_size, time_dim=1):
        raise NotImplementedError

    def _pitch_shift(self, x):
        step = random.choice(self.pitch_shift_steps)
        return pitch_shift(x, self.sample_rate, n_steps=step), step

    def add_noise(self, x, noise_factor):
        noise = torch.randn_like(x) * noise_factor
        noisy_x = x + noise
        return noisy_x.clamp(-1, 1)

    def _augment(self, x, sr):
        shape_before = x.shape
        if not self.train:
            return x
        if type(x) == torch.Tensor:
            x = x.detach().cpu().numpy()
        if random.random() < self.guitar_effects_prob:
            x = self.guitar_effects(x, sample_rate=sr)
            assert (
                x.shape == shape_before
            ), f"Shape before {shape_before} and after {x.shape} augmentation does not match"
        if random.random() < self.audio_augmentations_prob:
            if len(x.shape) > 1:
                x = np.squeeze(x, 0)
            x = self.audiomentations(x, sr)
            if len(x.shape) == 1:
                x = np.expand_dims(x, 0)
            assert (
                x.shape == shape_before
            ), f"Shape before {shape_before} and after {x.shape} augmentation does not match"
        x = torch.from_numpy(x)
        return x

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        raise NotImplementedError


class AGTFrameDataset(AGTDataset):
    """
    Dataset class for frame-wise guitar transcription.

    Metadata format: semi-colon separated csv file
    Metadata columns:
        feature_path: str - path to the feature pt file (optional)
        segment_path: str - path to the segment wav file
        target_path: str - path to the target pt file
    """

    def __init__(
        self,
        cfg: DictConfig,
        audio_processor: AudioProcessor,
        train=False,
        valid=False,
        test=False,
    ):

        super().__init__(cfg, audio_processor, train, valid, test)
        self.note_octaves = list(cfg.note_octaves)
        self.chromatic_scale = list(cfg.chromatic_scale)
        self.target_len_frames_upsample_method = cfg.target_len_frames_upsample_method

    def _load_targets(self, idx):  # TODO: make it configurable
        targets = {}
        for target in [
            "frets_path",
            "notes_path",
            "onsets_path",
            "tab_path",
        ]:
            target_path = self.metadata.iloc[idx][target]
            try:
                target_tensor = torch.load(target_path, map_location="cpu").float()
            except Exception as e:
                logger.error(
                    f"Error loading targets for '{target_path}': {e}\n{self.metadata.iloc[idx]}"
                )
                raise e
            targets[target.replace("_path", "")] = target_tensor

        return targets

    def _load_features(self, idx):
        file_name = self.metadata.iloc[idx]["file_name"]
        if "start_sec" not in self.metadata.columns:
            start_sec = 0
        else:
            start_sec = self.metadata.iloc[idx]["start_sec"]
        duration_sec = self.metadata.iloc[idx]["duration"]
        if "feature_path" in self.metadata.columns:
            feature_path = self.metadata.iloc[idx]["feature_path"]
        segment_path = self.metadata.iloc[idx]["segment_path"]

        # load_from_audio = self.features_dir is None

        if "pitch_step" in self.metadata.columns:
            pitch_step = self.metadata.iloc[idx]["pitch_step"]
        else:
            pitch_step = 0

        if self.cache_data_mem and file_name in self._features_cache:
            x = self._features_cache[file_name]
        else:
            if self.features_dir is not None:  # Faster
                logger.debug(f"Loading feature from {feature_path}")
                try:
                    x = torch.load(feature_path)
                except Exception as e:
                    logger.error(
                        f"Error loading feature {feature_path}. Will load from audio."
                    )
                    load_from_audio = True
            else:
                load_from_audio = True

        if load_from_audio:
            if segment_path is not None:
                wav = self.audio_processor.load_wav(segment_path)
                if self.augment_audio:
                    if self.pitch_shift_steps is not None and (
                        pitch_step is None or pitch_step == 0
                    ):
                        wav, online_pitch_shift = self._pitch_shift(wav)
                    wav = self._augment(wav, self.audio_processor.config.sr)
                x = self.audio_processor.get_feature_from_audio(wav)

            elif self.audio_dir is not None:  # Slow
                audio_path = os.path.join(self.audio_dir, file_name + ".wav")
                wav = self.audio_processor.load_wav(
                    audio_path, start_sec=start_sec, duration_sec=duration_sec
                )
                if self.augment_audio:
                    if self.pitch_shift_steps is not None and (
                        pitch_step is None or pitch_step == 0
                    ):
                        wav, online_pitch_shift = self._pitch_shift(wav)
                    wav = self._augment(wav, self.audio_processor.config.sr)
                x = self.audio_processor.get_feature_from_audio(wav)

            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)

            if self.features_dir is not None:
                os.remove(feature_path)
                torch.save(x, feature_path)

        if self.cache_data_mem:
            logger.debug(f"Caching feature {file_name} in memory")
            self._features_cache[file_name] = x

        return x

    def __getitem__(self, idx):
        file_name = self.metadata.iloc[idx]["file_name"]

        x = self._load_features(idx)

        targets = self._load_targets(idx)
        tab = targets["tab"]  # (num_strings, num_frames, num_frets)
        frets = targets["frets"]  # (num_strings, num_frames, num_frets)
        notes = targets["notes"]  # (num_frames, num_notes)
        onsets = targets["onsets"]  # (num_strings, num_frames)  # NOT USED, for onsets prediction we use "tab"

        if (
            len(tab) > self.num_strings
            or len(frets) > self.num_strings
            or len(onsets) > self.num_strings
        ):
            tab = tab[: self.num_strings, :, :]
            frets = frets[: self.num_strings, :, :]
            onsets = onsets[: self.num_strings, :]
            logger.warning(
                f"Detected more than {self.num_strings} strings in {file_name}. Trimming to {self.num_strings} strings"
            )

        if self.insert_ffm:
            ffm = self._get_ffm_map(frets)
        else:
            ffm = torch.tensor([])

        # TODO: create a method to handle size
        if self.feature_size is not None and len(x.shape) == 3:
            if x.shape[2] > self.feature_size:
                logger.warning(
                    f"Cutting feature {file_name} of size {x.shape[2]} to {self.feature_size} frames"
                )
                x = x[:, :, : self.feature_size]
            elif x.shape[2] < self.feature_size:
                x = torch.nn.functional.pad(x, (0, self.feature_size - x.shape[2]))
        elif self.feature_size is not None and len(x.shape) == 2:
            if x.shape[1] > self.feature_size:
                logger.warning(
                    f"Cutting feature {file_name} of size {x.shape[1]} to {self.feature_size} frames"
                )
                x = x[:, : self.feature_size]
            elif x.shape[1] < self.feature_size:
                x = torch.nn.functional.pad(x, (0, self.feature_size - x.shape[1]))

        if self.target_len_frames is not None:
            if tab.shape[1] > self.target_len_frames:
                logger.warning(
                    f"Cutting tab {file_name} of size {tab.shape[1]} to {self.target_len_frames} frames"
                )
                tab = tab[:, : self.target_len_frames, :]
            elif tab.shape[1] < self.target_len_frames:
                if self.target_len_frames_upsample_method == "pad":
                    tab = torch.nn.functional.pad(
                        tab, (0, 0, 0, self.target_len_frames - tab.shape[1], 0, 0)
                    )
                elif self.target_len_frames_upsample_method == "interpolate":
                    # TODO : not sure if this is the best way to interpolate
                    tab = torch.nn.functional.interpolate(
                        tab.transpose(1, 2),
                        size=self.target_len_frames,
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)
            if frets.shape[1] > self.target_len_frames:
                logger.warning(
                    f"Cutting frets {file_name} of size {frets.shape[1]} to {self.target_len_frames} frames"
                )
                frets = frets[:, : self.target_len_frames, :]
            elif frets.shape[1] < self.target_len_frames:
                if self.target_len_frames_upsample_method == "pad":
                    frets = torch.nn.functional.pad(
                        frets, (0, 0, 0, self.target_len_frames - frets.shape[1], 0, 0)
                    )
                elif self.target_len_frames_upsample_method == "interpolate":
                    # TODO : not sure if this is the best way to interpolate
                    frets = torch.nn.functional.interpolate(
                        frets.transpose(1, 2),
                        size=self.target_len_frames,
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)
            if notes.shape[0] > self.target_len_frames:
                logger.warning(
                    f"Cutting notes {file_name} of size {notes.shape[0]} to {self.target_len_frames} frames"
                )
                notes = notes[: self.target_len_frames, :]
            elif notes.shape[0] < self.target_len_frames:
                if self.target_len_frames_upsample_method == "pad":
                    notes = torch.nn.functional.pad(
                        notes, (0, 0, 0, self.target_len_frames - notes.shape[0])
                    )
                elif self.target_len_frames_upsample_method == "interpolate":
                    notes = (
                        torch.nn.functional.interpolate(
                            notes.unsqueeze(0).transpose(1, 2),
                            size=self.target_len_frames,
                            mode="linear",
                        )
                        .transpose(1, 2)
                        .squeeze(0)
                    )
            if onsets.shape[1] > self.target_len_frames:
                logger.warning(
                    f"Cutting onsets {file_name} of size {onsets.shape[1]} to {self.target_len_frames} frames"
                )
                onsets = onsets[:, : self.target_len_frames]
            elif onsets.shape[1] < self.target_len_frames:
                if self.target_len_frames_upsample_method == "pad":
                    onsets = torch.nn.functional.pad(
                        onsets, (0, self.target_len_frames - onsets.shape[1])
                    )
                elif self.target_len_frames_upsample_method == "interpolate":
                    onsets = (
                        torch.nn.functional.interpolate(
                            onsets.unsqueeze(0).unsqueeze(0),
                            size=self.target_len_frames,
                            mode="linear",
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )

        return (
            {"x": x, "ffm": ffm},
            {"tab": tab, "frets": frets, "notes": notes, "onsets": onsets},
        )

    def __len__(self):
        return len(self.data_list)

    def get_train_dataloader(
        cfg: DictConfig, audio_processor: AudioProcessor
    ) -> DataLoader:
        return DataLoader(
            dataset=AGTFrameDataset(cfg, audio_processor, train=True),
            batch_size=cfg.train_batch_size,
            shuffle=cfg.train_shuffle,
            persistent_workers=True if cfg.num_workers > 0 else False,
            pin_memory=False,
            num_workers=cfg.num_workers,
        )

    def get_valid_dataloader(
        cfg: DictConfig, audio_processor: AudioProcessor
    ) -> DataLoader:
        return DataLoader(
            dataset=AGTFrameDataset(cfg, audio_processor, train=False, valid=True),
            batch_size=cfg.valid_batch_size,
            persistent_workers=True if cfg.num_workers > 0 else False,
            pin_memory=False,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

    def get_test_dataloader(
        cfg: DictConfig, audio_processor: AudioProcessor
    ) -> DataLoader:
        return DataLoader(
            dataset=AGTFrameDataset(cfg, audio_processor, train=False, test=True),
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=0,
        )


class AGTSequenceDataset(AGTFrameDataset):

    def __init__(
        self,
        cfg: DictConfig,
        audio_processor: AudioProcessor,
        train=True,
        valid=False,
        test=False,
    ):
        super().__init__(cfg, audio_processor, train, valid, test)
        self.load_notes = cfg.load_notes
        self.other_string_token = cfg.other_string_token

        # if cfg.data_augmentation:
        #     self.data_augmentation = cfg.data_augmentation
        #     self.csv_samples = cfg.data_augmentation.csv_samples
        #     self.csv_samples = pd.read_csv(self.csv_samples, sep=";", index_col=False)
        #     self.data_augmentation_prob = cfg.data_augmentation.data_augmentation_prob
        #     self.min_seq_len = cfg.data_augmentation.min_seq_len
        #     self.max_seq_len = cfg.data_augmentation.max_seq_len
        #     self.audio_duration = cfg.data_augmentation.audio_duration

    def _load_targets(
        self, idx, pitch_shift=0, return_frets=False, load_notes=False, notes_seq_mctc=True
    ):
        frets_seq = [
            ast.literal_eval(self.metadata.iloc[idx]["fret_seq_0"]),
            ast.literal_eval(self.metadata.iloc[idx]["fret_seq_1"]),
            ast.literal_eval(self.metadata.iloc[idx]["fret_seq_2"]),
            ast.literal_eval(self.metadata.iloc[idx]["fret_seq_3"]),
            ast.literal_eval(self.metadata.iloc[idx]["fret_seq_4"]),
            ast.literal_eval(self.metadata.iloc[idx]["fret_seq_5"]),
        ]

        if load_notes:
            if notes_seq_mctc:
                notes_seq = ast.literal_eval(self.metadata.iloc[idx]["notes_sequence"])
                notes_seq_mat = np.zeros(
                    (len(notes_seq), len(self.note_octaves) + 1)
                )  # add one line for the new blank category (blank,not blank) (MCTC)
                for i, seq in enumerate(notes_seq):
                    for note in seq:
                        note_idx = (
                            self.note_octaves.index(note) + 1
                        )  # +1 to avoid 0 (epsilon token)
                        notes_seq_mat[i][note_idx + 1] = 1
                notes_seq_mat = notes_seq_mat.T  # (num_notes, len_seq)
                notes_seq_mat = torch.tensor(notes_seq_mat).float()
            else:
                raise NotImplementedError
        else:
            notes_seq_mat = torch.tensor([])

        # 0 is the blank token (silence)
        # 1 is the open string
        # [2, fret_size+1] are the frets
        # fret_size+2 is the other token
        # Add 1 to frets_seq to avoid 0 (blank token)
        # frets_seq = [torch.tensor(seq) + 1 
        #              for seq in frets_seq]  # TODO, map symbols from config
        def _map_fret(fret_seq_string):
            for fret in fret_seq_string:
                if not self.other_string_token and fret == '-': continue
                if fret == '-':
                    yield self.fret_size + 2
                else:
                    yield fret + 1
        frets_seq = [torch.tensor(list(_map_fret(seq))) for seq in frets_seq]            

        if pitch_shift != 0:
            raise NotImplementedError  # It is recommended to pitch shift in preprocessing

        if return_frets:
            frets = torch.load(self.metadata.iloc[idx]["frets_path"], map_location="cpu").float()
            return frets, frets_seq, notes_seq_mat
        return frets_seq, notes_seq_mat
        
    # def _generate_random_sample(self):
    #     audio = np.zeros((self.audio_processor.config.sr * self.audio_duration))
    #     for i in range(self.num_strings):
    #         fret_seq = random.choices(
    #             range(self.fret_size + 2), k=random.randint(self.min_seq_len, self.max_seq_len)
    #         )
    #         fret_seq = [str(fret) for fret in fret_seq]
    #         fret_seq = "-".join(fret_seq)
    #         for fret in fret_seq.split("-"):
    #             if fret != "-":
    #                 audio += self.audio_processor.generate_audio(fret, i)
    
    def __getitem__(self, idx):
        # if cfg.data_augmentation and random.random() < self.data_augmentation_prob:
        #     feature, ffm, frets_seq, notes_seq = self._generate_random_sample()
        #     return {
        #         "x": feature,
        #         "ffm": ffm,
        #         "frets_seq": frets_seq,
        #         "notes_seq": notes_seq,
        #     }

        file_name = self.metadata.iloc[idx]["file_name"]
        feature = self._load_features(idx)
        
        if self.feature_size is not None and len(feature.shape) == 3:
            if feature.shape[2] > self.feature_size:
                logger.warning(
                    f"Cutting feature {file_name} of size {feature.shape[2]} to {self.feature_size} frames"
                )
                feature = feature[:, :, : self.feature_size]
            elif feature.shape[2] < self.feature_size:
                feature = torch.nn.functional.pad(
                    feature, (0, self.feature_size - feature.shape[2])
                )
        elif self.feature_size is not None and len(feature.shape) == 2:
            if feature.shape[1] > self.feature_size:
                logger.warning(
                    f"Cutting feature {file_name} of size {feature.shape[1]} to {self.feature_size} frames"
                )
                feature = feature[:, : self.feature_size]
            elif feature.shape[1] < self.feature_size:
                feature = torch.nn.functional.pad(
                    feature, (0, self.feature_size - feature.shape[1])
                )

        if self.insert_ffm:
            frets, frets_seq, notes_seq = self._load_targets(idx, return_frets=True)
            ffm = self._get_ffm_map(frets)
        else:
            frets_seq, notes_seq = self._load_targets(idx, return_frets=False, load_notes=self.load_notes)
            ffm = torch.tensor([])

        if sum([len(seq) for seq in frets_seq]) == 0:
            logger.warning(f"Skipping item with zero frets")
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        return {
            "x": feature,
            "ffm": ffm,
            "frets_seq": frets_seq,
            "notes_seq": notes_seq,
        }

    def __len__(self):
        return len(self.data_list)

    def collate_batch_ctc_mctc(batch):
        num_strings = len(batch[0]["frets_seq"])

        features_collated_batch = []
        ffm_collated_batch = []

        fret_seq_collated_batch = [[] for _ in range(num_strings)]
        fret_target_lengths_batch = []
        notes_seq_collated_batch = []
        notes_target_lengths_batch = []

        max_len_notes_seq = max([data["notes_seq"].shape[-1] for data in batch])

        for data in batch:
            # features_collated_batch.append(data["x"])
            max_len_features = max([data["x"].shape[-1] for data in batch])
            features_collated = torch.nn.functional.pad(
                data["x"], (0, max_len_features - data["x"].shape[-1])
            )
            features_collated_batch.append(features_collated)
            ffm_collated_batch.append(data["ffm"])
            # pad notes_seq to max_len_notes_seq
            notes_seq_collated = torch.nn.functional.pad(
                data["notes_seq"], (0, max_len_notes_seq - data["notes_seq"].shape[-1])
            )
            for i in range(max_len_notes_seq):
                if notes_seq_collated[:, i].sum() == 0:
                    notes_seq_collated[0, i] = (
                        1  # add blank token (MCTC). Only MCTC is supported for now.
                    )
            notes_seq_collated_batch.append(
                notes_seq_collated # (num_notes+1, max_len_notes_seq)
            )  
            for i in range(num_strings):
                fret_seq_collated_batch[i].append(data["frets_seq"][i])
            fret_target_lengths_batch.append(
                torch.tensor([len(seq) for seq in data["frets_seq"]], dtype=torch.long)
            )
            notes_target_lengths_batch.append(
                torch.tensor(max_len_notes_seq, dtype=torch.long)  # maxlen or len(notes_seq)?
            )

        features_collated_batch = torch.stack(features_collated_batch)
        ffm_collated_batch = torch.stack(ffm_collated_batch)
        fret_target_lengths_batch = torch.stack(fret_target_lengths_batch).transpose(
            0, 1
        )  # S x B
        notes_target_lengths_batch = torch.stack(notes_target_lengths_batch)
        fret_seq_collated_batch = [
            pad_sequence(fret_seq_string, batch_first=True, padding_value=0)
            for fret_seq_string in fret_seq_collated_batch
        ]  # S x B x L
        notes_seq_collated_batch = torch.stack(notes_seq_collated_batch)  # B x S x L
        
        return (
            {"x": features_collated_batch, "ffm": ffm_collated_batch},
            {
                "frets_seq": fret_seq_collated_batch,
                "notes_seq": notes_seq_collated_batch,
                "fret_target_lengths": fret_target_lengths_batch,
                "notes_target_lengths": notes_target_lengths_batch,
            },
        )

    def get_train_dataloader(
        cfg: DictConfig, audio_processor: AudioProcessor
    ) -> DataLoader:
        return DataLoader(
            dataset=AGTSequenceDataset(cfg, audio_processor, train=True),
            batch_size=cfg.train_batch_size,
            collate_fn=AGTSequenceDataset.collate_batch_ctc_mctc,  # Only CTC and MCTC is supported for now
            shuffle=cfg.train_shuffle,
            num_workers=cfg.num_workers,
        )

    def get_valid_dataloader(
        cfg: DictConfig, audio_processor: AudioProcessor
    ) -> DataLoader:
        return DataLoader(
            dataset=AGTSequenceDataset(cfg, audio_processor, train=False, valid=True),
            batch_size=cfg.valid_batch_size,
            collate_fn=AGTSequenceDataset.collate_batch_ctc_mctc,
            num_workers=0,
        )

    def get_test_dataloader(
        cfg: DictConfig, audio_processor: AudioProcessor
    ) -> DataLoader:
        return DataLoader(
            dataset=AGTSequenceDataset(cfg, audio_processor, train=False, test=True),
            batch_size=cfg.test_batch_size,
            collate_fn=AGTSequenceDataset.collate_batch_ctc_mctc,
            shuffle=False,
            num_workers=0,
        )

    def get_dummy_dataloader(
        cfg: DictConfig, audio_processor: AudioProcessor
    ) -> DataLoader:
        return DataLoader(
            dataset=AGTSequenceDataset(cfg, audio_processor, train=True),
            batch_size=2,
            collate_fn=(
                AGTSequenceDataset.collate_batch_ctc_mctc if cfg.sequence else None
            ),
            shuffle=False,
            num_workers=0,
        )


class AGTContrastiveDataset(Dataset):

    def __init__(
        self,
        cfg: DictConfig,
        audio_processor: AudioProcessor,
        train=True,
        valid=False,
        test=False,
    ):
        super().__init__()
        logger.info("Init Dataset with parameters: " + str(OmegaConf.to_container(cfg)))
        self.audio_processor = audio_processor

        self.train = train
        self.valid = valid
        self.test = test
        self.feature_size = cfg.feature_size
        self.n_negative_samples = cfg.n_negative_samples
        self.augment_audio = cfg.audio_augmentation
        self.positive_augmentation_prob = cfg.positive_augmentation_prob
        self.negative_augmentation_prob = cfg.negative_augmentation_prob

        self.audio_augmentations_prob = (
            cfg.audio_augmentation.audio_augmentations_prob or 0.5
        )
        self.guitar_effects_prob = cfg.audio_augmentation.guitar_effects_prob or 0.5
        # if not self.train:
        #     self.augment_audio = False
        if cfg.audio_augmentation:
            self.pitch_shift_steps = cfg.audio_augmentation.pitch_shift_steps

        self.guitar_effects = GuitarEffects(**cfg.audio_augmentation.GuitarEffects)
        audiomentations_list = []
        for cls_name in cfg.audio_augmentation.AudioAugmentations:
            Cls = getattr(audiomentations, cls_name)
            audiomentations_list.append(
                Cls(**cfg.audio_augmentation.AudioAugmentations[cls_name])
            )
        self.audiomentations = audiomentations.Compose(audiomentations_list)

        self.seed = cfg.random_seed
        if self.seed is not None:
            # fix random seeds for reproducibility
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(self.seed)

        if self.train:
            logger.info("Setting up train dataset")
            assert os.path.isfile(
                cfg.train_csv_file
            ), f"{cfg.train_csv_file} does not exist"
            self.data_csv = cfg.train_csv_file
        elif self.valid:
            logger.info("Setting up valid dataset")
            assert os.path.isfile(
                cfg.valid_csv_file
            ), f"{cfg.valid_csv_file} does not exist"
            self.data_csv = cfg.valid_csv_file
        elif self.test:
            logger.info(
                "Setting up test dataset"
            ), f"{cfg.test_csv_file} does not exist"
            assert os.path.isfile(cfg.test_csv_file)
            self.data_csv = cfg.test_csv_file
        else:
            logger.info("Setting up dummy dataset")
            assert os.path.isfile(
                cfg.dummy_csv_file
            ), f"{cfg.dummy_csv_file} does not exist"
            self.data_csv = cfg.dummy_csv_file

        logger.info(f"Reading metadata from csv: {self.data_csv}")
        self.metadata = pd.read_csv(self.data_csv, sep=";", index_col=False)

        if "pitch_step" in self.metadata.columns:
            if len(self.metadata["pitch_step"].unique()) > 0 and self.pitch_shift_steps:
                logger.warning(
                    "Will perform online pitch shift but there are audios with pitch shift in the provided data."
                )

        if not self.train:
            if "pitch_step" in self.metadata.columns:
                self.metadata = self.metadata[self.metadata.pitch_step == 0]
            assert self.metadata.shape[0] > 0, f"No data found in {self.data_csv}"
        logger.debug(self.metadata)

        self.data_list = self.metadata.values
        logger.info(f"Succesfuly loaded data: data length is {len(self.data_list)}")

        self.cache_data_mem = cfg.cache_data_mem
        self.features_dir = self.audio_dir = self.segments_dir = None

        if self.cache_data_mem:
            logger.info(f"Data caching is set to true")
            logger.warning(f"Data caching in memory is an experimental feature")
            # self._cache_manager = Manager()
            # self._cache = self._cache_manager.dict()
            self._cache = {}
        if cfg.skip_cache:
            logger.info(
                f"Disabling data cache: cfg.cache_data_mem = cfg.features_dir = None"
            )
            cfg.cache_data_mem = False
            cfg.features_dir = None

        if cfg.segments_dir is not None:
            self.segments_dir = cfg.segments_dir
            logger.info(
                f"Will generate features dynamicaly from segments at {self.segments_dir}"
            )
        elif cfg.audio_dir is not None:
            self.audio_dir = cfg.audio_dir
            logger.info(
                f"Will generate features dynamicaly from raw audio at {self.audio_dir}"
            )
        else:
            raise ValueError(
                "Should provide either cfg.segments_dir or cfg.audio_dir to load data"
            )

        if cfg.preload_all_audios:
            if not self.cache_data_mem:
                logger.warning(
                    f"cfg.preload_all_audios is set to True but cfg.cache_data_mem is False. "
                    "Will preload all audios in memory."
                )
                cfg.cache_data_mem = True
            logger.info(f"Preloading all audios")
            self._preload_all_audios()

    def _preload_all_audios(self):
        for idx in tqdm(range(len(self.data_list))):
            self._get_item(idx)

    def add_noise(self, audio, noise_factor):
        noise = torch.randn_like(audio) * noise_factor
        noisy_audio = audio + noise
        return noisy_audio.clamp(-1, 1)

    def generate_positive(self, x, sr):
        if type(x) == torch.Tensor:
            x = x.detach().cpu().numpy()
        if random.random() < self.positive_augmentation_prob:
            if random.random() < self.guitar_effects_prob:
                x = self.guitar_effects(x, sample_rate=sr)
            if random.random() < self.audio_augmentations_prob:
                x = np.expand_dims(self.audiomentations(np.squeeze(x, 0), sr), 0)
        return x

    def generate_negative_samples(self):
        negative_samples = []
        for i in range(self.n_negative_samples):
            x = self._get_item(random.randint(0, len(self.data_list) - 1))
            if type(x) == torch.Tensor:
                x = x.detach().cpu().numpy()
            if random.random() < self.negative_augmentation_prob:
                if random.random() < self.guitar_effects_prob:
                    x = self.guitar_effects(
                        x, sample_rate=self.audio_processor.config.sr
                    )
                if random.random() < self.audio_augmentations_prob:
                    x = np.expand_dims(
                        self.audiomentations(
                            np.squeeze(x, 0), self.audio_processor.config.sr
                        ),
                        0,
                    )
            negative_samples.append(x)
        return negative_samples

    def cut_or_pad_if_necessary(self, x):
        if self.feature_size is not None and len(x.shape) == 2:
            if x.shape[1] > self.feature_size:
                x = x[:, : self.feature_size]
                logger.warning(
                    f"Cutting feature of size {x.shape[1]} to {self.feature_size} frames"
                )
            elif x.shape[1] < self.feature_size:
                x = torch.nn.functional.pad(x, (0, self.feature_size - x.shape[1]))
        return x

    def _get_item(self, idx):
        file_name = self.metadata.iloc[idx]["file_name"]
        start_sec = self.metadata.iloc[idx]["start_sec"]
        duration_sec = self.metadata.iloc[idx]["duration"]
        segment_path = self.metadata.iloc[idx]["segment_path"]
        pitch_step = self.metadata.iloc[idx]["pitch_step"]
        online_pitch_shift = 0

        if self.segments_dir is not None:
            if self.cache_data_mem and segment_path in self._cache:
                try:
                    x = self._cache[segment_path]
                except Exception as e:
                    logger.error(f"Error loading {segment_path} from cache: {e}")
                    x = self.audio_processor.load_wav(segment_path)
            else:
                x = self.audio_processor.load_wav(segment_path)
                if self.cache_data_mem:
                    self._cache[segment_path] = x
            if self.augment_audio:
                if self.pitch_shift_steps is not None and (
                    pitch_step is None or pitch_step == 0
                ):
                    x, online_pitch_shift = self._pitch_shift(x)

        elif self.audio_dir is not None:  # Slow
            if self.cache_data_mem and file_name in self._cache:
                try:
                    x = self._cache[file_name]
                except Exception as e:
                    logger.error(f"Error loading {file_name} from cache: {e}")
                    audio_path = os.path.join(self.audio_dir, file_name + ".wav")
                    x = self.audio_processor.load_wav(
                        audio_path, start_sec=start_sec, duration_sec=duration_sec
                    )
                    self._cache[file_name] = x
            else:
                audio_path = os.path.join(self.audio_dir, file_name + ".wav")
                x = self.audio_processor.load_wav(
                    audio_path, start_sec=start_sec, duration_sec=duration_sec
                )
                if self.cache_data_mem:
                    self._cache[file_name] = x
            if self.augment_audio:
                if self.pitch_shift_steps is not None and (
                    pitch_step is None or pitch_step == 0
                ):
                    x, online_pitch_shift = self._pitch_shift(x)
        else:
            raise Exception(f"Invalid data configuration:", self)

        x = self.cut_or_pad_if_necessary(x)
        return x

    def __getitem__(self, idx):
        x = self._get_item(idx)
        positive_sample = self.generate_positive(x, self.audio_processor.config.sr)
        negative_samples = self.generate_negative_samples()

        return x, positive_sample, negative_samples

    def __len__(self):
        return len(self.data_list)

    def get_train_dataloader(
        cfg: DictConfig, audio_processor: AudioProcessor
    ) -> DataLoader:
        return DataLoader(
            dataset=AGTContrastiveDataset(cfg, audio_processor, train=True),
            batch_size=cfg.train_batch_size,
            shuffle=cfg.train_shuffle,
            persistent_workers=True if cfg.num_workers > 0 else False,
            pin_memory=False,
            num_workers=cfg.num_workers,
        )

    def get_valid_dataloader(
        cfg: DictConfig, audio_processor: AudioProcessor
    ) -> DataLoader:
        return DataLoader(
            dataset=AGTContrastiveDataset(
                cfg, audio_processor, train=False, valid=True
            ),
            batch_size=cfg.valid_batch_size,
            shuffle=False,
            num_workers=0,
        )

    def get_test_dataloader(
        cfg: DictConfig, audio_processor: AudioProcessor
    ) -> DataLoader:
        return DataLoader(
            dataset=AGTContrastiveDataset(cfg, audio_processor, train=False, test=True),
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=0,
        )

    def get_dummy_dataloader(
        cfg: DictConfig, audio_processor: AudioProcessor
    ) -> DataLoader:
        return DataLoader(
            dataset=AGTContrastiveDataset(cfg, audio_processor, train=False),
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )


class AGTPretrainingDataset(AGTDataset):

    def __init__(
        self,
        cfg: DictConfig,
        audio_processor: AudioProcessor,
        train=False,
        valid=False,
        test=False,
    ):
        super().__init__(cfg, audio_processor, train, valid, test)
        self.audio_load_sec = cfg.audio_load_sec

    def __len__(self):
        return len(self.data_list)

    def cut_or_pad_if_necessary(self, x):
        if self.feature_size is not None and len(x.shape) == 2:
            if x.shape[1] > self.feature_size:
                x = x[:, : self.feature_size]
            elif x.shape[1] < self.feature_size:
                x = torch.nn.functional.pad(x, (0, self.feature_size - x.shape[1]))
        return x

    def __getitem__(self, idx):
        if "duration" in self.metadata.columns:
            duration_sec = self.metadata.iloc[idx]["duration"]
        else:
            audio_info = torchaudio.info(self.metadata.iloc[idx]["segment_path"])
            audio_duration = audio_info.num_frames / audio_info.sample_rate
        
        if duration_sec - (self.audio_load_sec if self.audio_load_sec else 0) > 0:
            random_start_sec = random.uniform(0, duration_sec - (self.audio_load_sec if self.audio_load_sec else 0))
        else:
            random_start_sec = 0
            
        wav = self.audio_processor.load_wav(
            self.metadata.iloc[idx]["segment_path"],
            start_sec=random_start_sec,
            duration_sec=self.audio_load_sec if self.audio_load_sec else duration_sec-random_start_sec,
        )

        if self.augment_audio:
            wav = self._augment(wav, self.audio_processor.config.sr)

        wav = self.cut_or_pad_if_necessary(wav)   
        x = self.audio_processor.get_feature_from_audio(wav)
        length = x.shape[1]
        
        return {"x": x, "length": length}

    def collate_batch(batch):
        batch_lengths = torch.tensor([data["length"] for data in batch])
        max_len = max(batch_lengths)
        features_collated_batch = []
        for data in batch:
            features_collated = torch.nn.functional.pad(
                data["x"], (0, max_len - data["x"].shape[1])
            )
            features_collated_batch.append(features_collated)
        features_collated_batch = torch.stack(features_collated_batch)
        return {"x": features_collated_batch, "lengths": batch_lengths}   

    def get_train_dataloader(
        cfg: DictConfig, audio_processor: AudioProcessor
    ) -> DataLoader:
        return DataLoader(
            dataset=AGTPretrainingDataset(cfg, audio_processor, train=True),
            batch_size=cfg.train_batch_size,
            shuffle=cfg.train_shuffle,
            persistent_workers=True if cfg.num_workers > 0 else False,
            pin_memory=False,
            collate_fn=AGTPretrainingDataset.collate_batch,
            num_workers=cfg.num_workers,
        )

    def get_valid_dataloader(
        cfg: DictConfig, audio_processor: AudioProcessor
    ) -> DataLoader:
        return DataLoader(
            dataset=AGTPretrainingDataset(
                cfg, audio_processor, train=False, valid=True
            ),
            batch_size=cfg.valid_batch_size,
            shuffle=False,
            collate_fn=AGTPretrainingDataset.collate_batch,
            num_workers=0,
        )

    def get_test_dataloader(
        cfg: DictConfig, audio_processor: AudioProcessor
    ) -> DataLoader:
        return DataLoader(
            dataset=AGTPretrainingDataset(cfg, audio_processor, train=False, test=True),
            batch_size=cfg.test_batch_size,
            shuffle=False,
            collate_fn=AGTPretrainingDataset.collate_batch,
            num_workers=0,
        )

    def get_dummy_dataloader(
        cfg: DictConfig, audio_processor: AudioProcessor
    ) -> DataLoader:
        return DataLoader(
            dataset=AGTContrastiveDataset(cfg, audio_processor, train=False),
            batch_size=2,
            collate_fn=AGTPretrainingDataset.collate_batch,
            shuffle=False,
            num_workers=0,
        )