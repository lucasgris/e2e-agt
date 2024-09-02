import csv
import os
import random
import hydra
import jams
import numpy as np
import torch
import torchaudio
from concurrent.futures import ProcessPoolExecutor, as_completed
from omegaconf import DictConfig, OmegaConf
from rich.progress import track

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.preprocess import *

import logging
logger = logging.getLogger(__name__)

GUITARSET_PATH = os.path.join(DATA_PATH, "raw", "GuitarSet")
AUDIO_PATH = os.path.join(GUITARSET_PATH, "audio", "audio_mic")
ANNOTATION_PATH = os.path.join(GUITARSET_PATH, "annotation")
PLAYERS_IDS = ["00", "01", "02", "03", "04", "05"]
TRAIN_PLAYERS_IDS = ["02", "03", "04", "05"]
VALID_PLAYERS_IDS = ["01"]
TEST_PLAYERS_IDS = ["00"]


def load_targets(jam, total_duration, frame_duration=0.05, num_classes=26, pitch_step=0):
    total_frames = int((total_duration+1)/frame_duration)

    if pitch_step != 0:
        raise NotImplementedError
    
    # frets_sequence = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    frets_sequence = [['-' for _ in range(6)] for _ in range(total_frames)]
    notes_sequence = [set() for _ in range(total_frames)]
    tab = []
    frets_map = []
    onsets_map = []

    note_midi_strings = jam.search(namespace="note_midi")
    if len(note_midi_strings) == 0:
        note_midi_strings = jam.search(namespace="pitch_midi")
    for s, note_midi_string in enumerate(note_midi_strings):
        fret_string = np.zeros((total_frames, num_classes))
        fret_string[:, 0] = 1
        tab_string = np.zeros((total_frames, num_classes))
        tab_string[:, 0] = 1
        onsets_string = np.zeros((total_frames))

        pitch_note_window = None
        for anno in note_midi_string:
            timestamp = anno[0]
            note_duration = anno[1]
            pitch_note_window = float(anno[2])
            if pitch_note_window is not None:
                fret_pos = (
                    int(round(pitch_note_window - STR_MIDI_DICT[s])) + pitch_step
                )
                note_name = fret_to_note(STR_NOTE_DICT[s], fret_pos)
                fret_pos_idx = fret_pos+1
                idx = int(timestamp/frame_duration)
                frets_sequence[idx][s] = fret_pos
                for z in range(6):
                    if z != s and frets_sequence[idx][z] == '-':
                        frets_sequence[idx][z] = '-'
                idx_duration = int((timestamp+note_duration)/frame_duration)
                if idx < total_frames:
                    notes_sequence[idx].add(note_name)
                    fret_string[idx:idx_duration, 0] = 0
                    fret_string[idx:idx_duration, fret_pos_idx] = 1
                    tab_string[idx, 0] = 0
                    tab_string[idx, fret_pos_idx] = 1
                    onsets_string[idx] = 1
                else:
                    logger.error(f"timestamp={timestamp} note_duration={note_duration} pitch_note_window={pitch_note_window} fret_pos={fret_pos_idx} idx={idx} idx_duration={idx_duration} total_frames={total_frames} total_duration={total_duration}")                 
                if fret_pos_idx < 0:
                    raise Exception(f"fret_pos < 0: {fret_pos_idx}")
        
        tab.append(tab_string)
        frets_map.append(fret_string)
        onsets_map.append(onsets_string)
    
    final_frets_sequence = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    for t in range(total_frames):
        all_empty = True
        for s in range(6):
            if frets_sequence[t][s] != '-':
                all_empty = False
                break
        if not all_empty:
            for s in range(6):
                final_frets_sequence[s].append(frets_sequence[t][s])
    
    len_frets = [len(final_frets_sequence[s]) for s in final_frets_sequence]
    assert len(set(len_frets)) == 1, f"len_frets={len_frets}, {set(len_frets)}"
    
    frets_sequence = final_frets_sequence
    all_empty = True
    for s in frets_sequence:
        if len(frets_sequence[s]) != 0:
            all_empty = False
            break
    
    # remove empty sets from notes_sequence
    notes_sequence = [sorted(list(notes)) for notes in notes_sequence if len(notes) > 0]
    
    tab = torch.from_numpy(np.array(tab, dtype=np.float32))
    frets_map = torch.from_numpy(np.array(frets_map, dtype=np.float32))
    onsets_map = torch.from_numpy(np.array(onsets_map, dtype=np.float32))
    notes_map = frets_to_notes(frets_map)
        
    return all_empty, tab, frets_sequence, notes_sequence, frets_map, onsets_map, notes_map


def process(anno_name, cfg, skip_features=False):
    data = []

    player_id, other, comp_or_solo = anno_name.split("_")
    style, tempo, tone = other.split("-")
    annotation_path = os.path.join(ANNOTATION_PATH, anno_name + ".jams")
    assert os.path.isfile(annotation_path)
    jam = jams.load(annotation_path)

    audio_path = os.path.join(AUDIO_PATH, anno_name + "_mic.wav")
    assert os.path.isfile(audio_path), audio_path
    info = torchaudio.info(audio_path)
    num_frames = info.num_frames
    sample_rate = info.sample_rate
    duration = int(num_frames / sample_rate)

    for pitch_step in (cfg.data.audio_augmentation.pitch_shift_steps or [0]):
        audio_load_sec = cfg.audio.audio_load_sec
        try:
            all_empty, tab, fret_sequence, notes_sequence, frets_map, onsets_map, notes_map = load_targets(jam, total_duration=duration, num_classes=cfg.data.target_num_classes, pitch_step=pitch_step)
            if all_empty:
                logger.warning(f"Empty segment {anno_name}")
                continue
        except Exception as e:
            logger.warning("load_frets failed: " + str(e))
            continue

        feature_path = None
        if cfg.data.features_dir:
            feature_path = os.path.join(
                cfg.data.features_dir,
                anno_name + f"-pitch_{pitch_step}.pt",
            )

        if cfg.data.segments_dir:
            segment_path = os.path.join(
                cfg.data.segments_dir,
                anno_name + f"-pitch_{pitch_step}.wav",
            )
        else:
            segment_path = None

        if cfg.data.targets_dir:
            tab_path = os.path.join(
                cfg.data.targets_dir,
                "tab-" + anno_name + f"-pitch_{pitch_step}.pt",
            )
            torch.save(tab, tab_path)
            frets_path = os.path.join(
                cfg.data.targets_dir, 
                "frets-" + anno_name + f"-pitch_{pitch_step}.pt",
            )
            torch.save(frets_map, frets_path)
            onsets_path = os.path.join(
                cfg.data.targets_dir,
                "onsets-" + anno_name + f"-pitch_{pitch_step}.pt",
            )
            torch.save(onsets_map, onsets_path)
            notes_path = os.path.join(
                cfg.data.targets_dir,
                "notes-" + anno_name + f"-pitch_{pitch_step}.pt",
            )
            torch.save(notes_map, notes_path)
        else:
            tab_path = None
            frets_path = None
            onsets_path = None
            notes_path = None

        feature_shape = None
        if not skip_features:
            try:
                feature_extractor = FeatureExtractor(cfg)
                feature_shape = extract_features(
                    cfg=cfg,
                    feature_extractor=feature_extractor,
                    audio_path=audio_path,
                    feature_path=feature_path,
                    segment_path=segment_path,
                    pitch_step=pitch_step,
                    force_reprocess=cfg.preprocessing.force_reprocess,
                )
            except Exception as e:
                logger.error(e)
                if cfg.raise_error:
                    raise e
            
        if feature_shape is None:
            if feature_path is not None and os.path.isfile(feature_path):
                feature = torch.load(feature_path)
                feature_shape = feature.shape
                del feature
                
        data.append(
            [
                anno_name,
                round(duration, 2),
                pitch_step,
                fret_sequence[0],
                fret_sequence[1],
                fret_sequence[2],
                fret_sequence[3],
                fret_sequence[4],
                fret_sequence[5],
                notes_sequence,
                str(list(feature_shape)).replace(',', ' ') if feature_shape is not None else None, 
                os.path.basename(feature_path) if feature_path is not None else None,
                os.path.basename(frets_path) if frets_path is not None else None,
                os.path.basename(notes_path) if notes_path is not None else None,
                os.path.basename(onsets_path) if onsets_path is not None else None,
                os.path.basename(segment_path) if segment_path is not None else None,
                os.path.basename(tab_path) if tab_path is not None else None,
                feature_path,
                frets_path,
                notes_path,
                onsets_path,
                segment_path,
                tab_path
            ]
        )

    return player_id, data


@hydra.main(version_base=None, config_path="../configs", config_name="audio_config")
def main(cfg: DictConfig) -> None:
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    logger.info(OmegaConf.to_yaml(cfg))

    if cfg.data.features_dir is not None:
        os.makedirs(os.path.join(cfg.data.features_dir), exist_ok=True)
    if cfg.data.segments_dir is not None:
        os.makedirs(os.path.join(cfg.data.segments_dir), exist_ok=True)

    file_names = list(filter(lambda p: p.endswith("jams"), os.listdir(ANNOTATION_PATH)))
    file_names = [os.path.splitext(p)[0] for p in file_names]

    header = [
        "file_name",
        "duration",
        "pitch_step",
        "fret_seq_0",
        "fret_seq_1",
        "fret_seq_2",
        "fret_seq_3",
        "fret_seq_4",
        "fret_seq_5",
        "notes_sequence",
        "feature_shape",
        "feature_filename",
        "frets_filename",
        "notes_filename",
        "onsets_filename",
        "segment_filename",
        "tab_filename",
        "feature_path",
        "frets_path",
        "notes_path",
        "onsets_path",
        "segment_path",
        "tab_path"
    ]

    train_data = []
    valid_data = []
    test_data = []

    if cfg.num_workers <= 1:
        for file_name in track(file_names):
            try:
                player_id, data = process(file_name, cfg)

                if player_id in TRAIN_PLAYERS_IDS:
                    train_data = [*train_data, *data]
                elif player_id in VALID_PLAYERS_IDS:
                    valid_data = [*valid_data, *data]
                elif player_id in TEST_PLAYERS_IDS:
                    test_data = [*test_data, *data]
            except Exception as e:
                logger.error(e)
                if cfg.raise_error:
                    raise e
    else:
        executor = ProcessPoolExecutor(cfg.num_workers)
        futures = [executor.submit(process, file_name, cfg) for file_name in file_names]

        for future in track(as_completed(futures), total=len(file_names)):
            try:
                player_id, data = future.result()

                if player_id in TRAIN_PLAYERS_IDS:
                    train_data = [*train_data, *data]
                elif player_id in VALID_PLAYERS_IDS:
                    valid_data = [*valid_data, *data]
                elif player_id in TEST_PLAYERS_IDS:
                    test_data = [*test_data, *data]
            except Exception as e:
                logger.error(e)
                if cfg.raise_error:
                    raise e
                
    random.shuffle(train_data)

    with open(os.path.join(DATA_PATH, f"guitarset.csv"), 'w') as train_csv:
        csv_writer = csv.writer(train_csv, delimiter=';')
        csv_writer.writerow(header)
        csv_writer.writerows(train_data)
    with open(os.path.join(DATA_PATH, f"valid.csv"), 'w') as valid_csv:
        csv_writer = csv.writer(valid_csv, delimiter=';')
        csv_writer.writerow(header)
        csv_writer.writerows(valid_data)        
    with open(os.path.join(DATA_PATH, f"test.csv"), 'w') as test_csv:
        csv_writer = csv.writer(test_csv, delimiter=';')
        csv_writer.writerow(header)
        csv_writer.writerows(test_data)

    r = input("Press any key to generate the folds metadata or c to cancel...")
    if r.lower() == 'c':
        return

    data_folds = {fold: {
        "train": [],
        "valid": [],
        "test": []
    } for fold in range(len(PLAYERS_IDS))}
        
    for file_name in track(file_names):
        player_id, data = process(file_name, cfg)
        
        for fold in range(len(PLAYERS_IDS)):
            train_ids = PLAYERS_IDS.copy()
            train_ids.pop(fold)
            test_id = PLAYERS_IDS[fold]

            if player_id in train_ids:
                if random.random() <= 0.1:
                    data_folds[fold]["valid"] = [*data_folds[fold]["valid"], *data]
                else:
                    data_folds[fold]["train"] = [*data_folds[fold]["train"], *data]
            elif player_id in test_id:
                data_folds[fold]["test"] = [*data_folds[fold]["test"], *data]

    for fold in range(len(PLAYERS_IDS)):
        logger.info(f"Processing fold {fold}")
        fold_path = os.path.join(DATA_PATH, 'folds', str(fold))
        os.makedirs(fold_path, exist_ok=True)
        random.shuffle(data_folds[fold]["train"])             

        with open(os.path.join(fold_path, "train.csv"), 'w') as train_csv:
            csv_writer = csv.writer(train_csv, delimiter=';')
            csv_writer.writerow(header)
            for row in data_folds[fold]["train"]:
                csv_writer.writerow(row)
        with open(os.path.join(fold_path, "valid.csv"), 'w') as valid_csv:
            csv_writer = csv.writer(valid_csv, delimiter=';')
            csv_writer.writerow(header)
            for row in data_folds[fold]["valid"]:
                csv_writer.writerow(row)
        with open(os.path.join(fold_path, "test.csv"), 'w') as test_csv:
            csv_writer

if __name__ == "__main__":
    main()