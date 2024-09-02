import csv
import os
import random

import hydra
import jams
import librosa
import nnAudio
import numpy as np
import torch
import torchaudio
from concurrent.futures import ProcessPoolExecutor, as_completed
from omegaconf import DictConfig, OmegaConf
from rich.progress import track
from guitarpro import Duration

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.preprocess import *

import logging
logger = logging.getLogger(__name__)


def ticks_to_seconds(ticks, tempo):
    time = (60 / tempo) * ticks / Duration.quarterTime
    return time


def load_frets(jam, tempo, start_sec, end_sec, total_frames, num_classes=23, pitch_step=0, skip_empty_notes=True, add_other_string_token=True):
    total_duration = end_sec - start_sec
    
    assert end_sec > start_sec, "end_sec must be greater than start_sec"
    if pitch_step != 0:
        raise NotImplementedError
    
    # frets_sequence = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    frets_sequence = [['-' for _ in range(6)] for _ in range(total_frames)]
    notes_sequence = [set() for _ in range(total_frames)]
    tab = []
    frets_map = []
    onsets_map = []

    note_tab_strings = jam.search(namespace='note_tab')
    for s, note_tab_string in enumerate(note_tab_strings):
        if s > 5:
            raise NotImplementedError("Only 6 strings are supported")
        fret_string = np.zeros((total_frames, num_classes))
        fret_string[:, 0] = 1
        tab_string = np.zeros((total_frames, num_classes))
        tab_string[:, 0] = 1
        onsets_string = np.zeros((total_frames))

        for anno in note_tab_string:
            timestamp = ticks_to_seconds(anno[0], tempo)
            if timestamp < start_sec:
                continue
            note_duration = ticks_to_seconds(anno[1], tempo)
            if start_sec < timestamp < end_sec:
                if anno[2] is not None:
                    fret_pos = int(anno[2]["fret"]) + pitch_step
                    if fret_pos < 0 or fret_pos >= num_classes:
                        logger.warning(f"fret_pos={fret_pos} num_classes={num_classes}")
                        raise Exception("Invalid fret position found")
                    # frets_sequence[s].append(int(fret_pos))
                    note_name = fret_to_note(STR_NOTE_DICT[s], fret_pos)
                    fret_pos_idx = fret_pos+1
                    relative_timestamp = timestamp - start_sec
                    idx = int((relative_timestamp*total_frames)/total_duration)
                    frets_sequence[idx][s] = fret_pos
                    if add_other_string_token:
                        for z in range(6):
                            if z != s and frets_sequence[idx][z] == '-':
                                frets_sequence[idx][z] = '-'
                    idx_duration = int(((relative_timestamp+note_duration)*total_frames)/total_duration)
                    if idx < total_frames:
                        notes_sequence[idx].add(note_name)
                        fret_string[idx:idx_duration, 0] = 0
                        fret_string[idx:idx_duration, fret_pos_idx] = 1
                        tab_string[idx, 0] = 0
                        tab_string[idx, fret_pos_idx] = 1
                        onsets_string[idx] = 1
                    else:
                        logger.error(f"timestamp={timestamp} note_duration={note_duration} fret_pos={fret_pos} relative_timestamp={relative_timestamp} idx={idx} total_frames={total_frames} total_duration={total_duration}")

            if timestamp > end_sec:
                break  # Go to the next string, we already got the data 
        
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
    
    if add_other_string_token:
        len_frets = [len(final_frets_sequence[s]) for s in final_frets_sequence]
        assert len(set(len_frets)) == 1, f"len_frets={len_frets}, {set(len_frets)}"
    
    frets_sequence = final_frets_sequence
    
    all_empty = True
    for s in frets_sequence:
        if len(frets_sequence[s]) != 0:
            all_empty = False
            break
    
    tab = torch.from_numpy(np.array(tab, dtype=np.float32))
    frets_map = torch.from_numpy(np.array(frets_map, dtype=np.float32))
    onsets_map = torch.from_numpy(np.array(onsets_map, dtype=np.float32))
    notes_map = frets_to_notes(frets_map)
    
    # remove empty sets from notes_sequence
    notes_sequence = [sorted(list(notes)) for notes in notes_sequence if len(notes) > 0]
    
    if notes_map[:, 1:].count_nonzero() == 0:
        if skip_empty_notes:
            raise Exception(f"No notes found in the segment {start_sec}-{end_sec}")
        logger.warning(f"No notes found in the segment {start_sec}-{end_sec}, fret_sequence={frets_sequence}")
        
    return all_empty, tab, frets_sequence, notes_sequence, frets_map, onsets_map, notes_map

def create_subdirs(base_path, dest_path):
    # Generate the destination folder structure
    for char in base_path:
        dest_path = os.path.join(dest_path, char.lower())
        if len(dest_path.split(os.path.sep)) > 8:
            break
    
    os.makedirs(dest_path, exist_ok=True)
    return dest_path

def process(data_to_process, cfg, skip_features=False, force_reprocess=False): 
    feature_extractor = FeatureExtractor(cfg)
    data = []

    annotation_path = data_to_process["jam_path"]
    jam = jams.load(annotation_path, strict=False, validate=False)
    audio_paths = data_to_process["audio_paths"]

    info = torchaudio.info(audio_paths[0])
    num_frames = info.num_frames
    sample_rate = info.sample_rate
    duration = int(num_frames / sample_rate)
    for audio_path in audio_paths:
        assert os.path.isfile(audio_path), audio_path
        
        audio_id = str(os.path.basename(audio_path)).replace(' ','').replace('/', '_').replace('.', '_')[:200]

        for start_sec in np.arange(
            start=0,
            stop=duration - cfg.audio.audio_load_sec + cfg.audio.slide_window_sec,
            step=cfg.audio.slide_window_sec,
        ):
            # print(f"Processing {audio_id} {start_sec} {start_sec+cfg.audio.audio_load_sec}")
            for pitch_step in (cfg.data.audio_augmentation.pitch_shift_steps or [0]):
                end_sec = start_sec + cfg.audio.audio_load_sec
                audio_load_sec = cfg.audio.audio_load_sec
                try:
                    all_empty, tab, fret_sequence, notes_sequence, frets_map, onsets_map, notes_map = load_frets(jam, 
                                    tempo=data_to_process["tempo"], 
                                    start_sec=start_sec, 
                                    end_sec=end_sec, total_frames=cfg.data.target_len, num_classes=cfg.data.target_num_classes, pitch_step=pitch_step)
                    if all_empty:
                        logger.warning(f"Empty segment {start_sec}-{end_sec}")
                        continue
                except Exception as e:
                    logger.warning("load_frets failed: " + str(e))
                    continue

                feature_path = None
                if cfg.data.features_dir:
                    feature_path = os.path.join(
                        create_subdirs(audio_id, cfg.data.features_dir),
                        audio_id + f"-{round(start_sec, 2)}_"
                        f"{round(start_sec+audio_load_sec, 2)}"
                        f"-pitch_{pitch_step}.pt",
                    )

                if cfg.data.segments_dir:
                    segment_path = os.path.join(
                        create_subdirs(audio_id, cfg.data.segments_dir),
                        audio_id + f"-{round(start_sec, 2)}_"
                        f"{round(start_sec+audio_load_sec, 2)}"
                        f"-pitch_{pitch_step}.wav",
                    )
                else:
                    segment_path = None

                if cfg.data.targets_dir:
                    tab_path = os.path.join(
                        create_subdirs(audio_id, cfg.data.targets_dir),
                        "tab-" + audio_id + f"-{round(start_sec, 2)}_"
                        f"{round(start_sec+audio_load_sec, 2)}"
                        f"-pitch_{pitch_step}.pt",
                    )
                    torch.save(tab, tab_path)
                    frets_path = os.path.join(
                        create_subdirs(audio_id, cfg.data.targets_dir),
                        "frets-" + audio_id + f"-{round(start_sec, 2)}_"
                        f"{round(start_sec+audio_load_sec, 2)}"
                        f"-pitch_{pitch_step}.pt"
                    )
                    torch.save(frets_map, frets_path)
                    onsets_path = os.path.join(
                        create_subdirs(audio_id, cfg.data.targets_dir),
                        "onsets-" + audio_id + f"-{round(start_sec, 2)}_"
                        f"{round(start_sec+audio_load_sec, 2)}"
                        f"-pitch_{pitch_step}.pt"
                    )
                    torch.save(onsets_map, onsets_path)
                    notes_path = os.path.join(
                        create_subdirs(audio_id, cfg.data.targets_dir),
                        "notes-" + audio_id + f"-{round(start_sec, 2)}_"
                        f"{round(start_sec+audio_load_sec, 2)}"
                        f"-pitch_{pitch_step}.pt"
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
                        feature_shape = extract_features(
                            cfg=cfg,
                            feature_extractor=feature_extractor,
                            audio_path=audio_path,
                            feature_path=feature_path,
                            start_sec=start_sec,
                            audio_load_sec=audio_load_sec,
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

                if not os.path.isfile(segment_path):
                    logger.error(f"Segment not found: {segment_path}")
                    continue

                data.append(
                    [
                        audio_id,
                        round(start_sec, 2),
                        round(start_sec + audio_load_sec, 2),
                        round(audio_load_sec, 2),
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

    return data
    

@hydra.main(version_base=None, 
            config_path="../configs", 
            config_name="audio_config")
def main(cfg : DictConfig) -> None:

    if "folder_name" in cfg:
        FOLDER_NAME = cfg.folder_name
    else:
        print("Please specify folder_name in config, e.g. folder_name: SynthTab_Dev or SynthTab_Acoustic")
        exit()
    SYNTHTAB_PATH = os.path.join(DATA_PATH, "raw", FOLDER_NAME)
    ANNOTATION_PATH = os.path.join(SYNTHTAB_PATH, "jams")
    AUDIO_PATH = os.path.join(SYNTHTAB_PATH, "audio")

    logger.info(OmegaConf.to_yaml(cfg))
    
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    if cfg.data.features_dir is not None:
        os.makedirs(os.path.join(cfg.data.features_dir), exist_ok=True)
    if cfg.data.segments_dir is not None:
        os.makedirs(os.path.join(cfg.data.segments_dir), exist_ok=True)

    
    data_to_process = {}
    for root, dirs, files in os.walk(AUDIO_PATH):
        for file in files:
            if file.endswith('.flac'):
                file_name = os.path.splitext(os.path.basename(file))[0]
                jam_path = os.path.join(ANNOTATION_PATH, file_name+".jams")
                if not os.path.isfile(jam_path):
                    logger.error(f"Jams file not found: {jam_path}")
                    continue
                if file_name in data_to_process:
                    data_to_process[file_name]["audio_paths"].append(os.path.join(root, file))
                else:
                    data_to_process[file_name] = {
                        "jam_path": jam_path,
                        "audio_paths": [os.path.join(root, file)],
                        "tempo": int(open(os.path.join(ANNOTATION_PATH, file_name+".tempo.txt")).readline().strip())
                    }
    logger.info(f"Found {len(data_to_process)} files")

    header = [
        "file_name",
        "start_sec",
        "end_sec",
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

    dataset = []

    force_reprocess = ("force_reprocess" in cfg and cfg.force_reprocess)
    if cfg.num_workers <= 1:
        for file_name in track(data_to_process):
            try:
                data = process(data_to_process[file_name], cfg, force_reprocess=force_reprocess)
                dataset = [*dataset, *data]
            except Exception as e:
                logger.error(e)
                if cfg.raise_error:
                    raise e
    else:
        executor = ProcessPoolExecutor(cfg.num_workers)
        futures = [
            executor.submit(
                process, 
                data_to_process[file_name], 
                cfg, force_reprocess=force_reprocess
            ) for file_name in data_to_process
        ]
            
        for future in track(as_completed(futures), total=len(data_to_process)):
            data = future.result()
            dataset = [*dataset, *data]
                    
    random.shuffle(dataset)

    with open(os.path.join(DATA_PATH, f"{FOLDER_NAME}.csv"), 'w') as file:
        csv_writer = csv.writer(file, delimiter=';')
        csv_writer.writerow(header)
        csv_writer.writerows(dataset)

if __name__ == "__main__":
    main()