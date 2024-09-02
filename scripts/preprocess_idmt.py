import csv
import os
import random

import hydra
import numpy as np
import torch
import torchaudio
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from omegaconf import DictConfig, OmegaConf
from rich.progress import track

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.preprocess import *

import logging
logger = logging.getLogger(__name__)


IDMT_PATH = os.path.join(DATA_PATH, "raw", "IDMT")
AUDIO_PATH = os.path.join(IDMT_PATH, "audio")
ANNOTATION_PATH = os.path.join(IDMT_PATH, "annotation")


def load_targets(xml, start_sec, end_sec, total_frames, num_classes=23, pitch_step=0, skip_empty_notes=True, add_other_string_token=True):
    total_duration = end_sec - start_sec

    if pitch_step != 0:
        raise NotImplementedError
    
    # frets_sequence = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    frets_sequence = [['-' for _ in range(6)] for _ in range(total_frames)]
    notes_sequence = [set() for _ in range(total_frames)]
    tab = []
    frets_map = []
    onsets_map = []
    
    for _ in range(6):
        fret_string = np.zeros((total_frames, num_classes))
        fret_string[:, 0] = 1
        frets_map.append(fret_string)
        tab_string = np.zeros((total_frames, num_classes))
        tab_string[:, 0] = 1
        tab.append(tab_string)
        onsets_string = np.zeros((total_frames))
        onsets_map.append(onsets_string)

    events = xml.findall(".//event")
    for event in events:
        onset_sec = float(event.find("onsetSec").text)
        pitch = float(event.find("pitch").text)
        offset_sec = float(event.find("offsetSec").text)
        excitation_style = event.find("excitationStyle").text
        expression_style = event.find("expressionStyle").text
        note_duration = offset_sec - onset_sec
        fret_pos = int(event.find("fretNumber").text)+pitch_step
        string_number = int(event.find("stringNumber").text)-1
        note_name = fret_to_note(STR_NOTE_DICT[string_number], fret_pos)
        fret_pos_idx = fret_pos+1

        if string_number > 5:
            logger.warning(f"Unsupport string number: {string_number}")
            return

        if start_sec <= onset_sec <= end_sec:
            # frets_sequence[string_number].append(fret_pos)

            relative_timestamp = onset_sec - start_sec
            idx = int((relative_timestamp*total_frames)/total_duration)
            frets_sequence[idx][string_number] = fret_pos
            if add_other_string_token:
                for z in range(6):
                    if z != string_number and frets_sequence[idx][z] == '-':
                        frets_sequence[idx][z] = '-'
            idx_duration = int(((relative_timestamp+note_duration)*total_frames)/total_duration)
            if idx < total_frames:
                notes_sequence[idx].add(note_name)
                frets_map[string_number][idx:idx_duration, 0] = 0
                frets_map[string_number][idx:idx_duration, fret_pos_idx] = 1
                tab[string_number][idx, 0] = 0
                tab[string_number][idx, fret_pos_idx] = 1
                onsets_map[string_number][idx] = 1
            else:
                logger.error(f"onset_sec={onset_sec} note_duration={note_duration} fret_number={fret_pos} relative_timestamp={relative_timestamp} idx={idx} total_frames={total_frames} total_duration={total_duration}")   
    
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
        
    # remove empty sets from notes_sequence
    notes_sequence = [sorted(list(notes)) for notes in notes_sequence if len(notes) > 0]
    
    tab = torch.from_numpy(np.array(tab, dtype=np.float32))
    frets_map = torch.from_numpy(np.array(frets_map, dtype=np.float32))
    onsets_map = torch.from_numpy(np.array(onsets_map, dtype=np.float32))
    notes_map = frets_to_notes(frets_map)
    
    if notes_map[:, 1:].count_nonzero() == 0:
        if skip_empty_notes:
            raise Exception(f"No notes found in the segment {start_sec}-{end_sec}")
        logger.warning(f"No notes found in the segment {start_sec}-{end_sec}")
        
    return all_empty, tab, frets_sequence, notes_sequence, frets_map, onsets_map, notes_map


def process(anno_name, cfg, skip_features=False):
    data = []
    feature_extractor = FeatureExtractor(cfg)
    
    audio_file_name = anno_name + ".wav"
    file_path = os.path.join(ANNOTATION_PATH, anno_name + ".xml")
    audio_path = os.path.join(AUDIO_PATH, audio_file_name)
    info = torchaudio.info(audio_path)
    num_frames = info.num_frames
    sample_rate = info.sample_rate
    duration = int(num_frames / sample_rate)
    xml = ET.parse(file_path).getroot()
    
    for start_sec in np.arange(
        start=0,
        stop=duration - cfg.audio.audio_load_sec + cfg.audio.slide_window_sec,
        step=cfg.audio.slide_window_sec,
    ):
        for pitch_step in (cfg.data.audio_augmentation.pitch_shift_steps or [0]):
            end_sec = start_sec + cfg.audio.audio_load_sec
            audio_load_sec = cfg.audio.audio_load_sec
            try:
                all_empty, tab, fret_sequence, notes_sequence, frets_map, onsets_map, notes_map = load_targets(xml, start_sec=start_sec, end_sec=end_sec, total_frames=cfg.data.target_len, num_classes=cfg.data.target_num_classes, pitch_step=pitch_step)
                if all_empty:
                    logger.warning(f"Empty segment {start_sec}-{end_sec}")
                    continue
            except Exception as e:
                logger.warning("load_frets failed: " + str(e))
                continue

            feature_path = None
            if cfg.data.features_dir:
                feature_path = os.path.join(
                    cfg.data.features_dir,
                    anno_name + f"-{round(start_sec, 2)}_"
                    f"{round(start_sec+audio_load_sec, 2)}"
                    f"-pitch_{pitch_step}.pt",
                )

            if cfg.data.segments_dir:
                segment_path = os.path.join(
                    cfg.data.segments_dir,
                    anno_name + f"-{round(start_sec, 2)}_"
                    f"{round(start_sec+audio_load_sec, 2)}"
                    f"-pitch_{pitch_step}.wav",
                )
            else:
                segment_path = None

            if cfg.data.targets_dir:
                tab_path = os.path.join(
                    cfg.data.targets_dir,
                    "tab-" + anno_name + f"-{round(start_sec, 2)}_"
                    f"{round(start_sec+audio_load_sec, 2)}"
                    f"-pitch_{pitch_step}.pt",
                )
                torch.save(tab, tab_path)
                frets_path = os.path.join(
                    cfg.data.targets_dir, 
                    "frets-" + anno_name + f"-{round(start_sec, 2)}_"
                    f"{round(start_sec+audio_load_sec, 2)}"
                    f"-pitch_{pitch_step}.pt",
                )
                torch.save(frets_map, frets_path)
                onsets_path = os.path.join(
                    cfg.data.targets_dir,
                    "onsets-" + anno_name + f"-{round(start_sec, 2)}_"
                    f"{round(start_sec+audio_load_sec, 2)}"
                    f"-pitch_{pitch_step}.pt",
                )
                torch.save(onsets_map, onsets_path)
                notes_path = os.path.join(
                    cfg.data.targets_dir,
                    "notes-" + anno_name + f"-{round(start_sec, 2)}_"
                    f"{round(start_sec+audio_load_sec, 2)}"
                    f"-pitch_{pitch_step}.pt",
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


@hydra.main(version_base=None, config_path="../configs", config_name="audio_config")
def main(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))

    random.seed(cfg.random_seed)

    if cfg.data.features_dir is not None:
        os.makedirs(os.path.join(cfg.data.features_dir), exist_ok=True)
    if cfg.data.segments_dir is not None:
        os.makedirs(os.path.join(cfg.data.segments_dir), exist_ok=True)

    file_names = list(filter(lambda p: p.endswith("xml"), os.listdir(ANNOTATION_PATH)))
    logger.info(f"Found {len(file_names)} files")
    file_names = [os.path.splitext(p)[0] for p in file_names]

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

    if cfg.num_workers <= 1:
        for file_name in track(file_names):
            d = process(file_name, cfg)
            dataset = [*dataset, *d]
    else:
        executor = ProcessPoolExecutor(cfg.num_workers)
        futures = [executor.submit(process, file_name, cfg) for file_name in file_names]

        for future in track(as_completed(futures), total=len(file_names)):
            d = future.result()
            dataset = [*dataset, *d]

    random.shuffle(dataset)

    with open(os.path.join(DATA_PATH, f"idmt.csv"), "w") as data_csv:
        csv_writer = csv.writer(data_csv, delimiter=';')
        csv_writer.writerow(header)
        csv_writer.writerows(dataset)


if __name__ == "__main__":
    main()
