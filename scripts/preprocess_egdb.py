import csv
import os
import random

import hydra
import librosa
import nnAudio
import numpy as np
import torch
import torchaudio
import mido
from concurrent.futures import ProcessPoolExecutor, as_completed
from omegaconf import DictConfig, OmegaConf
from rich.progress import track
from amt_tools import tools

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.preprocess import *

import logging
logger = logging.getLogger(__name__)

STR_MIDI_DICT = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}

EGDB_PATH = os.path.join(DATA_PATH, "raw", "EGDB")
AUDIO_PATHS = [
    os.path.join(EGDB_PATH, a)
    for a in [
        "audio_DI",
        "audio_Ftwin",
        "audio_JCjazz",
        "audio_Marshall",
        "audio_Mesa",
        "audio_Plexi",
    ]
]
ANNOTATION_PATH = os.path.join(EGDB_PATH, "audio_label")

REAL_DATA_PATH = os.path.join(EGDB_PATH, "RealData")
REAL_DATA_AUDIO_PATH = os.path.join(REAL_DATA_PATH, "Audio")
REAL_DATA_ANNOTATION_PATH = os.path.join(REAL_DATA_PATH, "Label")


def load_stacked_notes_midi(midi_path):
    """
    Extract MIDI notes spread across strings into a dictionary
    from a MIDI file following the EGDB format.

    Parameters
    ----------
    midi_path : string
      Path to MIDI file to read

    Returns
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    """

    # Standard tuning is assumed for all tracks in EGDB
    open_tuning = list(librosa.note_to_midi(tools.DEFAULT_GUITAR_TUNING))

    # Initialize a dictionary to hold the notes for each string
    stacked_notes = [tools.notes_to_stacked_notes([], [], p) for p in open_tuning]
    stacked_notes = {k: v for d in stacked_notes for k, v in d.items()}

    # Open the MIDI file
    midi = mido.MidiFile(midi_path)

    # Initialize a counter for the time
    time = 0

    # Initialize an empty list to store MIDI events
    events = []

    # Parse all MIDI messages
    for message in midi:
        # Increment the time
        time += message.time

        # Check if message is a note event (NOTE_ON or NOTE_OFF)
        if "note" in message.type:
            # Determine corresponding string index
            string_idx = 5 - message.channel
            # MIDI offsets can be either NOTE_OFF events or NOTE_ON with zero velocity
            onset = message.velocity > 0 if message.type == "note_on" else False

            # Create a new event detailing the note
            event = dict(time=time, pitch=message.note, onset=onset, string=string_idx)
            # Add note event to MIDI event list
            events.append(event)

    # Loop through all tracked MIDI events
    for i, event in enumerate(events):
        # Ignore note offset events
        if not event["onset"]:
            continue

        # Extract note attributes
        pitch = event["pitch"]
        onset = event["time"]
        string_idx = event["string"]

        # Determine where the corresponding offset occurs by finding the next note event
        # with the same string, clipping at the final frame if no correspondence is found
        offset = next(
            n
            for n in events[i + 1 :]
            if n["string"] == event["string"] or n is events[-1]
        )["time"]

        # Obtain the current collection of pitches and intervals
        pitches, intervals = stacked_notes.pop(open_tuning[string_idx])

        # Append the (nominal) note pitch
        pitches = np.append(pitches, pitch)
        # Append the note interval
        intervals = np.append(intervals, [[onset, offset]], axis=0)

        # Re-insert the pitch-interval pairs into the stacked notes dictionary under the appropriate key
        stacked_notes.update(
            tools.notes_to_stacked_notes(pitches, intervals, open_tuning[string_idx])
        )

    # Re-order keys starting from lowest string and switch to the corresponding note label
    stacked_notes = {
        librosa.midi_to_note(i): stacked_notes[i] for i in sorted(stacked_notes.keys())
    }

    return stacked_notes


def load_frets(midi_path, start_sec, end_sec, total_frames, num_classes=23, pitch_step=0, skip_empty_notes=True, add_other_string_token=True):
    total_duration = end_sec - start_sec

    if pitch_step != 0:
        raise NotImplementedError
    
    # frets_sequence = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    frets_sequence = [['-' for _ in range(6)] for _ in range(total_frames)]
    notes_sequence = [set() for _ in range(total_frames)]
    tab = []
    frets_map = []
    onsets_map = []

    stacked_notes = load_stacked_notes_midi(midi_path)
    for s, s_name in enumerate(stacked_notes):
        fret_string = np.zeros((total_frames, num_classes))
        fret_string[:, 0] = 1
        tab_string = np.zeros((total_frames, num_classes))
        tab_string[:, 0] = 1
        onsets_string = np.zeros((total_frames))

        pitches = stacked_notes[s_name][0]
        intervals = stacked_notes[s_name][1]
        for value, interval in zip(pitches, intervals):
            onset_sec = interval[0]
            note_duration = interval[1] - interval[0]
            if value is not None:
                fret_pos = int(round(value - STR_MIDI_DICT[s]))
                if not 0 <= fret_pos < num_classes:
                    logger.warning(f"Invalid fret_pos={fret_pos} value={value} s={s} STR_MIDI_DICT[s]={STR_MIDI_DICT[s]}")
                    continue

                if start_sec <= onset_sec <= end_sec:
                    # frets_sequence[s].append(fret_pos)
                    note_name = fret_to_note(STR_NOTE_DICT[s], fret_pos)
                    relative_timestamp = onset_sec - start_sec
                    fret_pos_idx = fret_pos+1
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
                        logger.error(f"onset_sec={onset_sec} note_duration={note_duration} fret_pos={fret_pos} relative_timestamp={relative_timestamp} idx={idx} total_frames={total_frames} total_duration={total_duration}")    
        
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


def process(annotation_path, audio_dirs, cfg, skip_features=False):
    feature_extractor = FeatureExtractor(cfg)
    data = []

    audio_load_sec = cfg.audio.audio_load_sec
    for audio_dir in audio_dirs:
        timbre = os.path.basename(audio_dir)
        audio_file_name = (
            os.path.splitext(os.path.basename(annotation_path))[0] + ".wav"
        )
        audio_path = os.path.join(audio_dir, audio_file_name)

        assert os.path.isfile(audio_path), f"File not found: {audio_path}"
        info = torchaudio.info(audio_path)
        duration = info.num_frames / info.sample_rate

        for start_sec in np.arange(
            start=0,
            stop=duration - cfg.audio.audio_load_sec + cfg.audio.slide_window_sec,
            step=cfg.audio.slide_window_sec,
        ):
            for pitch_step in (cfg.data.audio_augmentation.pitch_shift_steps or [0]):
                end_sec = start_sec + cfg.audio.audio_load_sec
                audio_load_sec = cfg.audio.audio_load_sec

                try:
                    all_empty, tab, fret_sequence, notes_sequence, frets_map, onsets_map, notes_map = load_frets(annotation_path, start_sec, end_sec, total_frames=cfg.data.target_len, num_classes=cfg.data.target_num_classes, pitch_step=pitch_step)
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
                        audio_file_name + f"-{timbre}_" + f"-{round(start_sec, 2)}_"
                        f"{round(start_sec+audio_load_sec, 2)}.pt",
                    )

                if not skip_features and cfg.data.segments_dir:
                    segment_path = os.path.join(
                        cfg.data.segments_dir,
                        audio_file_name + f"-{timbre}_" + f"-{round(start_sec, 2)}_"
                        f"{round(start_sec+audio_load_sec, 2)}.wav",
                    )
                else:
                    segment_path = None

                if cfg.data.targets_dir:
                    tab_path = os.path.join(
                        cfg.data.targets_dir,
                        "tab-" + audio_file_name + f"-{timbre}_" + f"-{round(start_sec, 2)}_"
                        f"{round(start_sec+audio_load_sec, 2)}"
                        f"-pitch_{pitch_step}.pt",
                    )
                    torch.save(tab, tab_path)
                    frets_path = os.path.join(
                        cfg.data.targets_dir, 
                        "frets-" + audio_file_name + f"-{timbre}_" + f"-{round(start_sec, 2)}_"
                        f"{round(start_sec+audio_load_sec, 2)}"
                        f"-pitch_{pitch_step}.pt",
                    )
                    torch.save(frets_map, frets_path)
                    onsets_path = os.path.join(
                        cfg.data.targets_dir,
                        "onsets-" + audio_file_name + f"-{timbre}_" + f"-{round(start_sec, 2)}_"
                        f"{round(start_sec+audio_load_sec, 2)}"
                        f"-pitch_{pitch_step}.pt",
                    )
                    torch.save(onsets_map, onsets_path)
                    notes_path = os.path.join(
                        cfg.data.targets_dir,
                        "notes-" + audio_file_name + f"-{timbre}_" + f"-{round(start_sec, 2)}_"
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
                        continue
                    
                if feature_shape is None:
                    if feature_path is not None and os.path.isfile(feature_path):
                        feature = torch.load(feature_path)
                        feature_shape = feature.shape
                        del feature

                data.append(
                    [
                        annotation_path,
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
    np.random.seed(cfg.random_seed)

    if cfg.data.features_dir is not None:
        os.makedirs(os.path.join(cfg.data.features_dir), exist_ok=True)
    if cfg.data.segments_dir is not None:
        os.makedirs(os.path.join(cfg.data.segments_dir), exist_ok=True)

    real_annotation_paths = [
        os.path.join(REAL_DATA_ANNOTATION_PATH, p)
        for p in list(
            filter(lambda p: p.endswith(".mid"), os.listdir(REAL_DATA_ANNOTATION_PATH))
        )
    ]
    annotation_paths = [
        os.path.join(ANNOTATION_PATH, p)
        for p in list(
            filter(lambda p: p.endswith(".midi"), os.listdir(ANNOTATION_PATH))
        )
    ]
    logger.info(f"Found {len(annotation_paths)+len(real_annotation_paths)} files")

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

    data = []
    data_real = []
    data_rendered = []

    if cfg.num_workers <= 1:
        for annotation_path in track(annotation_paths):
            r = process(annotation_path, AUDIO_PATHS, cfg)
            data = [*data, *r]
            data_rendered = [*data_rendered, *r]
        for annotation_path in track(real_annotation_paths):
            data = process(annotation_path, [REAL_DATA_AUDIO_PATH], cfg)
            data = [*data, *r]
            data_real = [*data_real, *r]
    else:
        executor = ProcessPoolExecutor(cfg.num_workers)
        futures = [
            executor.submit(process, annotation_path, AUDIO_PATHS, cfg)
            for annotation_path in annotation_paths
        ]
        for future in track(
            as_completed(futures), total=len(annotation_paths)
        ):
            r = future.result()
            data = [*data, *r]
            data_rendered = [*data_rendered, *r]

        futures = [
            executor.submit(process, annotation_path, [REAL_DATA_AUDIO_PATH], cfg)
            for annotation_path in real_annotation_paths
        ]
        for future in track(
            as_completed(futures), total=len(real_annotation_paths)
        ):
            r = future.result()
            data = [*data, *r]
            data_real = [*data_real, *r]

    random.shuffle(data)

    with open(os.path.join(DATA_PATH, f"egdb.csv"), "w") as data_csv:
        csv_writer = csv.writer(data_csv, delimiter=';')
        csv_writer.writerow(header)
        csv_writer.writerows(data)
    with open(os.path.join(DATA_PATH, f"egdb_rendered.csv"), "w") as data_csv:
        csv_writer = csv.writer(data_csv, delimiter=';')
        csv_writer.writerow(header)
        csv_writer.writerows(data_rendered)
    with open(os.path.join(DATA_PATH, f"egdb_real.csv"), "w") as data_csv:
        csv_writer = csv.writer(data_csv, delimiter=';')
        csv_writer.writerow(header)
        csv_writer.writerows(data_real)


if __name__ == "__main__":
    main()
