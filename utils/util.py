import os
import sys
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt

STR_MIDI_DICT = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
NOTE_OCTAVES  = ['E2', 'F2', 'F#2', 'G2', 'G#2', 'A2', 'A#2', 'B2', 'C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4', 'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5', 'C6', 'C#6', 'D6']
               
def load_config(config_path: str):
    from omegaconf import OmegaConf
    with open(config_path, "r") as f:
        config = OmegaConf.load(f)
    return config

def plot_ffm_map(frets_ohe, frets_map, path="ffm_map.png"):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    axs[0].imshow(frets_ohe.T, aspect="auto")
    axs[0].set_title("Frets OHE")
    axs[1].imshow(frets_map.T, aspect="auto")
    axs[1].set_title("Frets Map")
    plt.savefig(path)

def add_noise(x, noise_factor):
    noise = torch.randn_like(x) * noise_factor
    noisy_x = x + noise
    return noisy_x.clamp(-1, 1)

def segment_feature(feature, segment_length):
    if segment_length > feature.size(-1):
        raise ValueError("Segment length is greater than feature length")

    num_segments = feature.size(-1) // segment_length
    
    segments = list(feature.split(segment_length, dim=-1))
    if segments[-1].size(-1) < segment_length:
        segments[-1] = torch.nn.functional.pad(
            segments[-1],
            (0, segment_length - segments[-1].size(-1)),
            mode="constant",
            value=0,
        )
    segments = torch.stack(segments, dim=0)
    
    return segments
    
def get_ffm(frets_ohe, ffm_map_kernel, ffm_noise_factor=0, ffm_delete_prob=0):
        # We have to discard the silence and first string class to create the map of teh hand in the fretboard
        frets_map_raw = frets_ohe[:, :, 2:].clone().mean(0).unsqueeze(0)
        frets_map_binarized = (frets_map_raw > 0).float()
        frets_map = torch.nn.functional.conv2d(
            frets_map_binarized.unsqueeze(1),
            ffm_map_kernel,
            padding=[
                ffm_map_kernel.size(2) // 2,
                ffm_map_kernel.size(3) // 2,
            ],
            stride=ffm_map_kernel.size(2),
        ).squeeze(1)
        if ffm_noise_factor > 0:
            frets_map = add_noise(frets_map, ffm_noise_factor)
        frets_map = frets_map.squeeze()
        if ffm_delete_prob > 0:
            # mask in the time dimension
            mask = torch.rand(frets_map.shape[0]) > ffm_delete_prob
            frets_map = frets_map * mask.unsqueeze(-1)
        return frets_map
    
def notes_to_hz_mir_multipitch(notes, seconds_per_frame=0.05, real_class_shift=-1, silence=0):
    pitches = []
    intervals = []
    for t in range(notes.shape[0]):
        intervals.append(t * seconds_per_frame)
        pitches.append([])
        for i in range(notes.shape[1]):
            if notes[t, i] != silence:
                note_octave = NOTE_OCTAVES[notes[t, i] + real_class_shift]
                pitch = librosa.note_to_hz(note_octave)
                pitches[t].append(pitch)
        pitches[t] = np.array(pitches[t])
    intervals = np.array(intervals)
    return intervals, pitches 

def notes_to_hz_mir_transcription(notes, seconds_per_frame=0.05, real_class_shift=-1, silence=0):
    pitches = []
    intervals = []
    for i in range(notes.shape[1]):
        start_time = end_time = None
        for t in range(notes.shape[0]):
            if notes[t, i] != silence:
                note_octave = NOTE_OCTAVES[notes[t, i] + real_class_shift]
                pitch = librosa.note_to_hz(note_octave)
                if start_time is None:
                    start_time = t * seconds_per_frame
                    continue
            elif start_time is not None:  # end of event
                pitch = librosa.note_to_hz(note_octave)
                pitches.append(pitch)
                end_time = t * seconds_per_frame
                intervals.append([start_time, end_time])
                start_time = None
        
    pitches = np.array(pitches)
    intervals = np.array(intervals)
    return intervals, pitches          
            
def tab_to_hz_mir_eval(tab, seconds_per_frame=0.05, real_class_shift=-1, silence=0):
    pitches = [[] for _ in range(tab.shape[0])]
    intervals = [[] for _ in range(tab.shape[0])]
    for s in range(tab.shape[0]):
        current_fret = None
        for t in range(tab.shape[1]):
            if tab[s, t] != silence:
                fret = tab[s, t] + real_class_shift
                pitch = librosa.midi_to_hz(fret + STR_MIDI_DICT[s])
                if current_fret is None:
                    current_fret = fret
                    start_time = t * seconds_per_frame
                    continue
                if fret != current_fret:
                    pitch = librosa.midi_to_hz(current_fret + real_class_shift + STR_MIDI_DICT[s])
                    pitches[s].append(pitch)
                    end_time = t * seconds_per_frame
                    intervals[s].append([start_time, end_time])
                    
                    current_fret = fret
                    start_time = t * seconds_per_frame
            
        pitches[s] = np.array(pitches[s])
        intervals[s] = np.array(intervals[s])
    return intervals, pitches

def print_tab(tab, num_strings=6, real_class_shift=-1, silence=0, file=sys.stdout):
    max_sequence_length = max(
        len(tab[s]) for s in range(num_strings)
    )
    for s in range(num_strings):
        try:
            chars_to_print = min(
                max_sequence_length, os.get_terminal_size().columns - 25
            )
        except:
            chars_to_print = max_sequence_length
        print(f"{s+1}", end="|", file=file)
        for i in range(chars_to_print):
            if chars_to_print <= 0:
                break
            if int(tab[s, i]) != silence:
                t = int(tab[s, i]) + real_class_shift
                if t < 0:
                    print("?", end="-", file=file)
                    chars_to_print -= 2
                else:
                    if t < 10:
                        print(t, end="-", file=file)
                        chars_to_print -= 2
                    else:
                        print(f"{t:2d}", end="-", file=file)
                        chars_to_print -= 3
            else:
                print("-", end="", file=file)
                chars_to_print -= 1
        while chars_to_print > 0:
            print("-", end="", file=file)
            chars_to_print -= 1
        print(f"|", file=file)

def resize_target(tab, target_len=100, upsample_method="interpolate"):
    if upsample_method == "pad":
        tab = torch.nn.functional.pad(
            tab, (0, 0, 0, target_len - tab.shape[1], 0, 0)
        )
    elif upsample_method == "interpolate":
        tab = torch.nn.functional.interpolate(
            tab.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
    return tab

def min_max_normalize(tensor, max_val=None, min_val=None):
    if max_val is None:
        max_val = tensor.max()
    if min_val is None:   
        min_val = tensor.min()
    return (tensor - min_val) / (max_val - min_val)

def argmax_map(tensor, dim=-1):
    _, argmax = torch.max(tensor, dim=dim)
    tensor_argmax = torch.zeros_like(tensor)
    for i, arg in enumerate(argmax):
        tensor_argmax[i, arg] = 1
    return tensor_argmax

def colorize_arr(arr):
    unique_values = set(np.unique(arr).tolist()) # getting unique classes
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_values))] # generating random colors for each unique classes

    Rarr = np.zeros_like(arr, dtype = 'float64') # Red
    Garr = np.zeros_like(arr, dtype = 'float64') # Green
    Barr = np.zeros_like(arr, dtype = 'float64') # Blue
    for val, col in zip(unique_values, colors):
        Rarr[arr == val ] = col[0]
        Garr[arr == val ] = col[1]
        Barr[arr == val ] = col[2]
        
    return np.stack([Rarr, Garr, Barr])