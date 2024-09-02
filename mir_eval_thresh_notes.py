import os
import csv

import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import mir_eval

from utils.util import load_config, notes_to_hz_mir_transcription, notes_to_hz_mir_multipitch, resize_target, get_ffm
from models.cnn import CNN
from core.audio_processor import AudioProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

import sys
exp_dir = sys.argv[1]
data_csv = sys.argv[2]
output_csv = f"{exp_dir}/notes_thresh.csv"  

model_path = f"{exp_dir}/checkpoints/checkpoint_best.pt"
if not os.path.exists(model_path):
    print("Best model not found. Using last model.")
    model_path = f"{exp_dir}/checkpoints/checkpoint_last.pt"
config_path = f"{exp_dir}/.hydra/config.yaml"

config = load_config(config_path)
print(config)
if not config["predict_notes"]:
    raise ValueError("This script works only for models that predict notes.")

cnn = CNN(config.model)
print(cnn)
cnn.load_state_dict(torch.load(model_path, map_location=device)["state_dict"])

cnn.eval()
cnn.to(device)

audio_processor = AudioProcessor(config.audio)

with open(output_csv, "w") as f:
    f.write("threshold,precision,recall,f_measure\n")
    
with open(data_csv, "r") as f:
    reader = csv.DictReader(f, delimiter=";")
    data = list(reader)
instances = {}


for row in data:
    audio_path = row["segment_path"]
    print(os.path.basename(audio_path))
    instance_name = os.path.basename(row["file_name"])
    
    notes = torch.load(row["notes_path"], map_location="cpu").float().to(device)
    frets = torch.load(row["frets_path"], map_location="cpu").float().to(device)
    # notes = resize_target(notes, target_len=config.target_len_frames, upsample_method=config.data.target_len_frames_upsample_method).argmax(dim=-1)
    frets = resize_target(frets, target_len=config.target_len_frames, upsample_method=config.data.target_len_frames_upsample_method)
    
    if config.insert_ffm:
        ffm_map_kernel = (
            torch.tensor(list(config.ffm.map_kernel))
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
        ).to(device)
        ffm = get_ffm(frets, ffm_map_kernel=ffm_map_kernel).to(device).unsqueeze(0)
    else:
        ffm = None
    
    audio = audio_processor.load_wav(audio_path).to(device)
    feature = audio_processor.wav2feature(audio)
    feature = torch.tensor(feature).to(device)
    feature = feature.unsqueeze(0)

    with torch.no_grad():
        output = cnn(feature, ffm=ffm)["notes"]
    
    target_notes = notes.squeeze().cpu().numpy()
    if len(output.shape) == 3:
        pred_notes = torch.nn.functional.sigmoid(output).squeeze().cpu().numpy()
    elif len(output.shape) == 4:
        pred_notes = torch.nn.functional.sigmoid(output).squeeze().cpu().numpy()[1]  # not blank MCTC
    
    if instance_name not in instances:
        instances[instance_name] = {
            'target_notes': target_notes,
            'pred_notes': pred_notes
        }
    else:
        instances[instance_name]['target_notes'] = np.concatenate((instances[instance_name]['target_notes'], target_notes), axis=1)
        instances[instance_name]['pred_notes'] = np.concatenate((instances[instance_name]['pred_notes'], pred_notes), axis=1)

for thresh in np.arange(0.1, 1.0, 0.1):
    print('-'*50)
    print(f"Threshold: {thresh}:")
    all_precision, all_recall, all_f_measure = [], [], []
    for instance_name in instances:
        target_notes = instances[instance_name]['target_notes']
        pred_notes = instances[instance_name]['pred_notes']
        
        target_notes = target_notes.astype(int)
        pred_notes = (pred_notes >= thresh).astype(int)
            
        ref_intervals, ref_pitches = notes_to_hz_mir_transcription(target_notes)
        est_intervals, est_pitches = notes_to_hz_mir_transcription(pred_notes)

        if est_intervals.size == 0:
            print(f"{instance_name}: Estimated notes are empty. Maybe threshold is too high.")
            continue

        strings_precision, strings_recall, strings_f_measure = [], [], []
        for s in range(6):
            p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals, ref_pitches, est_intervals, est_pitches
            )
            strings_precision.append(p)
            strings_recall.append(r)
            strings_f_measure.append(f)

        all_precision.append(np.mean(strings_precision))
        all_recall.append(np.mean(strings_recall))
        all_f_measure.append(np.mean(strings_f_measure))
        
    print(f"Precision: {np.mean(all_precision)}, Recall: {np.mean(all_recall)}, F-measure: {np.mean(all_f_measure)}")
    
    with open(output_csv, "a") as f:
        f.write(f"{thresh},{np.mean(all_precision)},{np.mean(all_recall)},{np.mean(all_f_measure)}\n")
               