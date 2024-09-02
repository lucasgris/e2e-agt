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
notes_threshold = float(sys.argv[3])
output_csv = f"{exp_dir}/result_notes_{os.path.basename(data_csv)}" 
mean_output_csv = sys.argv[4]

if '--skip-ffm' in sys.argv:
    skip_ffm = True
    output_csv = f"{exp_dir}/result_notes_skip_ffm_{os.path.basename(data_csv)}"
else:
    skip_ffm = False
    
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
    f.write("instance,multipitch_precision,multipitch_recall,multipitch_accuracy,"
                    "precision_no_offset,recall_no_offset,f_measure_no_offset,"
                    "precision,recall,f_measure,mse\n")
if not os.path.exists(mean_output_csv):
    with open(mean_output_csv, "w") as f:
        f.write("exp_dir;mean_precision;std_precision;mean_precision_no_offset;std_precision_no_offset;mean_recall;std_recall;mean_recall_no_offset;std_recall_no_offset;mean_f_measure;std_f_measure;mean_f_measure_no_offset;std_f_measure_no_offset;mean_mse;std_mse\n")
    
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
        if skip_ffm:
            ffm = torch.zeros_like(ffm)
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


all_precision, all_recall, all_f_measure = [], [], []
all_precision_no_offset, all_recall_no_offset, all_f_measure_no_offset = [], [], []
all_multipitch_precision, all_multipitch_recall, all_multipitch_acc = [], [], []
all_mse = []

for instance_name in instances:
        target_notes = instances[instance_name]['target_notes']
        pred_notes = instances[instance_name]['pred_notes']
        
        target_notes = target_notes.astype(int)
        pred_notes = (pred_notes >= notes_threshold).astype(int)

        try:
            mse = np.mean((target_notes - pred_notes) ** 2)
        
            ref_time, ref_freqs = notes_to_hz_mir_multipitch(target_notes)
            est_time, est_freqs = notes_to_hz_mir_multipitch(pred_notes)
            
            multipitch_scores = mir_eval.multipitch.evaluate(
                ref_time, ref_freqs, est_time, est_freqs
            )
        
            ref_intervals, ref_pitches = notes_to_hz_mir_transcription(target_notes)
            est_intervals, est_pitches = notes_to_hz_mir_transcription(pred_notes)
            
            multipitch_precision = multipitch_scores["Precision"]
            multipitch_recall = multipitch_scores["Recall"]
            multipitch_acc = multipitch_scores["Accuracy"]
            
            precision, recall, f_measure, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals, ref_pitches, est_intervals, est_pitches
            )
            
            precision_no_offset, recall_no_offset, f_measure_no_offset, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals, ref_pitches, est_intervals, est_pitches, offset_ratio=None
            )
            
            all_mse.append(mse)
            all_multipitch_precision.append(multipitch_precision)
            all_multipitch_recall.append(multipitch_recall)
            all_multipitch_acc.append(multipitch_acc)
            all_precision.append(precision)
            all_recall.append(recall)
            all_f_measure.append(f_measure)
            all_precision_no_offset.append(precision_no_offset)
            all_recall_no_offset.append(recall_no_offset)
            all_f_measure_no_offset.append(f_measure_no_offset)
            
            with open(output_csv, "a") as f:
                f.write(f"{instance_name},{multipitch_precision},{multipitch_recall},{multipitch_acc},"
                        f"{precision_no_offset},{recall_no_offset},{f_measure_no_offset},"
                        f"{precision},{recall},{f_measure},{mse}\n")
        except Exception as e:
            print(f"Error in {instance_name}: {e}")
        
        
with open(output_csv, "a") as f:
    f.write(f"average,{np.mean(all_multipitch_precision)},{np.mean(all_multipitch_recall)},{np.mean(all_multipitch_acc)},"
            f"{np.mean(all_precision_no_offset)},{np.mean(all_recall_no_offset)},{np.mean(all_f_measure_no_offset)},"
            f"{np.mean(all_precision)},{np.mean(all_recall)},{np.mean(all_f_measure)},{np.mean(all_mse)}\n")

#"exp_dir,mean_precision,std_precision,mean_precision_no_offset,std_precision_no_offset,mean_recall,std_recall,mean_recall_no_offset,std_recall_no_offset,mean_f_measure,std_f_measure,mean_f_measure_no_offset,std_f_measure_no_offset,mean_mse,std_mse\n")

with open(mean_output_csv, "a") as f:
    f.write(f"{exp_dir};{np.mean(all_precision)};{np.std(all_precision)};{np.mean(all_precision_no_offset)};{np.std(all_precision_no_offset)};{np.mean(all_recall)};{np.std(all_recall)};{np.mean(all_recall_no_offset)};{np.std(all_recall_no_offset)};{np.mean(all_f_measure)};{np.std(all_f_measure)};{np.mean(all_f_measure_no_offset)};{np.std(all_f_measure_no_offset)};{np.mean(all_mse)};{np.std(all_mse)}\n")