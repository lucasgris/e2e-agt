import io
import os
import random
import csv
import logging
import traceback
from typing import Union

import logging
import numpy as np
import PIL
import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from utils.util import load_config, print_tab, resize_target
from utils.metrics import TabMetrics, f_measure
from models.cnn import CNN
from core.audio_processor import AudioProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

import sys
exp_dir = sys.argv[1]
data_csv = sys.argv[2]
output_csv = f"{exp_dir}/result_{os.path.basename(data_csv)}"
mean_output_csv = sys.argv[3]

model_path = f"{exp_dir}/checkpoints/checkpoint_best.pt"
if not os.path.exists(model_path):
    print("Best model not found. Using last model.")
    model_path = f"{exp_dir}/checkpoints/checkpoint_last.pt"
config_path = f"{exp_dir}/.hydra/config.yaml"

config = load_config(config_path)
print(config)
cnn = CNN(config.model)
print(cnn)
cnn.load_state_dict(torch.load(model_path, map_location=device)["state_dict"])

cnn.eval()
cnn.to(device)

audio_processor = AudioProcessor(config.audio)

with open(data_csv, "r") as f:
    reader = csv.DictReader(f, delimiter=";")
    data = list(reader)

print("Begin testing. Total samples:", len(data))

output_csv = open(output_csv, "w")
output_csv.write("segment;ter;ter_no_sil;fer;fer_no_sil\n")

if not os.path.exists(mean_output_csv):
    with open(mean_output_csv, "w") as f:
        f.write("exp_dir;mean_ter;std_ter;mean_ter_no_sil;std_ter_no_sil;mean_fer;std_fer;mean_fer_no_sil;std_fer_no_sil\n")

ters = []
ters_no_sil = []
fers = []
fers_no_sil = []

for row in data:
    audio_path = row["segment_path"]
    tab = torch.load(row["tab_path"], map_location="cpu").float()
    tab = resize_target(tab, target_len=config.target_len_frames, upsample_method=config.data.target_len_frames_upsample_method).argmax(dim=-1)
    print("Target tab:")
    print_tab(tab)

    audio = audio_processor.load_wav(audio_path)
    feature = audio_processor.wav2feature(audio)
    feature = torch.tensor(feature).to(device)
    feature = feature.unsqueeze(0)

    with torch.no_grad():
        output = cnn(feature)
        pred_tab = output["tab"].squeeze()
        pred_tab = resize_target(pred_tab, target_len=config.target_len_frames, upsample_method=config.data.target_len_frames_upsample_method).argmax(dim=-1)
        print("Predicted tab:")
        print_tab(pred_tab)
        
    print()
    print(f"Tab words:           {TabMetrics.tab_to_words(tab.cpu().numpy(), collapse_silences=True)}")
    print(f"Predicted tab words: {TabMetrics.tab_to_words(pred_tab.cpu().numpy(), collapse_silences=True)}")
    ter = TabMetrics.tab_error_rate(pred_tab, tab, collapse_silences=False)
    ter_no_sil = TabMetrics.tab_error_rate(pred_tab, tab, collapse_silences=True)
    print(f"TER: {ter:.3f} | TER (no silences): {ter_no_sil:.3f}")
    fer = TabMetrics.fret_error_rate(pred_tab, tab, collapse_silences=False)
    fer_no_sil = TabMetrics.fret_error_rate(pred_tab, tab, collapse_silences=True)
    print(f"FER: {fer:.3f} | FER (no silences): {fer_no_sil:.3f}")
    
    ters.append(ter)
    ters_no_sil.append(ter_no_sil)
    fers.append(fer)
    fers_no_sil.append(fer_no_sil)
    output_csv.write(f"{os.path.basename(audio_path)};{ter};{ter_no_sil};{fer};{fer_no_sil}\n")
    
    print('='*100)

with open(mean_output_csv, "a") as f:
    f.write(f"{exp_dir};{np.mean(ters):.3f};{np.std(ters):.3f};{np.mean(ters_no_sil):.3f};{np.std(ters_no_sil):.3f};{np.mean(fers):.3f};{np.std(fers):.3f};{np.mean(fers_no_sil):.3f};{np.std(fers_no_sil):.3f}\n")