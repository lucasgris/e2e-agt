import os
import csv
import logging 

import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import mir_eval

from utils.util import load_config, print_tab, resize_target, tab_to_hz_mir_eval, get_ffm
from models.cnn import CNN
from core.audio_processor import AudioProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

import sys
exp_dir = sys.argv[1]
data_csv = sys.argv[2]
output_csv = f"{exp_dir}/result_{os.path.basename(data_csv)}"
mean_output_csv = sys.argv[3]

logging.basicConfig(filename=f"{exp_dir}/test.log", encoding='utf-8', level=logging.DEBUG)

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

with open(data_csv, "r") as f:
    reader = csv.DictReader(f, delimiter=";")
    data = list(reader)

print("Begin testing. Total samples:", len(data))

output_csv = open(output_csv, "w")
if config.predict_onsets_and_frets:
    output_csv.write("segment;"
                     "onsets_precision;onsets_recall;onsets_f_measure;"
                     "strings_onsets_precision;strings_onsets_recall;strings_onsets_f_measure;"
                     "frets_precision;frets_recall;frets_f_measure;"
                     "strings_frets_precision;strings_frets_recall;strings_frets_f_measure\n")
    
    if not os.path.exists(mean_output_csv):
        with open(mean_output_csv, "w") as f:
            f.write("exp_dir;mean_onsets_precision;std_onsets_precision;mean_onsets_recall;std_onsets_recall;mean_onsets_f_measure;std_onsets_f_measure;")

elif config.predict_tab:
    output_csv.write("segment;precision;recall;f_measure;strings_precision;strings_recall;strings_f_measure\n")

    if not os.path.exists(mean_output_csv):
        with open(mean_output_csv, "w") as f:
            f.write("exp_dir;mean_precision;std_precision;mean_recall;std_recall;mean_f_measure;std_f_measure\n")

instances = {}

for row in data:
    audio_path = row["segment_path"]
    print(os.path.basename(audio_path))
    instance_name = os.path.basename(row["file_name"])
    
    onsets = tab = torch.load(row["tab_path"], map_location="cpu").float().to(device)
    onsets = tab = resize_target(tab, target_len=config.target_len_frames, upsample_method=config.data.target_len_frames_upsample_method).argmax(dim=-1)
        
    frets = torch.load(row["frets_path"], map_location="cpu").float().to(device)
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
    frets = frets.argmax(dim=-1)
    
    audio = audio_processor.load_wav(audio_path).to(device)
    feature = audio_processor.wav2feature(audio)
    feature = torch.tensor(feature).to(device)
    feature = feature.unsqueeze(0)

    with torch.no_grad():
        output = cnn(feature, ffm=ffm)
        if config.predict_onsets_and_frets:
            pred_frets = output["frets"].squeeze()
            pred_frets = resize_target(pred_frets, target_len=config.target_len_frames, upsample_method=config.data.target_len_frames_upsample_method).argmax(dim=-1)
            pred_onsets = output["onsets"].squeeze()
            pred_onsets = resize_target(pred_onsets, target_len=config.target_len_frames, upsample_method=config.data.target_len_frames_upsample_method).argmax(dim=-1)
        elif config.predict_tab:
            pred_tab = output["tab"].squeeze()
            pred_tab = resize_target(pred_tab, target_len=config.target_len_frames, upsample_method=config.data.target_len_frames_upsample_method).argmax(dim=-1)
    
    if config.predict_onsets_and_frets:
        print("\nTarget frets:");print_tab(frets)
        logging.debug(f"\nTarget frets: {frets}")
        print("\nPredicted frets:");print_tab(pred_frets)
        logging.debug(f"\nPredicted frets: {pred_frets}")
        print("\nTarget onsets:");print_tab(onsets)
        logging.debug(f"\nTarget onsets: {onsets}")
        print("\nPredicted onsets:");print_tab(pred_onsets)
        logging.debug(f"\nPredicted onsets: {pred_onsets}")
    elif config.predict_tab:
        print("\nTarget tab:");print_tab(tab)
        logging.debug(f"\nTarget tab: {tab}")
        print("\nPredicted tab:");print_tab(pred_tab)
        logging.debug(f"\nPredicted tab: {pred_tab}")
        
    if config.predict_onsets_and_frets:
        frets = frets.squeeze().cpu().numpy()
        pred_frets = pred_frets.squeeze().cpu().numpy()
        onsets = onsets.squeeze().cpu().numpy()
        pred_onsets = pred_onsets.squeeze().cpu().numpy()
    elif config.predict_tab:
        tab = tab.squeeze().cpu().numpy()
        pred_tab = pred_tab.squeeze().cpu().numpy()
    
    if instance_name not in instances:
        if config.predict_onsets_and_frets:
            instances[instance_name] = {
                "frets": frets,
                "pred_frets": pred_frets,
                "onsets": onsets,
                "pred_onsets": pred_onsets
            }
        elif config.predict_tab:
            instances[instance_name] = {
                "tab": tab,
                "pred_tab": pred_tab
            }
    else:
        if config.predict_onsets_and_frets:
            instances[instance_name]['frets'] = np.concatenate((instances[instance_name]['frets'], frets), axis=1)
            instances[instance_name]['pred_frets'] = np.concatenate((instances[instance_name]['pred_frets'], pred_frets), axis=1)
            instances[instance_name]['onsets'] = np.concatenate((instances[instance_name]['onsets'], onsets), axis=1)
            instances[instance_name]['pred_onsets'] = np.concatenate((instances[instance_name]['pred_onsets'], pred_onsets), axis=1)
        elif config.predict_tab:
            instances[instance_name]['tab'] = np.concatenate((instances[instance_name]['tab'], tab), axis=1)
            instances[instance_name]['pred_tab'] = np.concatenate((instances[instance_name]['pred_tab'], pred_tab), axis=1)
    
    if config.predict_onsets_and_frets:
        logging.debug(f"\nFrets numpy: {frets}")
        logging.debug(f"PRED Frets numpy: {pred_frets}")
        logging.debug(f"\nOnsets numpy: {onsets}")
        logging.debug(f"PRED Onsets numpy: {pred_onsets}")
    elif config.predict_tab:
        logging.debug(f"\nTAB numpy: {tab}")
        logging.debug(f"PRED TAB numpy: {pred_tab}")

if config.predict_onsets_and_frets:
    all_onsets_precision, all_onsets_recall, all_onsets_f_measure = [], [], []
    all_frets_precision, all_frets_recall, all_frets_f_measure = [], [], []
elif config.predict_tab:
    all_precision, all_recall, all_f_measure = [], [], []

def isNaN(num):
    return num != num

for instance_name in instances:
    print(instance_name)
    logging.debug(instance_name)
    if config.predict_onsets_and_frets:
        frets = instances[instance_name]['frets']
        pred_frets = instances[instance_name]['pred_frets']
        onsets = instances[instance_name]['onsets']
        pred_onsets = instances[instance_name]['pred_onsets']
        logging.debug(f"\nFrets: {frets}")
        logging.debug(f"PRED Frets: {pred_frets}")
        logging.debug(f"\nOnsets: {onsets}")
        logging.debug(f"PRED Onsets: {pred_onsets}")
        
        ref_intervals_onsets, ref_pitches_onsets = tab_to_hz_mir_eval(onsets)
        est_intervals_onsets, est_pitches_onsets = tab_to_hz_mir_eval(pred_onsets)
        ref_intervals_frets, ref_pitches_frets = tab_to_hz_mir_eval(frets)
        est_intervals_frets, est_pitches_frets = tab_to_hz_mir_eval(pred_frets)
        
        strings_onsets_precision, strings_onsets_recall, strings_onsets_f_measure = [], [], []
        strings_frets_precision, strings_frets_recall, strings_frets_f_measure = [], [], []
        for s in range(6):
            logging.debug(f"\nString {s+1}")
            logging.debug(f"\nReference intervals and pitches for onsets: {list(zip(ref_intervals_onsets[s].tolist(), ref_pitches_onsets[s]))}")
            logging.debug(f"\nPredicted intervals and pitches for onsets: {list(zip(est_intervals_onsets[s].tolist(), est_pitches_onsets[s]))}")
            logging.debug(f"\nReference intervals and pitches for frets: {list(zip(ref_intervals_frets[s].tolist(), ref_pitches_frets[s]))}")
            logging.debug(f"\nPredicted intervals and pitches for frets: {list(zip(est_intervals_frets[s].tolist(), est_pitches_frets[s]))}")
            
            if ref_intervals_onsets[s].shape[0] == 0 or est_intervals_onsets[s].shape[0] == 0:
                logging.debug(f"\nString {s+1} not possible to calculate metrics for onsets")
            else:
                p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                    ref_intervals_onsets[s], ref_pitches_onsets[s], est_intervals_onsets[s], est_pitches_onsets[s], offset_ratio=None
                )
                logging.debug(f"\nString {s+1} Precision for onsets: {p}, Recall: {r}, F-measure: {f}")
                if isNaN(p) or isNaN(r) or isNaN(f):
                    p = r = f = 0
                strings_onsets_precision.append(p)
                strings_onsets_recall.append(r)
                strings_onsets_f_measure.append(f)
            
            if ref_intervals_frets[s].shape[0] == 0 or est_intervals_frets[s].shape[0] == 0:
                logging.warning(f"\nString {s+1} not possible to calculate metrics for frets")
            else:
                p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                    ref_intervals_frets[s], ref_pitches_frets[s], est_intervals_frets[s], est_pitches_frets[s] , offset_ratio=None
                )
                if isNaN(p) or isNaN(r) or isNaN(f):
                    p = r = f = 0
                logging.debug(f"\nString {s+1} Precision for frets: {p}, Recall: {r}, F-measure: {f}")
                strings_frets_precision.append(p)
                strings_frets_recall.append(r)
                strings_frets_f_measure.append(f)
                
        onsets_precision = np.mean(strings_onsets_precision)
        onsets_recall = np.mean(strings_onsets_recall)
        onsets_f_measure = np.mean(strings_onsets_f_measure)
        frets_precision = np.mean(strings_frets_precision)
        frets_recall = np.mean(strings_frets_recall)
        frets_f_measure = np.mean(strings_frets_f_measure)
        
        if isNaN(onsets_precision) or isNaN(onsets_recall) or isNaN(onsets_f_measure):
            onsets_precision = onsets_recall = onsets_f_measure = 0
        if isNaN(frets_precision) or isNaN(frets_recall) or isNaN(frets_f_measure):
            frets_precision = frets_recall = frets_f_measure = 0
        all_onsets_precision.append(onsets_precision)
        all_onsets_recall.append(onsets_recall)
        all_onsets_f_measure.append(onsets_f_measure)
        all_frets_precision.append(frets_precision)
        all_frets_recall.append(frets_recall)
        all_frets_f_measure.append(frets_f_measure)
        
        logging.debug(f"\nOnsets Precision: {onsets_precision}, Recall: {onsets_recall}, F-measure: {onsets_f_measure}")
        logging.debug(f"\nFrets Precision: {frets_precision}, Recall: {frets_recall}, F-measure: {frets_f_measure}")
        print(f"Onsets Precision: {onsets_precision}, Recall: {onsets_recall}, F-measure: {onsets_f_measure}")
        print(f"Frets Precision: {frets_precision}, Recall: {frets_recall}, F-measure: {frets_f_measure}")
        
        output_csv.write(f"{instance_name};"
                            f"{onsets_precision};{onsets_recall};{onsets_f_measure};"
                            f"{strings_onsets_precision};{strings_onsets_recall};{strings_onsets_f_measure};"
                            f"{frets_precision};{frets_recall};{frets_f_measure};"
                            f"{strings_frets_precision};{strings_frets_recall};{strings_frets_f_measure}\n")
               
    elif config.predict_tab:
        tab = instances[instance_name]['tab']
        pred_tab = instances[instance_name]['pred_tab']
        logging.debug(f"\nTAB: {tab}")
        logging.debug(f"PRED TAB: {pred_tab}")
        
        ref_intervals, ref_pitches = tab_to_hz_mir_eval(tab)
        est_intervals, est_pitches = tab_to_hz_mir_eval(pred_tab)
        
        strings_onsets_precision, strings_onsets_recall, strings_onsets_f_measure = [], [], []

        for s in range(6):
            logging.debug(f"\nString {s+1}")
            logging.debug(f"\nReference intervals and pitches: {list(zip(ref_intervals[s].tolist(), ref_pitches[s]))}")
            logging.debug(f"\nPredicted intervals and pitches: {list(zip(est_intervals[s].tolist(), est_pitches[s]))}")
            
            if ref_intervals[s].shape[0] == 0 or est_intervals[s].shape[0] == 0:
                logging.debug(f"\nString {s+1} not possible to calculate metrics")
            else:
                p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                    ref_intervals[s], ref_pitches[s], est_intervals[s], est_pitches[s], offset_ratio=None
                )
                logging.debug(f"\nString {s+1} Precision: {p}, Recall: {r}, F-measure: {f}")
                if isNaN(p) or isNaN(r) or isNaN(f):
                    p = r = f = 0
                strings_onsets_precision.append(p)
                strings_onsets_recall.append(r)
                strings_onsets_f_measure.append(f)
                
        precision = np.mean(strings_onsets_precision)
        recall = np.mean(strings_onsets_recall)
        f_measure = np.mean(strings_onsets_f_measure)
        
        if isNaN(precision) or isNaN(recall) or isNaN(f_measure):
            precision = recall = f_measure = 0
        all_precision.append(precision)
        all_recall.append(recall)
        all_f_measure.append(f_measure)
        
        logging.debug(f"\nPrecision: {precision}, Recall: {recall}, F-measure: {f_measure}")
        print(f"Precision: {precision}, Recall: {recall}, F-measure: {f_measure}")
        
        output_csv.write(f"{instance_name};"
                            f"{precision};{recall};{f_measure};"
                            f"{strings_onsets_precision};{strings_onsets_recall};{strings_onsets_f_measure}\n")
                
    
print("All samples tested.")
if config.predict_onsets_and_frets:
    with open(mean_output_csv, "a") as f:
        f.write(f"{exp_dir};{np.mean(all_onsets_precision)};{np.std(all_onsets_precision)};{np.mean(all_onsets_recall)};{np.std(all_onsets_recall)};{np.mean(all_onsets_f_measure)};{np.std(all_onsets_f_measure)};")
    all_onsets_precision = np.mean(all_onsets_precision)
    all_onsets_recall = np.mean(all_onsets_recall)
    all_onsets_f_measure = np.mean(all_onsets_f_measure)
    all_frets_precision = np.mean(all_frets_precision)
    all_frets_recall = np.mean(all_frets_recall)
    all_frets_f_measure = np.mean(all_frets_f_measure)
    
    logging.debug(f"\nAll samples Onsets Precision: {all_onsets_precision}, Recall: {all_onsets_recall}, F-measure: {all_onsets_f_measure}")
    logging.debug(f"\nAll samples Frets Precision: {all_frets_precision}, Recall: {all_frets_recall}, F-measure: {all_frets_f_measure}")
    print(f"All samples Onsets Precision: {all_onsets_precision}, Recall: {all_onsets_recall}, F-measure: {all_onsets_f_measure}")
    print(f"All samples Frets Precision: {all_frets_precision}, Recall: {all_frets_recall}, F-measure: {all_frets_f_measure}")
    
    output_csv.write(f"ALL;"
                     f"{all_onsets_precision};{all_onsets_recall};{all_onsets_f_measure};"
                        f";;;"
                        f"{all_frets_precision};{all_frets_recall};{all_frets_f_measure};"
                        f";;;\n")

elif config.predict_tab:
    with open(mean_output_csv, "a") as f:
        f.write(f"{exp_dir};{np.mean(all_precision)};{np.std(all_precision)};{np.mean(all_recall)};{np.std(all_recall)};{np.mean(all_f_measure)};{np.std(all_f_measure)}\n")
    all_precision = np.mean(all_precision)
    all_recall = np.mean(all_recall)
    all_f_measure = np.mean(all_f_measure)
    
    logging.debug(f"\nAll samples Precision: {all_precision}, Recall: {all_recall}, F-measure: {all_f_measure}")
    print(f"All samples Precision: {all_precision}, Recall: {all_recall}, F-measure: {all_f_measure}")
    
    output_csv.write(f"ALL;"
                        f"{all_precision};{all_recall};{all_f_measure};"
                        f";;;\n")