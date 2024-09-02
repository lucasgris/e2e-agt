import os
import csv

all_features = os.listdir('features')
all_segments = os.listdir('segments')
all_features_in_csv = []
all_segments_in_csv = []

for file in os.listdir():
    if file.endswith('csv'):
        reader = csv.DictReader(open(file))
        for row in reader:
            feature_filename = row['feature_filename']
            onset_feature_filename = row['onset_feature_filename']
            segment_filename = row['segment_filename']
            all_features_in_csv.append(feature_filename)
            all_features_in_csv.append(onset_feature_filename)
            all_segments_in_csv.append(segment_filename)

for file in all_features:
    if file not in all_features_in_csv:
        print('Removing', file)
        # os.remove(os.path.join('features', file))
for segment in all_segments:
    if segment not in all_segments_in_csv:
        print('Removing', segment)
        # os.remove(os.path.join('segments', segment))