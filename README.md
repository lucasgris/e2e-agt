# End-to-end Guitar Transcription

Training a deep learning model to transcribe guitar audio to guitar tablature.

## Installation

1. Clone the repository
2. Install the required packages
```bash
pip install -r requirements.txt
```

## Preparing data

To train the models, you need to prepare the metadata csv files. The metadata csv files should have the necessary columns depending on the type of the training data. See the `data` directory for examples (GuitarSet). For aligned data, you need to have the output tablature files processed, and for unaligned data, you need to have the sequence of each string in the CSV file.

You can use the scripts in the `scripts` directory to prepare the metadata csv files for the GuitarSet, IDMT, EGDB and SynthTab datasets.

## Training

To train the model, run the following command:
```bash
python train.py --config-name <config_name>
```

There are several configuration files in the `configs` directory. You can create your own configuration file by modifying the existing ones. Currently the best performing model is the `cnn`.

There are two main approaches to train the model: using aligned data (trained with cross entropy) and using unaligned data (trained with CTC loss). The `cnn_ce` config uses the aligned data approach and the `cnn_ctc` uses the unaligned data approach.

### Multitask prediction

You can also train the model to predict tablature and notes at the same time. The process is similar to training a single task model, just use the right configuration file.

### Creating a new model

To create a new model, you need to create a new configuration file in the `configs` directory and a new model class in the `models` directory, importing the model class in the `models/__init__.py` file.

```
Target tablature:   
e|----------------9-------8------8---9----------------------------------------------------------------|
B|------------------------------------6---------------------------------------------------------------|
G|-------------------------------------------8------6-------6-----------------------------------------|
D|----------------------------------------------------------------------------------------------------|
A|----------------------------------------------------------------------------------------------------|
E|----------------------------------------------------------------------------------------------------|

Model trained with cross entropy:
e|------------------9--------8-------8--9-------------------------------------------------------------|
B|----------------------------------------------------------------------------------------------------|
G|-----------------------------------------10-------8-------6--------6--------------------------------|
D|---------------------------------------------------------------11-----------------------------------|
A|----------------------------------------------------------------------------------------------------|
E|----------------------------------------------------------------------------------------------------|

Model trained with CTC loss:
e|------------------9-9-------8-----------------------------------------------------------------------|
B|-----------------------------------------6----------------------------------------------------------|
G|-----------------------------------------------8-8-8------6-6-------6-------------------------------|
D|----------------------------------------------------------------------------------------------------|
A|----------------------------------------------------------------------------------------------------|
E|----------------------------------------------------------------------------------------------------|
```