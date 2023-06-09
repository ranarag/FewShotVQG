
The structure of the code is adopted from https://github.com/ranjaykrishna/iq and https://github.com/AaronCCWong/Show-Attend-and-Tell

## Installing Dependencies

In order to clone our repository and install all the required dependencies, follow these set of commands:

```
git clone https://github.com/ranarag/proposedFewShotVQG.git
cd proposedFewShotVQG
virtualenv -p python2.7 env
source env/bin/activate
pip install -r requirements.txt
git submodule init
git submodule update
mkdir -p data/processed
```

## Preparing Data

Download the train and validation sets of the <a href="https://visualqa.org/download.html">VQA Dataset</a>.

In order to prepare the data for training and evaluation, follow these set of commands:

```
# Create the vocabulary file.
python utils/vocab.py

# Create the hdf5 dataset.
python utils/store_dataset.py
python utils/store_dataset.py --output data/processed/iq_val_dataset.hdf5 --questions data/vqa/v2_OpenEnded_mscoco_val2014_questions.json --annotations data/vqa/v2_mscoco_val2014_annotations.json --image-dir data/vqa/val2014
```

## Training and Evaluation

For training and validation, run the following command:

```
python new_train.py```

## TODOS

[1] Add the answer category module

[2] Add sampler for meta-training and testing

[3] Dataset now needs to be split based on answer category
