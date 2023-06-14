# Few-Shot Visual Question Generation

## FSVQG
The implementation to the FSVQG model can be found in the folder named FSVQG. The structure of the code is adopted from https://github.com/ranjaykrishna/iq and https://github.com/AaronCCWong/Show-Attend-and-Tell

### Installing Dependencies

In order to clone our repository and install all the required dependencies, follow these set of commands:

```
git clone https://github.com/ranarag/FewShotVQG.git
cd FewShotVQG/FSVQG/
virtualenv -p python2.7 env
source env/bin/activate
pip install -r requirements.txt
git submodule init
git submodule update
mkdir -p data/processed
```

### Preparing Data

Download the train and test sets of the <a href="https://visualqa.org/download.html">VQA Dataset</a>.

In order to prepare the data for training and evaluation, follow these set of commands:

```
# Create the vocabulary file.
python utils/vocab.py
python utils/vocab_ans.py

# Get the Bert embeddings(optional)
python get_bert_embeds_from_vocab.py 

# Create the hdf5 dataset.
python utils/store_dataset.py --mode Train --image-encoder resnet
python utils/store_dataset.py --output data/processed/val_resnet_img_dataset.hdf5 --questions data/vqa/v2_OpenEnded_mscoco_val2014_questions.json --annotations data/vqa/v2_mscoco_val2014_annotations.json --image-dir data/vqa/val2014 --mode Test --image-encoder resnet
```

### Training and Evaluation

For training the answer + category model, run the following command:

```
python meta_train_ans_cats.py --mode Train --model <model_name> --network resnet --bert-embed '' --bert-ans-embed '' --train_query 10 --dataset-type vqg --dataset data/processed/train_resnet_img_dataset.hdf5 --val-dataset data/processed/val_resnet_img_dataset.hdf5
```

For evaluation, set the *--mode*  argument to *Test*.

Similarly, to run the category model use the file **meta_train_cats.py** and the answer model **meta_train_ans.py** files respectively.

To run the corresponding **NoSS** versions, of the corresponding models set the *--scaling-shifting* argument to **True**.


## VQG-23
The VQG-23 dataset can be found in the folder named VQG-23. The folder contains the following two files:

[1] - proposed_train_splits.json – contains a json dict of instances for the training split of the VQG-23 dataset.

[2] - proposed_test_splits.json – contains a json dict of instances for the testing split of the VQG-23 dataset.

Each entry in the dict of (1) and (2) has question-id as key and another dict as value. The value dict contains the following entries:
 - **image_id**:  The filename of the image
 - **question**: The  question
 - **answer**: The answer
 - **dataset**:  Source dataset (vqa or vgenome)
 - **qid**: Question-id
 - **Category**: The category name

