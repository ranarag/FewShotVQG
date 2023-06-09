"""Transform all the IQ VQA dataset into a hdf5 dataset.
"""

from PIL import Image
from torchvision import transforms
import torch
import argparse
import json
import h5py
import numpy as np
import os
import progressbar

from train_utils import Vocabulary
from vocab import load_vocab
from vocab import process_text
import json
from torchvision import transforms
# from models import IEncoder
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.utils.data as data
from tqdm import tqdm
BATCH_SIZE = 128
from ..models import IEncoder
def create_answer_mapping(annotations, ans2cat):
    """Returns mapping from question_id to answer.

    Only returns those mappings that map to one of the answers in ans2cat.

    Args:
        annotations: VQA annotations file.
        ans2cat: Map from answers to answer categories that we care about.

    Returns:
        answers: Mapping from question ids to answers.
        image_ids: Set of image ids.
    """
    answers = {}
    image_ids = set()
    for q in annotations['annotations']:
        question_id = q['question_id']
        answer = q['multiple_choice_answer']
        if answer in ans2cat:
            answers[question_id] = answer
            image_ids.add(q['image_id'])
    return answers, image_ids

def save_dataset(image_dir1, image_dir2, questions, annotations, vocab, ans2cat, output,
                 im_size=224, max_q_length=20, max_a_length=4,
                 with_answers=False):
    """Saves the Visual Genome images and the questions in a hdf5 file.

    Args:
        image_dir: Directory with all the images.
        questions: Location of the questions.
        annotations: Location of all the annotations.
        vocab: Location of the vocab file.
        ans2cat: Mapping from answers to category.
        output: Location of the hdf5 file to save to.
        im_size: Size of image.
        max_q_length: Maximum length of the questions.
        max_a_length: Maximum length of the answers.
        with_answers: Whether to also save the answers.
    """
    # Load the data.
    vocab = load_vocab(vocab)
    # print(len(vocab))
    with open(annotations) as f:
        annos = json.load(f)
    with open(questions) as f:
        questions = json.load(f)
    encoder = IEncoder()
    encoder.load_state_dict(torch.load('autoencoder_models/Encoder-epoch-44.pkl'))
    encoder.cuda()

    # Get the mappings from qid to answers.
    qid2ans, image_ids = create_answer_mapping(annos, ans2cat)
    total_questions = len(list(qid2ans.keys()))
    total_images = len(image_ids)
    print("Number of images to be written: %d" % total_images)
    print("Number of QAs to be written: %d" % total_questions)

    h5file = h5py.File(output, "w")
    d_questions = h5file.create_dataset(
        "questions", (total_questions, max_q_length), dtype='i')
    d_indices = h5file.create_dataset(
        "image_indices", (total_questions,), dtype='i')
    d_images = h5file.create_dataset(
        "images", (total_images, 64, 676), dtype='f')
#     d_images = h5file.create_dataset(
#        "images", (total_images, im_size, im_size, 3), dtype='f')
    d_answers = h5file.create_dataset(
        "answers", (total_questions, max_a_length), dtype='i')
    d_answer_types = h5file.create_dataset(
        "answer_types", (total_questions,), dtype='i')


#     Create the transforms we want to apply to every image.
#     transform = transforms.Compose([
#         transforms.Resize((im_size, im_size))])
    transform = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224,
                                     scale=(1.00, 1.2),
                                     ratio=(0.75, 1.3333333333333333)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])])
    # Iterate and save all the questions and images.
    bar = progressbar.ProgressBar(maxval=total_questions)
    i_index = 0
    q_index = 0
    im_list = []
    im_id_list = []
    done_img2idx = {}
    for entry in tqdm(questions['questions']):
        image_id = entry['image_id']
        question_id = entry['question_id']
        if image_id not in image_ids:
            continue
        if question_id not in qid2ans:
            continue
        if image_id in im_id_list:
            continue
        if image_id not in done_img2idx:
            #try:
            path1 = "%d.jpg" % (image_id)
            path2 = "%012d.jpg" % (image_id)

            if os.path.isfile(os.path.join(image_dir1, path1)):
                image = Image.open(os.path.join(image_dir1, path1)).convert('RGB')
            elif os.path.isfile(os.path.join(image_dir1, path2)):
                image = Image.open(os.path.join(image_dir1, path2)).convert('RGB')
            #if os.path.isfile(os.path.join(image_dir2, path1)):
            #    image = Image.open(os.path.join(image_dir2, path1)).convert('RGB')
            #else:
            #    image = Image.open(os.path.join(image_dir2, path2)).convert('RGB')
            
#             image = transform(image).unsqueeze(0).cuda()
            image = transform(image).cuda()
            im_list.append(image)
            im_id_list.append(image_id)
#             break
            if len(im_list) ==  BATCH_SIZE:
            # print()
                img_tensors = torch.stack(im_list, dim=0).cuda()
#                 print(img_tensors.size())
                img_embeds = encoder(img_tensors).cpu().detach().numpy()
#                 print(img_embeds.shape)
#                 exit()
                for ind in range(BATCH_SIZE):
#                     print("DONE")
                    d_images[i_index+ ind, :, :] = img_embeds[ind,:,:]
#             d_images[i_index, :, :] = encoder(image).squeeze().cpu().detach().numpy()
                
                    done_img2idx[im_id_list[ind]] = i_index + ind
                i_index += BATCH_SIZE
                im_list = []
                im_id_list = []
    if len(im_list):
        img_tensors = torch.zeros(BATCH_SIZE, 3, 224, 224).cuda()
        img_cuda = torch.stack(im_list, dim=0).cuda()
        img_tensors[:img_cuda.size(0),:, :] = img_cuda
        img_embeds = encoder(img_tensors).cpu().detach().numpy()
        for ind in range(len(im_list)):
            d_images[i_index+ ind, :, :] = img_embeds[ind,:,:]
            done_img2idx[im_id_list[ind]] = i_index + ind
#         print("DONE")
    for entry in questions['questions']:
        image_id = entry['image_id']
        question_id = entry['question_id']
        if image_id not in image_ids:
            continue
        if question_id not in qid2ans:
            continue
        # print(entry['question'])
        q, length = process_text(entry['question'], vocab,
                                 max_length=max_q_length)
#         # print(q)
        d_questions[q_index, :length] = q
        answer = qid2ans[question_id]
        a, length = process_text(answer, vocab,
                                 max_length=max_a_length)
        d_answers[q_index, :length] = a
        d_answer_types[q_index] = int(ans2cat[answer])
        d_indices[q_index] = done_img2idx[image_id]
        q_index += 1
        bar.update(q_index)

    h5file.close()
    print("Number of images written: %d" % i_index)
    print("Number of QAs written: %d" % q_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Inputs.
    parser.add_argument('--image-dir', type=str, default='data/vqa/train2014',
                        help='directory for resized images')
    parser.add_argument('--questions', type=str,
                        default='data/vqa/v2_OpenEnded_mscoco_train2014_questions.json',
                        help='Path for train annotation file.')
    parser.add_argument('--annotations', type=str,
                        default='data/vqa/v2_mscoco_train2014_annotations.json',
                        help='Path for train annotation file.')
    parser.add_argument('--cat2ans', type=str,
                        default='data/vqa/iq_dataset.json',
                        help='Path for the answer types.')
    parser.add_argument('--vocab-path', type=str,
                        default='data/processed/vocab_iq.json',
                        help='Path for saving vocabulary wrapper.')

    # Outputs.
    parser.add_argument('--output', type=str,
                        default='data/processed/latest_train_img_iq_dataset.hdf5',
                        help='directory for resized images.')
    parser.add_argument('--cat2name', type=str,
                        default='data/processed/cat2name.json',
                        help='Location of mapping from category to type name.')

    # Hyperparameters.
    parser.add_argument('--im_size', type=int, default=224,
                        help='Size of images.')
    parser.add_argument('--max-q-length', type=int, default=20,
                        help='maximum sequence length for questions.')
    parser.add_argument('--max-a-length', type=int, default=4,
                        help='maximum sequence length for answers.')
    args = parser.parse_args()

    ans2cat = {}
    with open(args.cat2ans) as f:
        cat2ans = json.load(f)
    # cats = sorted(cat2ans.keys())
    # with open(args.cat2name, 'w') as f:
    #     json.dump(cats, f)
    print(cat2ans.keys())
    
    val_cat =  ['activity', 'animal', 'predicate', 
                'other', 'binary', 
                'stuff']

    cat_keys = ['count', 'binary', 
                'predicate', 
                'material', 'time', 
                'color', 'attribute', 
                'object', 'stuff', 
                'food', 'shape', 
                'other', 'location', 
                'animal', 'spatial', 'activity']
#    for key in cat_keys:
#        if key not in val_cat:
#            del cat2ans[key]
#        else:
#            print(key)
    
    # print(cat2ans.keys())
    # exit()
    cats = sorted(cat2ans.keys())
#     with open(args.cat2name, 'w') as f:
#         json.dump(cats, f)
    for cat in cat2ans:
        if cat in val_cat:
            continue
        for ans in cat2ans[cat]:
            ans2cat[ans] = cats.index(cat)
    save_dataset(args.image_dir, 'data/vqa/val2014', args.questions, args.annotations, args.vocab_path,
                 ans2cat, args.output, im_size=args.im_size,
                 max_q_length=args.max_q_length, max_a_length=args.max_a_length)
    print(('Wrote dataset to %s' % args.output))
    # Hack to avoid import errors.
    Vocabulary()
