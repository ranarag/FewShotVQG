from PIL import Image
from torchvision import transforms
import torch
import argparse
import json
import h5py
import numpy as np
import os
from tqdm import tqdm

from train_utils import Vocabulary
from vocab import load_vocab
from vocab import process_text
import json
from torchvision import transforms
from Autoencoder import IEncoder

from efficientnet_pytorch import EfficientNet


def main():
    image_dir = '../visual7w/images/'
    output = 'img_id2embeds_nonet_7w.hdf5'
    im_size=224
    with open('../visual7w/dataset_v7w_telling.json', 'r') as fid:
        dataset = json.load(fid)
    dataset = dataset['images']
    qid2data = {}
#     encoder = EfficientNet.from_pretrained('efficientnet-b3', advprop=True)
#     encoder = IEncoder()
#     encoder.load_state_dict(torch.load('autoencoder_models/encoder_imgnet_ft.pkl'))
#     encoder.cuda()
    transform = transforms.Compose([
    transforms.Resize((im_size, im_size))])
#     transform = transforms.Compose([
#         transforms.Resize((im_size, im_size)),
#         transforms.ToTensor(),
#         transforms.ToPILImage(),
#         transforms.RandomResizedCrop(224,
#                                      scale=(1.00, 1.2),
#                                      ratio=(0.75, 1.3333333333333333)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])])
    h5file = h5py.File(output, "w")
    
    for data in tqdm(dataset):
        image_id = data['image_id']
        
        path1 = "v7w_" +"%d.jpg" % (image_id)
        path2 = "v7w_" + "%012d.jpg" % (image_id)
    #     path1 = 
        if os.path.isfile(os.path.join(image_dir, path1)):
            image = Image.open(os.path.join(image_dir, path1)).convert('RGB')
        elif os.path.isfile(os.path.join(image_dir, path2)):
            image = Image.open(os.path.join(image_dir, path2)).convert('RGB')
        #if os.path.isfile(os.path.join(image_dir2, path1)):
        #    image = Image.open(os.path.join(image_dir2, path1)).convert('RGB')
        #else:
        #    image = Image.open(os.path.join(image_dir2, path2)).convert('RGB')

#         image = transform(image).unsqueeze(0).cuda()
    #     image = transform(image)
        # print()
        # exit()
        img_data = encoder.extract_features(image).squeeze().reshape(1536, -1).cpu().detach().numpy()
#         img_data = ncoder(image).squeeze().cpu().detach().numpy()
        h5file.create_dataset(str(image_id), data=img_data, dtype='f')
    #     d_images[i_index, :, :] = np.array(image)
    #             d_images[i_index, :, :] = encoder(image).squeeze().cpu().detach().numpy()
#         done_img2idx[image_id] = i_index
#         i_index += 1
        for qa in data['qa_pairs']:
            qid = qa['qa_id']
            answer = qa['answer']
            question = qa['question']
            img_id = qa['image_id']
            cat = qa['type']
            qid2data[qid] = {'image_id': str(img_id), 'question': question, 'answer': answer, 'cat': cat}

    with open('qid2data_nonet_7w.json', 'w') as fid:
        json.dump(qid2data, fid)
            

main()
        