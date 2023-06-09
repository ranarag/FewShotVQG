"""Transform all the IQ VQA dataset into a hdf5 dataset.
"""

from PIL import Image
from torchvision import transforms

import argparse
import json
import h5py
import numpy as np
import os
import progressbar
import pickle as pkl
import json
import torch
# from fixmatchModel import FixMatchEncoder as IEncoder
# from efficientnet_pytorch import EfficientNet
from Autoencoder import IEncoder

from Autoencoder import IEncoder
def get_imlist(qfile):

#     with open(qfile) as f:
#         questions = json.load(f)
    with open('new_qid2data.pkl', 'rb') as fid:
        qid2d_file = pkl.load(fid)
    img_list = []
    for _, entry in qid2d_file.items():
#         print(entry)
#     for entry in questions['questions']:
        image_id = entry['image_id']

        # if image_id not in img_list:
        img_list.append(image_id)
    img_list = list(set(img_list))
    return img_list


def main():
#     encoder = EfficientNet.from_pretrained('efficientnet-b3', advprop=True)
#     encoder.load_state_dict(torch.load('autoencoder_models/encoder_imgnet_ft.pkl'))
#     encoder = IEncoder()
#     encoder.cuda()

    output = 'val_img_id_raw.hdf5'
    val_qfile = 'data/vqa/v2_OpenEnded_mscoco_val2014_questions.json'
#     val_qfile = 'data/vqa/v2_OpenEnded_mscoco_val2014_questions.json'
    val_img_list = get_imlist( val_qfile)
#     test_img_list = get_imlist(val_qfile)
    im_size = 224
    val_image_dir = 'data/vqa/val2014'
#     test_image_dir = 'data/vqa/val2014'
    print("DONE")
    h5file = h5py.File(output, "w")
#     transform = transforms.Compose([
#         transforms.Resize((im_size, im_size))])
    total_images = len(val_img_list)
    transform = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224,
                                     scale=(1.00, 1.2),
                                     ratio=(0.75, 1.3333333333333333)),
        transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
     transforms.Lambda(lambda img: img * 2.0 - 1.0)])

    bar = progressbar.ProgressBar(maxval=total_images)
    i_index = 0
    for image_id in val_img_list:
        try:
            path = "%d.jpg" % (image_id)
            image = Image.open(os.path.join(val_image_dir, path)).convert('RGB')
        except IOError:
            path = "%012d.jpg" % (image_id)
            image = Image.open(os.path.join(val_image_dir, path)).convert('RGB')
        
#         image = transform(image).unsqueeze(0).cuda()
#         print(encoder.extract_features(image).squeeze().cpu().detach().numpy().shape)
#         exit()
        h5file.create_dataset(str(image_id), data=image, dtype='f')
#         h5file.create_dataset(str(image_id), data=encoder(image).squeeze().cpu().detach().numpy(), dtype='f')
#         d_images[i_index, :, :, :] = image
        i_index += 1
        bar.update(i_index)


#     for image_id in test_img_list:
#         try:
#             path = "%d.jpg" % (image_id)
#             image = Image.open(os.path.join(test_image_dir, path)).convert('RGB')
#         except IOError:
#             path = "%012d.jpg" % (image_id)
#             image = Image.open(os.path.join(test_image_dir, path)).convert('RGB')
#         image = transform(image)
#         d_images[i_index, :, :, :] = image
#         i_index += 1
#         bar.update(i_index)
    h5file.close()

main()


