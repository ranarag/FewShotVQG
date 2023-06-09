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

import json



def get_imlist(qfile):

    with open(qfile) as f:
        questions = json.load(f)
    img_list = []
    for entry in questions['questions']:
        image_id = entry['image_id']

        # if image_id not in img_list:
        img_list.append(image_id)
    img_list = list(set(img_list))
    return img_list


def main():
    output = 'data/processed/image_dataset.hdf5'
    train_qfile = 'data/vqa/v2_OpenEnded_mscoco_train2014_questions.json'
    val_qfile = 'data/vqa/v2_OpenEnded_mscoco_val2014_questions.json'
    train_img_list = get_imlist( train_qfile)
    test_img_list = get_imlist(val_qfile)
    im_size = 224
    train_image_dir = 'data/vqa/train2014'
    test_image_dir = 'data/vqa/val2014'
    print("DONE")
    h5file = h5py.File(output, "w")
    transform = transforms.Compose([
        transforms.Resize((im_size, im_size))])
    total_images = len(train_img_list) + len(test_img_list)
    d_images = h5file.create_dataset(
        "images", (total_images, im_size, im_size, 3), dtype='f')

    bar = progressbar.ProgressBar(maxval=total_images)
    i_index = 0
    for image_id in train_img_list:
        try:
            path = "%d.jpg" % (image_id)
            image = Image.open(os.path.join(train_image_dir, path)).convert('RGB')
        except IOError:
            path = "%012d.jpg" % (image_id)
            image = Image.open(os.path.join(train_image_dir, path)).convert('RGB')
        image = transform(image)
        d_images[i_index, :, :, :] = image
        i_index += 1
        bar.update(i_index)


    for image_id in test_img_list:
        try:
            path = "%d.jpg" % (image_id)
            image = Image.open(os.path.join(test_image_dir, path)).convert('RGB')
        except IOError:
            path = "%012d.jpg" % (image_id)
            image = Image.open(os.path.join(test_image_dir, path)).convert('RGB')
        image = transform(image)
        d_images[i_index, :, :, :] = image
        i_index += 1
        bar.update(i_index)
    h5file.close()

main()


