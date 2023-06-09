"""Loads question answering data and feeds it to the models.
"""

import h5py
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from .fixmatch import TransformFixMatch
from .vocab import process_text
import pickle as pkl
from .factory import Factory

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class DatasetLoader(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, dataset, transform=None, max_examples=None,
                 indices=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            dataset: hdf5 file with questions and images.
            images: hdf5 file with questions and imags.
            transform: image transformer.
            max_examples: Used for debugging. Assumes that we have a
                maximum number of training examples.
            indices: List of indices to use.
        """
        self.dataset = dataset
        self.transform = transform
        self.max_examples = max_examples
        self.indices = indices
        # self.annos_allowed = annos_allowed
        if not hasattr(self, 'images'):
            annos = h5py.File(self.dataset, 'r')
            self.questions = annos['questions']
            self.answers = annos['answers']
            self.answer_types = annos['answer_types']
            self.image_indices = annos['image_indices']
            self.images = annos['images']
            self.labeln = np.array(self.answer_types)
            self.unique_labels = np.unique(self.labeln)
            try:
                self.question_ids = annos['question_ids']
            except:
                self.question_ids = annos['qids']

    def __getitem__(self, index):
        """Returns one data pair (image and caption).
        """
        if self.indices is not None:
            index = self.indices[index]
        question = self.questions[index]
        answer = self.answers[index]
        answer_type = self.answer_types[index]
        image_index = self.image_indices[index]
        image = self.images[image_index]
        qids = self.question_ids[index]
        
        
        question = torch.from_numpy(question)
        answer = torch.from_numpy(answer)
#         image = torch.from_numpy(image)
        alength = answer.size(0) - answer.eq(0).sum(0).squeeze()
        qlength = question.size(0) - question.eq(0).sum(0).squeeze()
#         print("Question = {}, qlength = {}".format(question, qlength))
        
#         return
        if self.transform is None:
            image = torch.from_numpy(image)
        else:
            image = self.transform(image)
#         print(image.size())
        return (image, question, answer, answer_type,
                qlength, alength.item(), qids)

    def __len__(self):
        if self.max_examples is not None:
            return self.max_examples
        if self.indices is not None:
            return len(self.indices)
        annos = h5py.File(self.dataset, 'r')
#         print(annos['questions'].shape)
        return annos['questions'].shape[0]


class NewDatasetLoader(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, img_dataset, qid2data, dataset_type='vqg', transform=None, max_examples=None,
                 indices=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            dataset: hdf5 file with questions and images.
            images: hdf5 file with questions and imags.
            transform: image transformer.
            max_examples: Used for debugging. Assumes that we have a
                maximum number of training examples.
            indices: List of indices to use.
        """
#         self.dataset = dataset
        self.dataset_type = dataset_type
        self.transform = transform
        self.max_examples = max_examples
        self.indices = indices
        self.img_embed = h5py.File(img_dataset, 'r')
#         print(list(self.img_embed.keys())[:10])
        with open(qid2data, 'rb') as fid:
            self.qid2data = pkl.load(fid)




    def __getitem__(self, index):
        """Returns one data pair (image and caption).
        """
#         print(type(index))
#         exit()
        
        if self.dataset_type == '7w':
            
        
            index = str(index)

#             print(index)
            qid2data = self.qid2data[index]
            img_indx, answer, question, cat = qid2data['image_id'], qid2data['answer'], qid2data['question'], qid2data['cat']
#         print(qid2data.keys())
        
        else:
#             index = str(index)
            qid2data = self.qid2data[index]
            img_indx, answer, question, cat = qid2data['image_id'], qid2data['a_embed'], qid2data['q_embed'], qid2data['category']
#             print(img_indx)
#             if 'COCO' not in img_indx:
#                 img_indx = 'v7w_'+ img_indx
        
        image = self.img_embed[str(img_indx)][()]

        if self.transform is None:
            image = torch.from_numpy(image)
        else:
            image = self.transform(image)
        question = torch.from_numpy(question)
        answer = torch.from_numpy(answer)
        return image, question, answer, cat, index, img_indx

    def __len__(self):
        return len(self.qid2data)

class LMDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, data_path, max_examples=None,
                 indices=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            dataset: hdf5 file with questions and images.
            images: hdf5 file with questions and imags.
            transform: image transformer.
            max_examples: Used for debugging. Assumes that we have a
                maximum number of training examples.
            indices: List of indices to use.
        """

        self.data_path = data_path
        annos = h5py.File(data_path, 'r')
        self.questions = annos['questions']


        self.max_examples = max_examples
        self.indices = indices

    def __getitem__(self, index):
        """Returns one data pair (image and caption).
        """
        x = self.questions[index]

        y = np.zeros_like(x)
    
        y[:-1] = x[1:]
        y[-1] = 1
        y = torch.from_numpy(y).long()
        x = torch.from_numpy(x).long()
        return x, y

    def __len__(self):
        if self.max_examples is not None:
            return self.max_examples
        if self.indices is not None:
            return len(self.indices)
        else:
            annos = h5py.File(self.data_path, 'r')
            return annos['questions'].shape[0]

class FixMatchDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, data_path, mean, std, max_examples=None,
                 indices=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            dataset: hdf5 file with questions and images.
            images: hdf5 file with questions and imags.
            transform: image transformer.
            max_examples: Used for debugging. Assumes that we have a
                maximum number of training examples.
            indices: List of indices to use.
        """
        self.transform = TransformFixMatch(mean=mean, std=std)
        self.data_path = data_path
        annos = h5py.File(data_path, 'r')
        self.images = annos['images']


        self.max_examples = max_examples
        self.indices = indices

    def __getitem__(self, index):
        """Returns one data pair (image and caption).
        """
        image = self.images[index]

        weak, strong, target = self.transform(image)

        return weak, strong

    def __len__(self):
        if self.max_examples is not None:
            return self.max_examples
        if self.indices is not None:
            return len(self.indices)
        else:
            annos = h5py.File(self.data_path, 'r')
            return annos['images'].shape[0]





class UnSupDataset(data.Dataset):
    def __init__(self, data_path, clean_transform=None, noise_transform=None, max_examples=None,
                 indices=None):
 
        self.data_path = data_path
        annos = h5py.File(data_path, 'r')
        self.images = annos['images']


        self.clean_transform = clean_transform
        self.noise_transform = noise_transform
        self.max_examples = max_examples
        self.indices = indices

    def __getitem__(self, index):
        """Returns one data pair (image and caption).
        """
        image = self.images[index]
        target = self.clean_transform(image)
        input = self.noise_transform(target)

        return 

    def __len__(self):
        if self.max_examples is not None:
            return self.max_examples
        if self.indices is not None:
            return len(self.indices)
        else:
            annos = h5py.File(self.data_path, 'r')
            return annos['images'].shape[0]


def new_collate(images):
    images = torch.stack(images, 0)
    return images

def collate_fn(data):
    images, questions, answers, answer_types, qlengths, _, qids = zip(*data)
#     unique_ans = np.unique(answer_types)
#     new_ans = np.zeros_like(answer_types)
#     for indx, aid in enumerate(unique_ans):
#         ind = np.argwhere(answer_types == aid)
#         new_ans[ind] = indx
#     qids = torch.Tensor(qids).long()
    
    images = torch.stack(images, 0)
    questions = torch.stack(questions, 0).long()
    answers = torch.stack(answers, 0).long()
    answer_types = torch.Tensor(answer_types).long()
    qindices = np.flip(np.argsort(qlengths), axis=0).copy()
    qindices = torch.Tensor(qindices).long()
    qlengths = torch.stack(qlengths).long()
    return images, questions, qlengths, answers, answer_types, qindices, qids



def newcollate_fn(data):
    images, questions, answers, answer_types, qids, img_ids = zip(*data)
#     unique_ans = np.unique(answer_types)
#     new_ans = np.zeros_like(answer_types)
#     for indx, aid in enumerate(unique_ans):
#         ind = np.argwhere(answer_types == aid)
#         new_ans[ind] = indx
#     qids = torch.Tensor(new_qids).long()  
    images = torch.stack(images, 0)
    questions = torch.stack(questions, 0).long()
    answers = torch.stack(answers, 0).long()
    answer_types = torch.Tensor(answer_types).long()

    return images, questions, answers, answer_types, qids, img_ids



# class DatasetFactory(object):
#     def __init__(self):
#         self._dataset_dict = {}
        
#     def register_dataset(self, name, dataset):
#         self._dataset_dict[name] = dataset
    
#     def get_dataset_obj(self, name, *args, **kwargs):
#         try:
#             obj = self._dataset_dict[name](*args, **kwargs)
#         except KeyError as e:
#             print("{} is not implemented or linked to factory class DatasetFactory; exiting".format(name))
#             exit(1)
#         return obj

#     @property
#     def list(self):
#         for name, dataset in self._dataset_dict.items():
#             print("{} : {}".format(name, dataset))
        

datasets = Factory()
datasets.register_class("Train", DatasetLoader)
datasets.register_class("Val", NewDatasetLoader)
datasets.register_class("FxMatch", FixMatchDataset)
datasets.register_class("UnSup", UnSupDataset)
datasets.register_class("LM", LMDataset)



