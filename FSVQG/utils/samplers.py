""" Sampler for dataloader. """
import torch
import numpy as np
import random
from itertools import combinations 
import json
from .factory import Factory
import pickle as pkl
# Customize such as total way number of distinct classes to segment in a meta task

def nCr(n, r): 
  
    return (fact(n) / (fact(r)  
                * fact(n - r))) 
  
# Returns factorial of n 
def fact(n): 
  
    res = 1
      
    for i in range(2, n+1): 
        res = res * i 
          
    return res 

class NewCategoriesSampler7w():
    """The class to generate episodic data"""
    def __init__(self, label_dict, n_batch, K, N, Q):
        #K Way, N shot(train query), Q(test query)
        with open(label_dict, 'r') as fid:
            self.m_ind = json.load(fid)
        self.unique_labels = list(self.m_ind.keys())
#         print(self.m_ind.keys())
#         exit()
        self.K = K
        self.N = N
        self.Q = Q
        comb = combinations(self.unique_labels, K)
        self.label_combos = [c for c in comb]
        self.n_batch = 100
        print("NUM batches = {}".format(self.n_batch))
#         self.m_ind = {}
#         for i in self.unique_labels:
#             ind = np.argwhere(labeln == i).reshape(-1)
#             ind = torch.from_numpy(ind)
#             self.m_ind[i] = ind
        self.index = 0
    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i in range(self.n_batch):
            classes = self.label_combos[self.index]
            self.index = self.index % len(self.unique_labels)

            lr=[]
            dr=[]
            for c in classes:
#                 print(classes)
#                 exit()
                l = self.m_ind[c]
#                 pos = torch.randperm(len(l))[:(self.N +self.Q)]
                m= random.sample(l, self.N + self.Q)

                for i in range(0,self.N):
                    lr.append(m[i])
                    
                for i in range(self.N, (self.N +self.Q)):
                    dr.append(m[i])
                
            batch = lr + dr
#             batch=[]
#             for i in range(len(lr)):
#                 batch.append(lr[i])
            
#             for i in range(len(dr)):
#                 batch.append(dr[i])
                        
#             batch = torch.stack(batch, 0)    
                
            yield batch


        
class UnsupSampler7w():
    """The class to generate episodic data"""
    def __init__(self, qid2data, batch_size):
        with open(qid2data, 'rb') as fid:
            qid2data = pkl.load(fid)
        self.qidlist = list(qid2data.keys())
        print(self.qidlist[:10])
        self.batch_size = batch_size
        self.tot = len(self.qidlist)
    def __len__(self):
        return self.tot // self.batch_size

    def __iter__(self):
        random.shuffle(self.qidlist)
        for ind in range(0, self.tot, self.batch_size):
            yield self.qidlist[ind: ind + self.batch_size]
class CategoriesSampler():
    """The class to generate episodic data"""
    def __init__(self, labeln, unique_labels, n_batch, K, N, Q):
        #K Way, N shot(train query), Q(test query)
        self.unique_labels = unique_labels.tolist()

        self.K = K
        self.N = N
        self.Q = Q
        comb = combinations(self.unique_labels, K)
        self.label_combos = [c for c in comb]
        self.n_batch = len(self.label_combos)
        print("NUM batches = {}".format(self.n_batch))
        self.m_ind = {}
        for i in self.unique_labels:
            ind = np.argwhere(labeln == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind[i] = ind
        self.index = 0
    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        random.shuffle(self.label_combos)
        for i in range(self.n_batch):
            self.index += 1
            if self.index >= self.n_batch:
                random.shuffle(self.label_combos)
                self.index -= self.n_batch
            classes = self.label_combos[self.index]
            

            lr=[]
            dr=[]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:(self.N +self.Q)]
                m=l[pos]

                for i in range(0,self.N):
                    lr.append(m[i])
                    
                for i in range(self.N, (self.N +self.Q)):
                    dr.append(m[i])
            
            batch=[]
            for i in range(len(lr)):
                batch.append(lr[i])
            
            for i in range(len(dr)):
                batch.append(dr[i])
                        
            batch = torch.stack(batch, 0)    
                
            yield batch



class ValCategoriesSampler():
    """The class to generate episodic data"""
    def __init__(self, qid_file, N, Q):
        #K Way, N shot(train query), Q(test query)
        with open(qid_file, 'r') as fid:
            self.qid_list = json.load(fid)
#         with open(qid2ind_file, 'r') as fid:
#             self.qid2ind = json.load(fid)
        self.n_batch = len(self.qid_list)

        self.N = N
        self.Q = Q


    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            qid_combo = self.qid_list[i_batch]
            classes = qid_combo['cat']
#             batch = []
            if self.N < 0:
                qid_list = qid_combo['question_ids']
            else:
                qid_list = qid_combo['question_ids'][:self.N]
#             for qid in qid_list:

#                 batch.append(qid)
                
            yield qid_list


class ValCategoriesSamplerVQG():
    """The class to generate episodic data"""
    def __init__(self, qid_file, i_batch, p, batch_size):
        #K Way, N shot(train query), Q(test query)
        with open(qid_file, 'r') as fid:
            self.qid_list = json.load(fid)
#         with open(qid2ind_file, 'r') as fid:
#             self.qid2ind = json.load(fid)
        self.qid_combo = self.qid_list[i_batch]['question_ids'][p:]
        self.N= len(self.qid_combo)
        self.batch_size = batch_size

    def __len__(self):
        return self.N // self.batch_size
    
    def __iter__(self):
        for idx in range(0, self.N, self.batch_size):
            yield self.qid_combo[idx: idx + self.batch_size]
#         for i_batch in range(self.n_batch):
#             qid_combo = self.qid_list[i_batch]
#             classes = qid_combo['cat']
#             batch = []
#             for qid in qid_combo['question_ids']:

#                 batch.append(qid)
                
#             yield batch            
            
            
class CategoriesSampler7w():
    """The class to generate episodic data"""
    def __init__(self, label_dict, n_batch, K, N, Q):
        #K Way, N shot(train query), Q(test query)
        with open(label_dict, 'r') as fid:
            self.m_ind = json.load(fid)
        self.unique_labels = list(self.m_ind.keys())
#         print(self.m_ind.keys())
#         exit()
        self.K = K
        self.N = N
        self.Q = Q
        comb = combinations(self.unique_labels, K)
        self.label_combos = [c for c in comb]
        self.n_batch = n_batch
        print("NUM batches = {}".format(self.n_batch))
#         self.m_ind = {}
#         for i in self.unique_labels:
#             ind = np.argwhere(labeln == i).reshape(-1)
#             ind = torch.from_numpy(ind)
#             self.m_ind[i] = ind
        self.index = 0
    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i in range(self.n_batch):
            classes = self.label_combos[self.index]
            self.index = self.index % len(self.unique_labels)

            lr=[]
            dr=[]
            for c in classes:
#                 print(classes)
#                 exit()
                l = self.m_ind[c]
#                 pos = torch.randperm(len(l))[:(self.N +self.Q)]
                m= random.sample(l, self.N + self.Q)

                for i in range(0,self.N):
                    lr.append(m[i])
                    
                for i in range(self.N, (self.N +self.Q)):
                    dr.append(m[i])
                
            batch = lr + dr
#             batch=[]
#             for i in range(len(lr)):
#                 batch.append(lr[i])
            
#             for i in range(len(dr)):
#                 batch.append(dr[i])
                        
#             batch = torch.stack(batch, 0)    
                
            yield batch
        

samplers = Factory()
samplers.register_class("Train", CategoriesSampler)
samplers.register_class("Val", ValCategoriesSampler)
samplers.register_class("ValVQG", ValCategoriesSamplerVQG)
samplers.register_class("Visual7w", CategoriesSampler7w)