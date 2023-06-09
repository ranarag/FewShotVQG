"""Utility functions for training.
"""

import json
import torch
import torchtext

from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import nltk
import torch

# ===========================================================
# Vocabulary.
# ===========================================================

class Vocabulary(object):
    """Keeps track of all the words in the vocabulary.
    """

    # Reserved symbols
    SYM_PAD = '<pad>'    # padding.
    SYM_SOQ = '<start>'  # Start of question.
    SYM_SOR = '<resp>'   # Start of response.
    SYM_EOS = '<end>'    # End of sentence.
    SYM_UNK = '<unk>'    # Unknown word.

    def __init__(self):
        """Constructor for Vocabulary.
        """
        # Init mappings between words and ids
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.add_word(self.SYM_SOQ)
        self.add_word(self.SYM_EOS)
        self.add_word(self.SYM_UNK)
        self.add_word(self.SYM_PAD)        
        self.add_word(self.SYM_SOR)

    def __call__(self, word):
        return self.__getitem__(word)
        
    def add_word(self, word):
        """Adds a new word and updates the total number of unique words.

        Args:
            word: String representation of the word.
        """
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def remove_word(self, word):
        """
        Removes a specified word and updates the total number of unique words.

        Args:
            word: String representation of the word.
        """
        if word in self.word2idx:
            self.word2idx.pop(word)
            self.idx2word.pop(self.idx)
            self.idx -= 1

    def __getitem__(self, word):
        if word not in self.word2idx:
            return self.word2idx[self.SYM_UNK]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def save(self, location):
        with open(location, 'w') as f:
            f.write(json.dumps({'word2idx': self.word2idx,
                       'idx2word': self.idx2word,
                       'idx': self.idx}))

    def load(self, location):
        with open(location, 'r') as f:
            data = json.loads(f.read())
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.idx = data['idx']

    def tokens_to_words(self, tokens):
        """Converts tokens to vocab words.

        Args:
            tokens: 1D Tensor of Token outputs.

        Returns:
            A list of words.
        """
        words = []
        for token in tokens:
            try:
                word = self.idx2word[str(token)]
            except:
                word = self.idx2word[str(token.item())]
            if word == self.SYM_EOS:
                break
            if word not in [self.SYM_PAD, self.SYM_SOQ,
                            self.SYM_SOR, self.SYM_EOS]:
                words.append(word)
        sentence = str(' '.join(words))
        return sentence

    
    
def bert_text_preparation(text, tokenizer):
    """Preparing the input for BERT
    
    Takes a string argument and performs
    pre-processing like adding special tokens,
    tokenization, tokens to ids, and tokens to
    segment ids. All tokens are mapped to seg-
    ment id = 1.
    
    Args:
        text (str): Text to be converted
        tokenizer (obj): Tokenizer object
            to convert text into BERT-re-
            adable tokens and ids
        
    Returns:
        list: List of BERT-readable tokens
        obj: Torch tensor with token ids
        obj: Torch tensor segment ids
    
    
    """
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors
    
def get_bert_embed_list(tokens_tensor, segments_tensors, model):
    """Get embeddings from an embedding model
    
    Args:
        tokens_tensor (obj): Torch tensor size [n_tokens]
            with token ids for each token in text
        segments_tensors (obj): Torch tensor size [n_tokens]
            with segment ids for each token in text
        model (obj): Embedding model to generate embeddings
            from token and segment ids
    
    Returns:
        list: List of list of floats of size
            [n_tokens, n_embedding_dimensions]
            containing embeddings for each token
    
    """
    
    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs[2][1:]

    # Getting embeddings from the final BERT layer
    token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # Converting torchtensors to lists
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

    return list_token_embeddings    

from tqdm import tqdm
def get_bert_embedding(name, embed_size, vocab):
    vocab_size = len(vocab)
    model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True,
                                  )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    embedding = torch.zeros(vocab_size, embed_size)
    for i in tqdm(range(vocab_size)):
        text = vocab.idx2word[str(i)]
        target_embeds = []
        tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer)
#         tokens_tensor = tokens_tensor.cuda()
#         segments_tensors = segments_tensors.cuda()
        list_token_embeddings = get_bert_embed_list(tokens_tensor, segments_tensors, model)
        
        for token in tokenized_text[1:-1]:
            word_index = tokenized_text.index(token)
            word_embedding = list_token_embeddings[word_index]
            target_embeds.append(word_embedding)
        target_embeds = torch.tensor(target_embeds)
        target_embeds = torch.mean(target_embeds, 0)
        embedding[i] = target_embeds
#         print("OK")
#     for i in range(vocab_size):
#         embedding[i] = glove[vocab.idx2word[str(i)]]
    return embedding
def get_glove_embedding(name, embed_size, vocab):
    """Construct embedding tensor.

    Args:
        name (str): Which GloVe embedding to use.
        embed_size (int): Dimensionality of embeddings.
        vocab: Vocabulary to generate embeddings.
    Returns:
        embedding (vocab_size, embed_size): Tensor of
            GloVe word embeddings.
    """

    glove = torchtext.vocab.GloVe(name=name,
                                  dim=str(embed_size))
    vocab_size = len(vocab)
    embedding = torch.zeros(vocab_size, embed_size)
    for i in range(vocab_size):
        embedding[i] = glove[vocab.idx2word[str(i)]]
    return embedding


# ===========================================================
# Helpers.
# ===========================================================

def process_lengths(inputs, pad=0):
    """Calculates the lenght of all the sequences in inputs.

    Args:
        inputs: A batch of tensors containing the question or response
            sequences.

    Returns: A list of their lengths.
    """
    max_length = inputs.size(1)
    if inputs.size(0) == 1:
        lengths = list(max_length - inputs.data.eq(pad).sum(1))
    else:
        lengths = list(max_length - inputs.data.eq(pad).sum(1).squeeze())
    return lengths


# ===========================================================
# Evaluation metrics.
# ===========================================================

def gaussian_KL_loss(mus, logvars, eps=1e-8):
    """Calculates KL distance of mus and logvars from unit normal.

    Args:
        mus: Tensor of means predicted by the encoder.
        logvars: Tensor of log vars predicted by the encoder.

    Returns:
        KL loss between mus and logvars and the normal unit gaussian.
    """
    KLD = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp())
    kl_loss = KLD/(mus.size(0) + eps)
    """
    if kl_loss > 100:
        print kl_loss
        print KLD
        print mus.min(), mus.max()
        print logvars.min(), logvars.max()
        1/0
    """
    return kl_loss


def vae_loss(outputs, targets, mus, logvars, criterion):
    """VAE loss that combines cross entropy with KL divergence.

    Args:
        outputs: The predictions made by the model.
        targets: The ground truth indices in the vocabulary.
        mus: Tensor of means predicted by the encoder.
        logvars: Tensor of log vars predicted by the encoder.
        criterion: The cross entropy criterion.
    """
    CE = criterion(outputs, targets)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = gaussian_KL_loss(mus, logvars)
    return CE + KLD
