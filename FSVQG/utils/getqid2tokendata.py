from train_utils import Vocabulary
from vocab7w import load_vocab
from vocab7w import process_text
import json
import pickle as pkl
import numpy as np
from tqdm import tqdm
with open('qid2data_7w.json', 'r') as fid:
    dataset = json.load(fid)

categories = ["where", "what", "how", "when", "who", "why"]
qid2toks = {}
vocab = load_vocab('data/processed/vocab_7w.json')
ansvocab = load_vocab('vocab_7w_ans.json')
for k, entry in tqdm(dataset.items()):
#     print(type(k))
#     exit()
    cat = categories.index(entry['cat'])
    if cat > 2:
        continue
    npques = np.zeros(20, dtype=int)
    question = entry['question'].lower()
    qtoks, qlength = process_text(question, vocab, 20)
    npques[:qlength] = qtoks
    npans = np.zeros(4, dtype=int)
    answer = entry['answer']
    atoks, alength = process_text(answer, ansvocab, 4)
    npans[:alength] = atoks

    qid2toks[k] = {'image_id': entry['image_id'], 'q_embed':npques, 'a_embed': npans, 'category': cat}

    
with open('qid2toks7w_train.pkl', 'wb') as fid:
    pkl.dump(qid2toks, fid)
    
