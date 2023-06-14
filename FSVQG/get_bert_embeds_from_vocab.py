from utils import load_vocab
from utils import get_glove_embedding, get_bert_embedding
from utils import Vocabulary
import argparse, json
import pickle as pkl

parser = argparse.ArgumentParser(description='Preprocess IQ VQA dataset.')

parser.add_argument('--vocab-path', type=str,
                    default='data/processed/vocab_iq.json',
                    help='Path for vocabulary wrapper.')
parser.add_argument('--vocab-path-ans', type=str,
                    default='data/processed/vocab_iq_ans.json',
                    help='Path for vocabulary wrapper.')

parser.add_argument('--bert-ans-embed', type=str, default='data/processed/bert_embedding_ans.pkl',
                    help='Path for saving bert embedding of answer.')

parser.add_argument('--bert-cat-embed', type=str, default='data/processed/bert_embedding.pkl',
                    help='Path for saving bert embedding of category.')

args = parser.parse_args()


word_dict = load_vocab(args.vocab_path)
vocabulary_size = len(word_dict)
ans_dict = load_vocab(args.vocab_path_ans)
ans_size = len(ans_dict)
ans_embedding = get_bert_embedding('840B',
                                    768,
                                    ans_dict)

with open(args.bert_ans_embed, 'wb') as fid:
    pkl.dump(ans_embedding, fid)

print('Saved bert embedding of answer to %s' % args.bert_ans_embed)

embedding = get_bert_embedding('840B',
                                768,
                                word_dict)

with open(args.bert_cat_embed, 'wb') as fid:
    pkl.dump(embedding, fid)

print('Saved bert embedding of category to %s' % args.bert_cat_embed)