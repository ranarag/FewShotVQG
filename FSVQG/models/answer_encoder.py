import torch
import torch.nn as nn
from .encoder_rnn import EncoderRNN
from .transformer import Encoder
class AnswerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encoder_dim, encoder_type, embedding=None, n_layers=None, n_heads=None, scale_shift=True):
        super(AnswerEncoder, self).__init__()
#         print(num_categories)
#         print(embedding_dim)
        self.encoder_type = encoder_type
        self.scale_shift = scale_shift
        print(self.scale_shift, encoder_type)
        if encoder_type == 'lstm':
            self.answer_encoder = EncoderRNN(vocab_size, 4, embedding_dim,
                                            input_dropout_p=0,
                                            dropout_p=0,
                                            n_layers=0,
                                            bidirectional=False,
                                            rnn_cell="LSTM",
                                            variable_lengths=True, embedding=embedding)
        else:
            self.answer_encoder = Encoder(n_layers, vocab_size, embedding_dim, 0.1, n_heads, embedding)
        if self.scale_shift:
            self.cat_wts = nn.Linear(embedding_dim, encoder_dim)
            self.cat_biases = nn.Linear(embedding_dim, encoder_dim)
#         else:
#             self.out_embed = nn.Linear(embedding_dim, encoder_dim)




    def forward(self, answers, alengths, weights=None):
        """Encode answers.
        Args:
            answers: Batch of answer Tensors.
        Returns:
            Batch of answers encoded into features.
        """
#         print(answers.size())
#         batch_size = answers.size(0)
#         embedding_answers = self.embedding(answers).mean(1)
        _, embedding_answers = self.answer_encoder(answers, alengths)
        
        if self.encoder_type == 'transformer':
            embedding_answers = embedding_answers.mean(1)
        if self.scale_shift:
            cat_wt_vals = self.cat_wts(embedding_answers)
            cat_b_vals = self.cat_biases(embedding_answers)
            return cat_wt_vals, cat_b_vals
#         else:
#             return self.out_embed(embedding_answers)
        else:
            return embedding_answers
        # return encoded_vals
 
