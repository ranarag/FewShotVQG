import torch
import torch.nn as nn
from .BahdanauAttention import Attention2, Attention
import torch.nn.functional as F

class LSTM_Decoder(nn.Module):
    def __init__(self, vocabulary_size, encoder_dim, decoder_dim=2048, side_embed_dim=None, tf=False, embedding=None, scale_shift=True):
        super(LSTM_Decoder, self).__init__()
        self.use_tf = tf

        self.vocabulary_size = vocabulary_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.scale_shift = scale_shift
        
#         print(encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.tanh = nn.Tanh()

#         self.attention = Attention(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.deep_output = nn.Linear(decoder_dim, vocabulary_size)
        self.dropout = nn.Dropout()
        self.embedding = nn.Embedding(vocabulary_size, decoder_dim)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding, requires_grad=False)
        self.lstm = nn.LSTMCell(decoder_dim + encoder_dim, decoder_dim)
        
#         self.attention = Attention2(encoder_dim, 80, 4, decoder_dim)
        
    

        self.attention = Attention(encoder_dim, decoder_dim)
        if self.scale_shift:
            self.lstm = nn.LSTMCell(decoder_dim + encoder_dim, decoder_dim)
        else:
            if side_embed_dim is None:
                side_embed_dim = decoder_dim
            self.lstm = nn.LSTMCell(decoder_dim + encoder_dim + side_embed_dim, decoder_dim)

#         if self.scale_shift:
#             self.attention = Attention(encoder_dim, decoder_dim)
#         else:
#             if side_embed_dim is None:
#                 side_embed_dim = decoder_dim
#             self.attention = Attention(encoder_dim, decoder_dim, side_embed_dim)
        self.side_embed_dim = side_embed_dim
    def forward(self, img_features, captions, side_embeds=None):
        """
        We can use teacher forcing during training. For reference, refer to
        https://www.deeplearningbook.org/contents/rnn.html

        """
        batch_size = img_features.size(0)
#         print(side_embeds.size())
#         print(img_features.size())
        h, c = self.get_init_lstm_state(img_features)
        max_timespan = max([len(caption) for caption in captions]) - 1

        prev_words = torch.zeros(batch_size, 1).long().cuda()
        if self.use_tf:
            embedding = self.embedding(captions) if self.training else self.embedding(prev_words)
        else:
            embedding = self.embedding(prev_words)

        preds = torch.zeros(batch_size, max_timespan, self.vocabulary_size).cuda()
        alphas = torch.zeros(batch_size, max_timespan, 49).cuda()
        for t in range(max_timespan):
#             if self.scale_shift:
#                 enc = h
#             else:
# #                 print(side_embeds.size())
#                 enc = torch.cat([h, side_embeds], dim=1)
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context
#             print(context.size())
            
            if not self.scale_shift:
# #                 print("TRIGGERED")
                gated_context = torch.cat([gated_context, side_embeds], dim=1)
#                 print(gated_context.size())
                
            if self.use_tf and self.training:
                lstm_input = torch.cat((embedding[:, t], gated_context), dim=1)
            else:
                embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding
                lstm_input = torch.cat((embedding, gated_context), dim=1)
#             print(h.size(), c.size(), lstm_input.size())
            h, c = self.lstm(lstm_input, (h, c))
#             exit()
            output = self.deep_output(self.dropout(h))
            # output = F.log_softmax(output, dim=1)
            preds[:, t] = output
            alphas[:, t] = alpha

            if not self.training or not self.use_tf:
                embedding = self.embedding(output.max(1)[1].reshape(batch_size, 1))
        return preds, alphas

    def get_init_lstm_state(self, img_features):
#         n_img_features = img_features.view(img_features.size(0), img_features.size(1), -1)
#         n_img_features = n_img_features.permute(0, 2, 1)
        avg_features = img_features.mean(dim=1)

        c = self.init_c(avg_features)
        c = self.tanh(c)

        h = self.init_h(avg_features)
        h = self.tanh(h)

        return h, c
 
    def caption(self, img_features, word_map, beam_size, side_embeds=None):
        """
        We use beam search to construct the best sentences following a
        similar implementation as the author in
        https://github.com/kelvinxu/arctic-captions/blob/master/generate_caps.py
        """
        # prev_words = torch.zeros(beam_size, 1).long().cuda()
#         print(word_map.keys())
        prev_words = torch.LongTensor([[word_map['<start>']]] * beam_size).cuda()  # (k, 1)
        # print(prev_words.size())
        sentences = prev_words
        top_preds = torch.zeros(beam_size, 1).cuda()
        alphas = torch.ones(beam_size, 1, img_features.size(1)).cuda()
        # print(word_map['<start>'], word_map['<end>'])
        completed_sentences = []
        completed_sentences_alphas = []
        completed_sentences_preds = []

        step = 1
        h, c = self.get_init_lstm_state(img_features)

        while step <= 20:
            embedding = self.embedding(prev_words).squeeze(1)
#             if self.scale_shift:
#                 enc = h
#             else:
#                 n_side_embeds = side_embeds[:beam_size, :]
#                 enc = torch.cat([h, n_side_embeds], dim=1)
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context
            if not self.scale_shift:
#                 if gated_context.size(0) != side_embeds.size(0):
#                     print(gated_context.size(), side_embeds.size())
#                 print(gated_context.size(), side_embeds.size())
                n_side_embeds = side_embeds[:beam_size, :]
#                 print(n_side_embeds.size())
                gated_context = torch.cat([gated_context, n_side_embeds], dim=1)
            
            lstm_input = torch.cat((embedding, gated_context), dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(h)
            output = F.log_softmax(output, dim=1)
            # print(output.size())
            # exit()
            output = top_preds.expand_as(output) + output

            if step == 1:
                top_preds, top_words = output[0].topk(beam_size, 0, True, True)
            else:
                top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True)
            # print(top_words.size())
            prev_word_idxs = torch.floor_divide(top_words, output.size(1))
            next_word_idxs = top_words % output.size(1)
            # print(next_word_idxs)
            # exit()
            sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
            alphas = torch.cat((alphas[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)

            incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != word_map['<end>']]
            complete = list(set(range(len(next_word_idxs))) - set(incomplete))
            # print(complete)
            if len(complete) > 0:
#                 print(len(complete))
                completed_sentences.extend(sentences[complete].tolist())
                completed_sentences_alphas.extend(alphas[complete].tolist())
                completed_sentences_preds.extend(top_preds[complete])
            beam_size -= len(complete)

            if beam_size == 0:
                break
            sentences = sentences[incomplete]
            alphas = alphas[incomplete]
            h = h[prev_word_idxs[incomplete]]
            c = c[prev_word_idxs[incomplete]]
            img_features = img_features[prev_word_idxs[incomplete]]
            top_preds = top_preds[incomplete].unsqueeze(1)
            prev_words = next_word_idxs[incomplete].unsqueeze(1)

            # if step > 50:
            #     break
            step += 1
        if len(completed_sentences):
            idx = completed_sentences_preds.index(max(completed_sentences_preds))
            sentence = completed_sentences[idx]
            alpha = completed_sentences_alphas[idx]
        else:
            sentence = [word_map['<start>']*20]
            alpha[word_map['<start>']*20]
        return sentence, alpha
