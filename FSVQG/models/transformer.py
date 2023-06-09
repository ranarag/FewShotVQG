import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import math
from torch.autograd import Variable
from transformers import GPT2LMHeadModel, GPT2Config
#-----taken from https://raw.githubusercontent.com/RoyalSkye/Image-Caption/master/transformer.py --------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# channel_number = 2048
MAX_QLEN = 20

class ScaledDotProductAttention(nn.Module):
    def __init__(self, QKVdim):
        super(ScaledDotProductAttention, self).__init__()
        self.QKVdim = QKVdim

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch_size, n_heads, -1(len_q), QKVdim]
        :param K, V: [batch_size, n_heads, -1(len_k=len_v), QKVdim]
        :param attn_mask: [batch_size, n_heads, len_q, len_k]
        """
        # scores: [batch_size, n_heads, len_q, len_k]
#         print("ATTN mask = ", attn_mask.size())
#         print("Q size = ", Q.size())
#         print("K size = ", K.size())
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.QKVdim)
        # Fills elements of self tensor with value where mask is True.
        scores.to(device).masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_heads, len_q, len_k]
        context = torch.matmul(attn, V).to(device)  # [batch_size, n_heads, len_q, QKVdim]
        return context, attn


class Attention(nn.Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x    
    
class Multi_Head_Attention(nn.Module):
    def __init__(self, Q_dim, K_dim, QKVdim, n_heads=8, dropout=0.1):
        super(Multi_Head_Attention, self).__init__()
        self.W_Q = nn.Linear(Q_dim, QKVdim * n_heads).to(device)
        self.W_K = nn.Linear(K_dim, QKVdim * n_heads).to(device)
        self.W_V = nn.Linear(K_dim, QKVdim * n_heads).to(device)
        # print(Q_dim, QKVdim, K_dim)
        self.n_heads = n_heads
        self.QKVdim = QKVdim
        self.embed_dim = Q_dim
        self.dropout = nn.Dropout(p=dropout)
        self.W_O = nn.Linear(self.n_heads * self.QKVdim, self.embed_dim).to(device)

    def forward(self, Q, K, V, attn_mask):
        """
        In self-encoder attention:
                Q = K = V: [batch_size, num_pixels=49, encoder_dim=2048]
                attn_mask: [batch_size, len_q=49, len_k=49]
        In self-decoder attention:
                Q = K = V: [batch_size, max_len=MAX_QLEN, embed_dim=512]
                attn_mask: [batch_size, len_q=MAX_QLEN, len_k=MAX_QLEN]
        encoder-decoder attention:
                Q: [batch_size, MAX_QLEN, 512] from decoder
                K, V: [batch_size, 49, 2048] from encoder
                attn_mask: [batch_size, len_q=MAX_QLEN, len_k=49]
        return _, attn: [batch_size, n_heads, len_q, len_k]
        """
        
        residual, batch_size = Q, Q.size(0)
        # q_s: [batch_size, n_heads=8, len_q, QKVdim] k_s/v_s: [batch_size, n_heads=8, len_k, QKVdim]
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        # attn_mask: [batch_size, self.n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # attn: [batch_size, n_heads, len_q, len_k]
        # context: [batch_size, n_heads, len_q, QKVdim]
        context, attn = ScaledDotProductAttention(self.QKVdim)(q_s, k_s, v_s, attn_mask)
        # context: [batch_size, n_heads, len_q, QKVdim] -> [batch_size, len_q, n_heads * QKVdim]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.QKVdim).to(device)
        # output: [batch_size, len_q, embed_dim]
        output = self.W_O(context)
        output = self.dropout(output)
        return nn.LayerNorm(self.embed_dim).to(device)(output + residual), attn  # Need to change this


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, embed_dim, d_ff, dropout):
        super(PoswiseFeedForwardNet, self).__init__()
        """
        Two fc layers can also be described by two cnn with kernel_size=1.
        """
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=d_ff, kernel_size=1).to(device)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=embed_dim, kernel_size=1).to(device)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dim = embed_dim

    def forward(self, inputs):
        """
        encoder: inputs: [batch_size, len_q=49, embed_dim=2048]
        decoder: inputs: [batch_size, max_len=MAX_QLEN, embed_dim=512]
        """
        residual = inputs
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        return nn.LayerNorm(self.embed_dim).to(device)(output + residual)


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, side_info_dim, side_info_embed_dim, dropout, attention_method, n_heads, n_channels=1024, att_type='q'):
        super(DecoderLayer, self).__init__()
        self.attention_method = attention_method
        self.n_channels = n_channels
        self.att_type = att_type
        self.dec_self_attn_side_info_enc = nn.Linear(side_info_dim, side_info_embed_dim).to(device)
        self.dec_self_conv = nn.Conv1d(1, 20, 1)
        tot_dim = embed_dim + side_info_embed_dim
        self.dec_self_q_resize = nn.Linear(tot_dim, embed_dim)
        self.dec_enc_attn_side_info_enc = nn.Linear(side_info_dim, side_info_embed_dim).to(device)
        
        if self.att_type == 'q':
            
            self.dec_enc_conv = nn.Conv1d(1, 20, 1)
            self.dec_enc_q_resize = nn.Linear(tot_dim, embed_dim)
            self.forward = self.forward_q
        else:
            self.dec_enc_conv = nn.Conv1d(1, 49, 1)
            self.dec_enc_q_resize = nn.Linear(side_info_embed_dim + self.n_channels , self.n_channels)
            if self.att_type == 'k':
                self.forward = self.forward_k
            elif self.att_type == 'kv':
                self.forward = self.forward_kv
            else:
                print("Attention Type {} does not exist.".format(self.att_type))

        self.dec_self_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=embed_dim, QKVdim=16, n_heads=n_heads, dropout=dropout)
        
        if attention_method == "ByPixel":
            self.dec_enc_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=n_channels, QKVdim=16, n_heads=n_heads, dropout=dropout)
            self.pos_ffn = PoswiseFeedForwardNet(embed_dim=embed_dim, d_ff=n_channels, dropout=dropout)
        elif attention_method == "ByChannel":
            self.dec_enc_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=49, QKVdim=16, n_heads=n_heads, dropout=dropout)
            self.pos_ffn = PoswiseFeedForwardNet(embed_dim=embed_dim, d_ff=n_channels, dropout=dropout)  # need to change

            
    def forward_q(self, dec_inputs, enc_outputs, side_info, dec_self_attn_mask, dec_enc_attn_mask):
        """
        :param dec_inputs: [batch_size, max_len=MAX_QLEN, embed_dim=512]
        :param enc_outputs: [batch_size, num_pixels=49, 2048]
        :param dec_self_attn_mask: [batch_size, MAX_QLEN, MAX_QLEN]
        :param dec_enc_attn_mask: [batch_size, MAX_QLEN, 49]
        """
        self_attn_side_info = self.dec_self_attn_side_info_enc(side_info).unsqueeze(1)
        self_attn_side_info = self.dec_self_conv(self_attn_side_info)

        self_att_q = torch.cat([dec_inputs, self_attn_side_info], dim=2)
        self_att_q = self.dec_self_q_resize(self_att_q)

        dec_outputs, dec_self_attn = self.dec_self_attn(self_att_q, dec_inputs,  dec_inputs, dec_self_attn_mask)

        cross_attn_side_info = self.dec_enc_attn_side_info_enc(side_info).unsqueeze(1)
        cross_attn_side_info = self.dec_enc_conv(cross_attn_side_info)
        
        cross_att_q = torch.cat([dec_outputs, cross_attn_side_info], dim=2)
        cross_att_q = self.dec_enc_q_resize(cross_att_q)

        if self.attention_method == "ByChannel":
            enc_outputs = enc_outputs.permute(0, 2, 1).contiguous()
#             cross_att_q = cross_att_q.permute(0, 2, 1).contiguous()
        dec_outputs, dec_enc_attn = self.dec_enc_attn(cross_att_q, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

    def forward_k(self, dec_inputs, enc_outputs, side_info, dec_self_attn_mask, dec_enc_attn_mask):
        """
        :param dec_inputs: [batch_size, max_len=MAX_QLEN, embed_dim=512]
        :param enc_outputs: [batch_size, num_pixels=49, 2048]
        :param dec_self_attn_mask: [batch_size, MAX_QLEN, MAX_QLEN]
        :param dec_enc_attn_mask: [batch_size, MAX_QLEN, 49]
        """
        self_attn_side_info = self.dec_self_attn_side_info_enc(side_info).unsqueeze(1)
        self_attn_side_info = self.dec_self_conv(self_attn_side_info)

        self_att_k = torch.cat([dec_inputs, self_attn_side_info], dim=2)
        self_att_k = self.dec_self_q_resize(self_att_k)

        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, self_att_k,  dec_inputs, dec_self_attn_mask)
        cross_attn_side_info = self.dec_enc_attn_side_info_enc(side_info).unsqueeze(1)
        cross_attn_side_info = self.dec_enc_conv(cross_attn_side_info)

        cross_att_k = torch.cat([enc_outputs, cross_attn_side_info], dim=2)
        cross_att_k = self.dec_enc_q_resize(cross_att_k)        

        if self.attention_method == "ByChannel":
            enc_outputs = enc_outputs.permute(0, 2, 1).contiguous()
            cross_att_k = cross_att_k.permute(0, 2, 1).contiguous()
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, cross_att_k, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn
    
    def forward_kv(self, dec_inputs, enc_outputs, side_info, dec_self_attn_mask, dec_enc_attn_mask):
        """
        :param dec_inputs: [batch_size, max_len=MAX_QLEN, embed_dim=512]
        :param enc_outputs: [batch_size, num_pixels=49, 2048]
        :param dec_self_attn_mask: [batch_size, MAX_QLEN, MAX_QLEN]
        :param dec_enc_attn_mask: [batch_size, MAX_QLEN, 49]
        """
        self_attn_side_info = self.dec_self_attn_side_info_enc(side_info).unsqueeze(1)
        self_attn_side_info = self.dec_self_conv(self_attn_side_info)

        self_att_kv = torch.cat([dec_inputs, self_attn_side_info], dim=2)
        self_att_kv = self.dec_self_q_resize(self_att_kv)

        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, self_att_kv,  self_att_kv, dec_self_attn_mask)
        cross_attn_side_info = self.dec_enc_attn_side_info_enc(side_info).unsqueeze(1)
        cross_attn_side_info = self.dec_enc_conv(cross_attn_side_info)

        cross_att_kv = torch.cat([enc_outputs, cross_attn_side_info], dim=2)
        cross_att_kv = self.dec_enc_q_resize(cross_att_kv)        

        if self.attention_method == "ByChannel":
            enc_outputs = enc_outputs.permute(0, 2, 1).contiguous()
            cross_att_kv = cross_att_kv.permute(0, 2, 1).contiguous()
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, cross_att_kv, cross_att_kv, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, n_layers, vocab_size, embed_dim, side_info_dim, \
                 side_info_embed_dim, dropout, attention_method, n_heads, \
                 embedding, n_input_channels, n_hidden_channels, att_type):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.channel_number = n_hidden_channels
        self.tgt_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
#         print(embed_dim)
        if embedding is not None:
            self.tgt_emb.weight = nn.Parameter(embedding, requires_grad=False)
        self.pos_emb = nn.Embedding.from_pretrained(self.get_position_embedding_table(embed_dim), freeze=True)
        self.pos_emb.weight.requires_grad = False
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, side_info_dim, side_info_embed_dim, \
                                                  dropout, attention_method, n_heads, n_channels=n_hidden_channels, att_type=att_type) for _ in range(n_layers)])
        self.projection = nn.Linear(embed_dim, vocab_size, bias=False).to(device)
        self.attention_method = attention_method
        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=n_hidden_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   pooling_kernel_size=3,
                                   pooling_stride=1,
                                   pooling_padding=1,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=1,
                                   conv_bias=False)

    def get_position_embedding_table(self, embed_dim):
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / embed_dim)
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_idx) for hid_idx in range(embed_dim)]

        embedding_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(MAX_QLEN)])
        embedding_table[:, 0::2] = np.sin(embedding_table[:, 0::2])  # dim 2i
        embedding_table[:, 1::2] = np.cos(embedding_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(embedding_table).to(device)

    def get_attn_pad_mask(self, seq_q, seq_k):
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # In wordmap, <pad>:0
        # pad_attn_mask: [batch_size, 1, len_k], one is masking
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
        pad_attn_mask[:, 0, 0] = False
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

    def get_attn_subsequent_mask(self, seq):
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        subsequent_mask = np.triu(np.ones(attn_shape), k=1)
        subsequent_mask = torch.from_numpy(subsequent_mask).byte().to(device)
        return subsequent_mask

    def forward(self, encoder_out, side_info, encoded_captions):
        """
        :param encoder_out: [batch_size, num_pixels=49, 2048]
        :param encoded_captions: [batch_size, MAX_QLEN]
        :param caption_lengths: [batch_size, 1]
        """
        batch_size = encoder_out.size(0)
        encoder_out = self.tokenizer(encoder_out)
        # Sort input data by decreasing lengths.
        # caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        # encoder_out = encoder_out[sort_ind]
        # encoded_captions = encoded_captions[sort_ind]
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        # decode_lengths = (caption_lengths - 1).tolist()

        # dec_outputs: [batch_size, max_len=MAX_QLEN, embed_dim=512]
        # dec_self_attn_pad_mask: [batch_size, len_q=MAX_QLEN, len_k=MAX_QLEN], 1 if id=0(<pad>)
        # dec_self_attn_subsequent_mask: [batch_size, MAX_QLEN, MAX_QLEN], Upper triangle of an array with 1.
        # dec_self_attn_mask for self-decoder attention, the position whose val > 0 will be masked.
        # dec_enc_attn_mask for encoder-decoder attention.
        # e.g. 9488, 23, 53, 74, 0, 0  |  dec_self_attn_mask:
        # 0 1 1 1 2 2
        # 0 0 1 1 2 2
        # 0 0 0 1 2 2
        # 0 0 0 0 2 2
        # 0 0 0 0 1 2
        # 0 0 0 0 1 1
#         print(encoder_out.size())
#         print(encoded_captions.size())
        dec_outputs = self.tgt_emb(encoded_captions) + self.pos_emb(torch.LongTensor([list(range(MAX_QLEN))]*batch_size).to(device))
        dec_outputs = self.dropout(dec_outputs)
        dec_self_attn_pad_mask = self.get_attn_pad_mask(encoded_captions, encoded_captions)
        dec_self_attn_subsequent_mask = self.get_attn_subsequent_mask(encoded_captions)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        if self.attention_method == "ByPixel":
            dec_enc_attn_mask = (torch.tensor(np.zeros((batch_size, MAX_QLEN, 49))).to(device) == torch.tensor(np.ones((batch_size, MAX_QLEN, 49))).to(device))
        elif self.attention_method == "ByChannel":
            dec_enc_attn_mask = (torch.tensor(np.zeros((batch_size, MAX_QLEN, self.channel_number))).to(device) == torch.tensor(np.ones((batch_size, MAX_QLEN, self.channel_number))).to(device))

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # attn: [batch_size, n_heads, len_q, len_k]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, encoder_out, side_info, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        predictions = self.projection(dec_outputs)
        alphas = {"dec_self_attns": dec_self_attns, "dec_enc_attns": dec_enc_attns}
        return predictions, alphas

    def caption(self, img_features, side_info, word_map, beam_size):
        """
        We use beam search to construct the best sentences following a
        similar implementation as the author in
        https://github.com/kelvinxu/arctic-captions/blob/master/generate_caps.py
        """
         
        enc_image_size = 7
        prev_words = torch.LongTensor([[word_map['<start>']] * MAX_QLEN] * beam_size).cuda()  # (k, 1)
        # print(prev_words.size())
        sentences = torch.LongTensor([[word_map['<start>']]] * beam_size).to(device)
        top_preds = torch.zeros(beam_size, 1).cuda()
        alphas = torch.ones(beam_size, 1, enc_image_size, enc_image_size).cuda()
        # print(word_map['<start>'], word_map['<end>'])
        completed_sentences = []
        completed_sentences_alphas = []
        completed_sentences_preds = []

        step = 1

        while step < 20:
#             print(img_features.size(), side_info.size(), prev_words.size())
            output, alpha_dict = self.forward(img_features, side_info[:beam_size,], prev_words)
            output = output[:, step - 1, :].squeeze(1)
            # alpha = alpha_dict["dec_enc_attns"][-1]            
            # alpha = alpha[:, 0, step-1, :].view(beam_size, 1, enc_image_size, enc_image_size) 
            
            output = F.log_softmax(output, dim=1)

            output = top_preds.expand_as(output) + output

            if step == 1:
                top_preds, top_words = output[0].topk(beam_size, 0, True, True)
            else:
                top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True)
            prev_word_idxs = torch.floor_divide(top_words, output.size(1))
            next_word_idxs = top_words % output.size(1)
            sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
            # alphas = torch.cat([alphas[prev_word_idxs], alpha[prev_word_idxs]], dim=1)

            incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != word_map['<end>']]
            complete = list(set(range(len(next_word_idxs))) - set(incomplete))
            if len(complete) > 0:
                completed_sentences.extend(sentences[complete].tolist())
                # completed_sentences_alphas.extend(alphas[complete].tolist())
                completed_sentences_preds.extend(top_preds[complete])
            beam_size -= len(complete)

            if beam_size == 0:
                break
            sentences = sentences[incomplete]
            # alphas = alphas[incomplete]
            img_features = img_features[prev_word_idxs[incomplete]]
            top_preds = top_preds[incomplete].unsqueeze(1)
            prev_words = prev_words[incomplete]
            prev_words[:, :step + 1] = sentences


            step += 1
        alpha = []
        if len(completed_sentences):
            idx = completed_sentences_preds.index(max(completed_sentences_preds))
            sentence = completed_sentences[idx]
            # alpha = completed_sentences_alphas[idx]
        else:
            sentence = [word_map['<start>']*20]
            # alpha[word_map['<start>']*20]
        return sentence, alpha


#------ Encoder specific things-------------------#
# taken from https://blog.floydhub.com/the-transformer-in-pytorch/ #
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and iEmbedder
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], \
        requires_grad=False).cuda()
        return x
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff).to(device)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model).to(device)
        self.norm = nn.LayerNorm(d_model).to(device)
    def forward(self, x):
        residual = x
        x_out = self.dropout(F.relu(self.linear_1(x)))
        x_out = self.linear_2(x_out)
        x_out = self.dropout(x_out)
        return self.norm(x_out + residual)
    


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout = 0.1):
        super().__init__()
        self.Q_dim = d_model
        self.K_dim = d_model
        self.QKVdim = d_model // n_heads
        self.attn = Multi_Head_Attention(Q_dim=self.Q_dim, K_dim=self.K_dim, 
                                         QKVdim=self.QKVdim, n_heads=n_heads, 
                                         dropout=dropout)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, enc_inputs, enc_self_attn_mask):
       enc_outputs, attn = self.attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
       enc_outputs = self.ff(enc_outputs)
       return enc_outputs, attn  
          
    # def forward(self, x, mask):
    #     x2 = self.norm_1(x)
    #     x = self.attn(x2,x2,x2,mask)
    #     x2 = self.norm_2(x)
    #     x = x + self.dropout_2(self.ff(x2))
    #     return x
    


class Encoder(nn.Module):
    def __init__(self, n_layers, vocab_size, embed_dim, dropout, n_heads, embedding=None):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding is not None:
            self.embed.weight = nn.Parameter(embedding, requires_grad=False)
        self.pe = PositionalEncoder(embed_dim)
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, n_heads, dropout) for _ in range(n_layers)])
        
    def get_attn_pad_mask(self, seq_q, seq_k):
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # In wordmap, <pad>:0
        # pad_attn_mask: [batch_size, 1, len_k], one is masking
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

    def get_attn_subsequent_mask(self, seq):
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        subsequent_mask = np.triu(np.ones(attn_shape), k=1)
        subsequent_mask = torch.from_numpy(subsequent_mask).byte().to(device)
        return subsequent_mask

    def forward(self, src, seq_len):
        x = self.embed(src)
        x = self.pe(x)
        enc_self_attns = []
        dec_self_attn_pad_mask = self.get_attn_pad_mask(src, src)
        dec_self_attn_subsequent_mask = self.get_attn_subsequent_mask(src)
        mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        for layer in self.layers:
            x, enc_self_attn = layer(x, mask)
            enc_self_attns.append(enc_self_attn)
        return enc_self_attns, x 
    
    
    ############# GPT 2 Model ###############
class GPT2Decoder(nn.Module):
    def __init__(self):
        super(GPT2Decoder, self).__init__()
        config = GPT2Config(add_cross_attention=True)
        self.model = GPT2LMHeadModel(config)
        self.model.from_pretrained('/scratch/18cs92p02/david_workspace/p_dataset/GPT2/gpt2_model_files/')
        device_map = {0: [0, 1, 2, 3, 4, 5], 1: [6, 7, 8, 9, 10, 11]}
        self.model.parallelize(device_map)
        self.resizer = nn.Linear(49*1536, 20*768)
        # self.model.config.add_cross_attention = True

    def get_position_embedding_table(self, embed_dim):
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / embed_dim)
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_idx) for hid_idx in range(embed_dim)]

        embedding_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(MAX_QLEN)])
        embedding_table[:, 0::2] = np.sin(embedding_table[:, 0::2])  # dim 2i
        embedding_table[:, 1::2] = np.cos(embedding_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(embedding_table).to(device)

    def get_attn_pad_mask(self, seq_q, seq_k):
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # In wordmap, <pad>:0
        # pad_attn_mask: [batch_size, 1, len_k], one is masking
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

    def get_attn_subsequent_mask(self, seq):
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        subsequent_mask = np.triu(np.ones(attn_shape), k=1)
        subsequent_mask = torch.from_numpy(subsequent_mask).byte().to(device)
        return subsequent_mask

    def forward(self, encoder_out, encoded_captions, is_Train=True):
        """
        :param encoder_out: [batch_size, num_pixels=49, 2048]
        :param encoded_captions: [batch_size, MAX_QLEN]
        :param caption_lengths: [batch_size, 1]
        """
        batch_size = encoder_out.size(0)
        print(encoder_out.size())
        encoder_out = torch.flatten(encoder_out, start_dim=1)
        print(encoder_out.size())
        encoder_out = self.resizer(encoder_out).view(batch_size, 20, 768)
        # exit()
        if is_Train:
            outputs = self.model(input_ids = encoded_captions, encoder_hidden_states=encoder_out, labels=encoded_captions)
        else:
            outputs = self.model(input_ids = encoded_captions, encoder_hidden_states=encoder_out)
        return outputs.logits, outputs.loss
        
    def caption(self, img_features, word_map, beam_size):
        """
        We use beam search to construct the best sentences following a
        similar implementation as the author in
        https://github.com/kelvinxu/arctic-captions/blob/master/generate_caps.py
        """
         
        enc_image_size = 7
        prev_words = torch.LongTensor([[word_map['<start>']] * MAX_QLEN] * beam_size).cuda()  # (k, 1)
        # print(prev_words.size())
        sentences = torch.LongTensor([[word_map['<start>']]] * beam_size).to(device)
        top_preds = torch.zeros(beam_size, 1).cuda()
        alphas = torch.ones(beam_size, 1, enc_image_size, enc_image_size).cuda()
        # print(word_map['<start>'], word_map['<end>'])
        completed_sentences = []
        completed_sentences_alphas = []
        completed_sentences_preds = []

        step = 1

        while step < 20:
            
            output, _ = self.model(img_features, prev_words, is_Train=False)
            output = output[:, step - 1, :].squeeze(1)
            # alpha = alpha_dict["dec_enc_attns"][-1]            
            # alpha = alpha[:, 0, step-1, :].view(beam_size, 1, enc_image_size, enc_image_size) 
            
            output = F.log_softmax(output, dim=1)

            output = top_preds.expand_as(output) + output

            if step == 1:
                top_preds, top_words = output[0].topk(beam_size, 0, True, True)
            else:
                top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True)
            prev_word_idxs = torch.floor_divide(top_words, output.size(1))
            next_word_idxs = top_words % output.size(1)
            sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
            # alphas = torch.cat([alphas[prev_word_idxs], alpha[prev_word_idxs]], dim=1)

            incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != word_map['<end>']]
            complete = list(set(range(len(next_word_idxs))) - set(incomplete))
            if len(complete) > 0:
                completed_sentences.extend(sentences[complete].tolist())
                # completed_sentences_alphas.extend(alphas[complete].tolist())
                completed_sentences_preds.extend(top_preds[complete])
            beam_size -= len(complete)

            if beam_size == 0:
                break
            sentences = sentences[incomplete]
            # alphas = alphas[incomplete]
            img_features = img_features[prev_word_idxs[incomplete]]
            top_preds = top_preds[incomplete].unsqueeze(1)
            prev_words = prev_words[incomplete]
            prev_words[:, :step + 1] = sentences


            step += 1
        alpha = []
        if len(completed_sentences):
            idx = completed_sentences_preds.index(max(completed_sentences_preds))
            sentence = completed_sentences[idx]
            # alpha = completed_sentences_alphas[idx]
        else:
            sentence = [word_map['<start>']*20]
            # alpha[word_map['<start>']*20]
        return sentence, alpha