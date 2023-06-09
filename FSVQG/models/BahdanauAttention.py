import torch.nn as nn
import torch.nn.functional as F
import torch
class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""
    
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()
        
        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 32, bias=False)
        
        # to store attention scores
        self.alphas = None
        
    def forward(self, query=None, proj_key=None, value=None, mask=None):
#         assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)
        
        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
#         print scores.size()
#         scores = scores.squeeze(2)
#         print(value.size())
        value = value.view(-1, 32, 169)
        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
#         scores.data.masked_fill_(mask == 0, -float('inf'))
        
        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas        
        
        # The context vector is the weighted sum of the values.
        context = (value * alphas.unsqueeze(2)).sum(1)
        
        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas

    
class Attention(nn.Module):
    def __init__(self, encoder_dim, hidden_dim, side_embed_dim=None):
        super(Attention, self).__init__()
        if side_embed_dim is None:
            side_embed_dim = 0
#         print("side embed dim = ", side_embed_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.W = nn.Linear(encoder_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)
#         print("hidden dim = ", hidden_dim)
#         print("encoder dim = ", encoder_dim)
    def forward(self, img_features, hidden_state):
#         print(hidden_state.size())
        U_h = self.U(hidden_state).unsqueeze(1)
        W_s = self.W(img_features)
#         print(W_s.size(), U_h.size())
        att = self.tanh(W_s + U_h)
        e = self.v(att).squeeze(2)
        alpha = self.softmax(e)
#         print(alpha.size())
#         print(img_features.size(), alpha.size())
        context = (img_features * alpha.unsqueeze(2)).sum(1)
        return context, alpha
    
class Attention2(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, hidden_dim):
        super(Attention2, self).__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.W = nn.Conv2d(in_channels, hidden_dim, 3, 1, 1)
        self.v = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, img_features, hidden_state):
        U_h = self.U(hidden_state).unsqueeze(1)
        W_s = self.W(img_features)
        W_s = W_s.view(W_s.size(0), self.hidden_dim, -1).permute(0, 2, 1)
        att = self.tanh(W_s + U_h)
        e = self.v(att).squeeze(2)
        alpha = self.softmax(e)
        n_img_features = img_features.view(img_features.size(0), img_features.size(1), -1)
        n_img_features = n_img_features.permute(0, 2, 1)
#         print(n_img_features.size(), alpha.size())
        context = (n_img_features * alpha.unsqueeze(2)).sum(1)
        return context, alpha

class newAttention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(newAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha