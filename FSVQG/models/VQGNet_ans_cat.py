from .new_decoder2 import LSTM_Decoder
from .answer_encoder import AnswerEncoder
from .category_encoder import CategoryEncoder
from .Autoencoder import IEncoder
import math
from utils import Factory
import torch.nn as nn
import torch
from .transformer import *
from .new_utils import count_parameters
class EncoderFactory(object):
    def __init__(self):
        self._sampler_dict = {}
        
    def register_sampler(self, name, sampler):
        self._sampler_dict[name] = sampler
    
    def get_sampler_obj(self, name, *args, **kwargs):
        try:
            obj = self._sampler_dict[name](*args, **kwargs)
        except KeyError as e:
            print("{} is not implemented or linked to factory class EncoderFactory; exiting".format(name))
            exit(1)
            
        return obj

    @property
    def list(self):
        for name, sampler in self._sampler_dict.items():
            print("{} : {}".format(name, sampler))
            
encoder_types = Factory()


encoder_types.register_class('CATEGORY', CategoryEncoder)
encoder_types.register_class('ANSWER', AnswerEncoder)

def recast_answer_embeds(ans_embeds, ans_biases, img_feat_size):
    ans_embeds = ans_embeds.unsqueeze(2)
    ans_biases = ans_biases.unsqueeze(2)
    ret_embeds = torch.cat(img_feat_size *[ans_embeds], axis=2)
    ret_biases = torch.cat(img_feat_size *[ans_biases], axis=2)
    return ret_embeds, ret_biases


class VQGNetANSCAT(nn.Module):

    def __init__(self, num_categories, cat_hidden_size, ans_vocab_size, ans_hidden_size,
                 vocab_size,  side_embed_dim=8, ans_side_embed_dim=768, teacher_forcing=True, encoder_type='lstm', \
                 decoder_type='lstm', dec_n_layers=2, dec_n_heads=4, \
                 encoder_model="ANSWER", encoder_dim=2048, decoder_dim=768, \
                 isCuda = True, embedding=None, \
                 cat_embedding=None, ans_embedding=None, \
                 scale_shift=False):
        super(VQGNetANSCAT, self).__init__()
        global encoder_types
#         print(num_categories)
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.scale_shift = scale_shift
        self.side_embed_dim = side_embed_dim
        self.decoder_type = decoder_type
        self.ans_side_embed_dim = ans_side_embed_dim
        print(self.scale_shift)
        if self.scale_shift:
            self.comb_embed = nn.Sequential(
                            nn.Linear(2 * self.encoder_dim, 3 *self.encoder_dim // 2),
                            nn.BatchNorm1d(3 * self.encoder_dim // 2),
                            nn.ReLU(),
                            nn.Linear(3 * self.encoder_dim // 2, self.encoder_dim),
                            nn.BatchNorm1d(self.encoder_dim),
                            nn.ReLU()
                            )
            self.comb_biases = nn.Sequential(
                            nn.Linear(2 * self.encoder_dim, 3 *self.encoder_dim // 2),
                            nn.BatchNorm1d(3 * self.encoder_dim // 2),
                            nn.ReLU(),
                            nn.Linear(3 * self.encoder_dim // 2, self.encoder_dim),
                            nn.BatchNorm1d(self.encoder_dim),
                            nn.ReLU()
                            )
            cat_embed_dim = self.encoder_dim
            ans_encoder_dim = self.encoder_dim
        else:
            self.comb_embed = nn.Sequential(
                            nn.Linear(self.side_embed_dim + self.ans_side_embed_dim, 3 * self.ans_side_embed_dim // 2),
                            nn.ReLU(),
                            nn.Linear(3 * self.ans_side_embed_dim // 2, self.ans_side_embed_dim),
                            nn.ReLU()
                            )
            cat_embed_dim = side_embed_dim
            ans_encoder_dim = ans_side_embed_dim
        
        self.add_module("category_encoder", CategoryEncoder(num_categories, cat_hidden_size, cat_embed_dim, \
                                                            embedding=cat_embedding, scale_shift=self.scale_shift))        
        self.add_module("answer_encoder", AnswerEncoder(ans_vocab_size, ans_hidden_size, \
                                                        ans_encoder_dim, encoder_type,
                                                        embedding=ans_embedding, \
                                                        n_layers=dec_n_layers, n_heads=dec_n_heads, \
                                                        scale_shift=self.scale_shift))

        if decoder_type == 'lstm':
            if not self.scale_shift:
                num_channels = decoder_dim
            else:
                num_channels = encoder_dim
            self.add_module("decoder", LSTM_Decoder(vocab_size, num_channels, decoder_dim=self.decoder_dim, \
                                                    side_embed_dim = ans_side_embed_dim, tf=teacher_forcing, \
                                                    embedding=embedding, scale_shift=self.scale_shift))
        elif decoder_type == 'transformer':
            self.scale_shift = False
            self.add_module("decoder", Decoder(n_layers=dec_n_layers, \
                                               vocab_size=vocab_size, \
                                               embed_dim=self.decoder_dim, \
                                               side_info_dim=8, side_info_embed_dim=5, \
                                               dropout=0.1, \
                                               attention_method="ByChannel", \
                                               n_heads=dec_n_heads, embedding=embedding, \
                                               n_input_channels=self.encoder_dim,  \
                                               n_hidden_channels=1024, att_type='q'))

        if not self.scale_shift and self.decoder_type == 'lstm':
            self.conv_net = nn.Conv2d(encoder_dim, num_channels, 1)
            self.img_feat_size = 7


        if isCuda:
            self.answer_encoder.cuda()
            self.category_encoder.cuda()

            self.decoder.cuda()
            if not scale_shift:                
                self.conv_net.cuda()
                self.comb_embed.cuda()
            else:
                self.comb_embed.cuda()
                self.comb_biases.cuda()
               
    def get_img_embeds_no_SS(self, imgs, cats, ans, alen):

        n_img_features = self.conv_net(imgs)

 
        ans_embeds = self.answer_encoder(ans, alen)
        cat_embeds = self.category_encoder(cats)
        tot_embeds = torch.cat([ans_embeds, cat_embeds], dim=1)
        tot_embeds = self.comb_embed(tot_embeds)
        n_img_features = n_img_features.view(n_img_features.size(0), n_img_features.size(1), -1)
        n_img_features = n_img_features.permute(0, 2, 1)
        return n_img_features, tot_embeds


    def get_img_embeds(self, imgs, cats, ans, alen):
        img_features = imgs.squeeze()
        answer_embeds, ans_biases = self.answer_encoder(ans, alen)
        category_embeds, cat_biases = self.category_encoder(cats)
        inp_embed = torch.cat((answer_embeds, category_embeds), 1)
        inp_bias = torch.cat((ans_biases, cat_biases), 1)
        comb_embed = self.comb_embed(inp_embed)
        comb_bias = self.comb_biases(inp_bias)
        recasted_embeds, recasted_biases = recast_answer_embeds(comb_embed, comb_bias, img_features.size(2))

        
        n_img_features = img_features.mul(recasted_embeds) + recasted_biases

        n_img_features = n_img_features.permute(0, 2, 1)
        return n_img_features
        
    def forward(self, imgs, cats, ans, alen, captions):
        if not self.scale_shift:
            n_img_features, tot_embeds = self.get_img_embeds_no_SS(imgs, cats, ans, alen)
            preds, alphas = self.decoder(n_img_features, captions, tot_embeds)
        else:
            n_img_features = self.get_img_embeds(imgs, cats, ans, alen)    
            preds, alphas = self.decoder(n_img_features, captions)
        return preds, alphas
