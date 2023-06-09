from .new_decoder2 import LSTM_Decoder
from .answer_encoder import AnswerEncoder
from .category_encoder import CategoryEncoder
from .Autoencoder import IEncoder
import math
from utils import Factory, process_lengths
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


class VQGNetANS(nn.Module):

    def __init__(self, ans_vocab_size, ans_hidden_size,
                 vocab_size, side_embed_dim=768, teacher_forcing=True, 
                 encoder_model="ANSWER", encoder_type='lstm', decoder_type='lstm', dec_n_layers=2, dec_n_heads=4, 
                 encoder_dim=1536, decoder_dim=768, 
                 isCuda = True, embedding=None, ans_embedding=None, scale_shift=False):
        super(VQGNetANS, self).__init__()
        global encoder_types
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.decoder_type = decoder_type
        self.scale_shift = scale_shift
        self.side_embed_dim = side_embed_dim
        
        if not self.scale_shift:
            ans_encoder_dim = side_embed_dim
        else:
            ans_encoder_dim = self.encoder_dim
           
        self.add_module("answer_encoder", AnswerEncoder(ans_vocab_size, ans_hidden_size, ans_encoder_dim, 
                                                                    encoder_type,
                                                                    embedding=ans_embedding, 
                                                                    n_layers=dec_n_layers, n_heads=dec_n_heads, scale_shift=self.scale_shift))
      
        

        if decoder_type == 'lstm':
            if not self.scale_shift:
                num_channels = decoder_dim
            else:
                num_channels = encoder_dim
            print("num channels = ", num_channels)
            self.add_module("decoder", LSTM_Decoder(vocab_size, num_channels, decoder_dim=self.decoder_dim, \
                                                    side_embed_dim = side_embed_dim, tf=teacher_forcing, \
                                                    embedding=embedding, scale_shift=self.scale_shift))
        elif decoder_type == 'transformer':
            self.add_module("decoder", Decoder(n_layers=dec_n_layers, 
                                               vocab_size=vocab_size, 
                                               embed_dim=self.decoder_dim, 
                                               dropout=0.1, attention_method='ByChannel', 
                                               n_heads=dec_n_heads, embedding=embedding, \
                                               num_channels=self.encoder_dim))
        else:
            print("Decoder type {} does not exist.".format(decoder_type))
            exit(1)

        if not self.scale_shift and self.decoder_type == 'lstm':
            self.conv_net = nn.Conv2d(encoder_dim, num_channels, 1)
            self.img_feat_size = 7

            
        if isCuda:
            self.answer_encoder.cuda()
            self.decoder.cuda()
            if not self.scale_shift:
                self.conv_net.cuda()


    def get_img_embeds_no_SS(self, imgs, ans, alen):

        n_img_features = self.conv_net(imgs)


        ans_embeds = self.answer_encoder(ans, alen)

        n_img_features = n_img_features.view(n_img_features.size(0), n_img_features.size(1), -1)
        n_img_features = n_img_features.permute(0, 2, 1)
        return n_img_features, ans_embeds

    
    def get_img_embeds(self, imgs, ans, alen):
        img_features = imgs.squeeze()

        answer_embeds, cat_biases = self.answer_encoder(ans, alen)
        recasted_embeds, recasted_biases = recast_answer_embeds(answer_embeds, cat_biases, img_features.size(2))

        
        n_img_features = img_features.mul(recasted_embeds) + recasted_biases

        n_img_features = n_img_features.permute(0, 2, 1)
        return n_img_features


    def forward(self, imgs, ans, alen, captions):
        if not self.scale_shift:
            n_img_features, ans_embeds = self.get_img_embeds_no_SS(imgs, ans, alen)
#             print(ans_embeds.size())
            preds, alphas = self.decoder(n_img_features, captions, ans_embeds)
        else:
            n_img_features = self.get_img_embeds(imgs, ans, alen)
            preds, alphas = self.decoder(n_img_features, captions)
    

        
        return preds, alphas


