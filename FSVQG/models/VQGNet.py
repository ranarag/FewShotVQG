from .new_decoder2 import LSTM_Decoder
from .category_encoder import CategoryEncoder
from .Autoencoder import IEncoder
import math
from .transformer import *
import torch.nn as nn
import torch


def recast_category_embeds(cat_embeds, cat_biases, img_feat_size):
    cat_embeds = cat_embeds.unsqueeze(2)
    cat_biases = cat_biases.unsqueeze(2)
    ret_embeds = torch.cat(img_feat_size *[cat_embeds], axis=2)
    ret_biases = torch.cat(img_feat_size *[cat_biases], axis=2)
    return ret_embeds, ret_biases


class VQGNet(nn.Module):

    def __init__(self, num_categories, cat_hidden_size,
                 vocab_size, side_embed_dim=8, teacher_forcing=True, decoder_type='lstm', dec_n_layers=2, dec_n_heads=2,
                 encoder_dim=2048, decoder_dim=768, isCuda = True, embedding=None, cat_embedding=None, scale_shift=True, att_type='q'):
        super(VQGNet, self).__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.scale_shift = scale_shift
        self.decoder_type = decoder_type
        self.side_embed_dim = side_embed_dim
        print(scale_shift,  decoder_type)
        if decoder_type == 'lstm':
            if not self.scale_shift:
                num_channels = 768
            else:
                num_channels = encoder_dim
            self.add_module("decoder", LSTM_Decoder(vocab_size, num_channels, decoder_dim=self.decoder_dim, side_embed_dim = side_embed_dim, tf=teacher_forcing, \
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
                                               n_input_channels=self.encoder_dim,  n_hidden_channels=1024, att_type=att_type))
        else:
            print("Decoder type {} does not exist.".format(decoder_type))
            exit(1)

        if self.scale_shift:
            self.add_module("category_encoder", CategoryEncoder(num_categories, cat_hidden_size, self.encoder_dim, embedding=cat_embedding, scale_shift=self.scale_shift))
        else:
            self.add_module("category_encoder", CategoryEncoder(num_categories, cat_hidden_size, side_embed_dim, embedding=cat_embedding, scale_shift=self.scale_shift))
        if not self.scale_shift and self.decoder_type == 'lstm':
            self.conv_net = nn.Conv2d(encoder_dim, num_channels, 1)
            self.img_feat_size = 7

#         self.encoder.load_state_dict(torch.load(encoder_model))
#         if isCuda:
#             self.category_encoder.cuda()
# #             self.encoder.cuda()
#             self.decoder.cuda()
#             if not scale_shift:
#                 self.conv_net.cuda()
        # self._init_weights()

#         self.encoder.eval()

    def get_img_embeds_no_SS(self, imgs, cats):

        n_img_features = self.conv_net(imgs)


        cat_embeds = self.category_encoder(cats)
        # tot_features = torch.cat([img_features, category_embeds, cat_biases], dim=1)
        # n_img_features = self.net(tot_features)
        # n_img_features = n_img_features.view(n_img_features.size(0), self.img_c_size, self.img_feat_size, self.img_feat_size)
        # n_img_features = self.conv_net2(n_img_features)
        n_img_features = n_img_features.view(n_img_features.size(0), n_img_features.size(1), -1)
        n_img_features = n_img_features.permute(0, 2, 1)
#         print(img)
        return n_img_features, cat_embeds


    def get_img_embeds(self, imgs, cats):
        img_features = imgs.squeeze()
        
        category_embeds, cat_biases = self.category_encoder(cats)
#         print(img_features.size(), category_embeds.size(), cat_biases.size())
        recasted_embeds, recasted_biases = recast_category_embeds(category_embeds, cat_biases, img_features.size(2))
        
        n_img_features = img_features.mul(recasted_embeds) + recasted_biases
        n_img_features = n_img_features.permute(0, 2, 1)
        return n_img_features

    def forward(self, imgs, cats, captions):
        if self.decoder_type == 'lstm':
            if not self.scale_shift:
                n_img_features, cat_embeds = self.get_img_embeds_no_SS(imgs, cats)
#                 print(n_img_features.size(), cat_embeds.size())
                preds, alphas = self.decoder(n_img_features, captions, cat_embeds)
            else:
                n_img_features = self.get_img_embeds(imgs, cats)
                preds, alphas = self.decoder(n_img_features, captions)
        else: # decoder_type == 'transformer'
            cat_embeds = self.category_encoder(cats) 
        
            preds, alphas = self.decoder(imgs, cat_embeds, captions)
        return preds, alphas


