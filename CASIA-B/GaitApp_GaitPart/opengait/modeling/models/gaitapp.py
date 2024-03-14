from statistics import mode
from tkinter import BASELINE
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks, SetBlockWrapper, HorizontalPoolingPyramid, BasicConv2d

from torchvision.utils import save_image
from utils.common import get_attr_from, get_valid_args, is_list, is_dict
from modeling import backbones

class Encoder(nn.Module):
    def __init__(self, model_cfg):
        super(Encoder, self).__init__()
        in_c_encoder = model_cfg['Encoder']['in_channels']
        self.global_block1 = nn.Sequential(BasicConv2d(in_c_encoder[0], in_c_encoder[1], 5, 1, 2),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c_encoder[1], in_c_encoder[1], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2)
                                        )
        # 32 x 32 x 32

        self.global_block2 = nn.Sequential(BasicConv2d(in_c_encoder[1], in_c_encoder[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2),
                                        BasicConv2d(in_c_encoder[2], in_c_encoder[2], 3, 1, 1),                                       
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2)
                                        )
        # 64 x 8 x 8

        self.global_block3 = nn.Sequential(BasicConv2d(in_c_encoder[2], in_c_encoder[3], 3, 1, 1),                                      
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2),
                                        BasicConv2d(in_c_encoder[3], in_c_encoder[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.AdaptiveAvgPool2d(1)
                                        )
        # 128 x 1 x 1

        self.global_block1 = SetBlockWrapper(self.global_block1)
        self.global_block2 = SetBlockWrapper(self.global_block2)
        self.global_block3 = SetBlockWrapper(self.global_block3)



    def forward(self, x):
        feature_outs = self.global_block1(x)

        feature_outs = self.global_block2(feature_outs)

        feature_outs = self.global_block3(feature_outs)

        outs_str = feature_outs[:, :96, :, :, :]

        outs_cl = feature_outs[:, 96:, :, :, :]


        return outs_str, outs_cl


class Decoder(nn.Module):
    def __init__(self, model_cfg):
        super(Decoder, self).__init__()
        in_c_decoder = model_cfg['Decoder']['in_channels']
        self.convTran1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_c_decoder[0], in_c_decoder[1], 4, 1, 0, bias=False),
            nn.BatchNorm2d(in_c_decoder[1]),
            nn.LeakyReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(in_c_decoder[1], in_c_decoder[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(in_c_decoder[2]),
            nn.LeakyReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(in_c_decoder[2], in_c_decoder[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(in_c_decoder[3]),
            nn.LeakyReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(in_c_decoder[3], in_c_decoder[4], 4, 2, 1, bias=False),
            nn.BatchNorm2d(in_c_decoder[4]),
            nn.LeakyReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(in_c_decoder[4], in_c_decoder[5], 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

        self.convTran1 = SetBlockWrapper(self.convTran1)
        

    def forward(self, x):
        
        out = self.convTran1(x)

        return out


# GaitCloth-GaitPart
# # define GaitPart
from utils import clones
class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        ret = self.conv(x)
        return ret


class TemporalFeatureAggregator(nn.Module):
    def __init__(self, in_channels, squeeze=4, parts_num=16):
        super(TemporalFeatureAggregator, self).__init__()
        hidden_dim = int(in_channels // squeeze)
        self.parts_num = parts_num

        # MTB1
        conv3x1 = nn.Sequential(
            BasicConv1d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            BasicConv1d(hidden_dim, in_channels, 1))
        self.conv1d3x1 = clones(conv3x1, parts_num)
        self.avg_pool3x1 = nn.AvgPool1d(3, stride=1, padding=1)
        self.max_pool3x1 = nn.MaxPool1d(3, stride=1, padding=1)

        # MTB1
        conv3x3 = nn.Sequential(
            BasicConv1d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            BasicConv1d(hidden_dim, in_channels, 3, padding=1))
        self.conv1d3x3 = clones(conv3x3, parts_num)
        self.avg_pool3x3 = nn.AvgPool1d(5, stride=1, padding=2)
        self.max_pool3x3 = nn.MaxPool1d(5, stride=1, padding=2)

        # Temporal Pooling, TP
        self.TP = torch.max

    def forward(self, x):
        """
          Input:  x,   [n, c, s, p]
          Output: ret, [n, c, p]
        """
        n, c, s, p = x.size()
        x = x.permute(3, 0, 1, 2).contiguous()  # [p, n, c, s]
        feature = x.split(1, 0)  # [[1, n, c, s], ...]
        x = x.view(-1, c, s)

        # MTB1: ConvNet1d & Sigmoid
        logits3x1 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
                               for conv, _ in zip(self.conv1d3x1, feature)], 0)
        scores3x1 = torch.sigmoid(logits3x1)
        # MTB1: Template Function
        feature3x1 = self.avg_pool3x1(x) + self.max_pool3x1(x)
        feature3x1 = feature3x1.view(p, n, c, s)
        feature3x1 = feature3x1 * scores3x1

        # MTB2: ConvNet1d & Sigmoid
        logits3x3 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
                               for conv, _ in zip(self.conv1d3x3, feature)], 0)
        scores3x3 = torch.sigmoid(logits3x3)
        # MTB2: Template Function
        feature3x3 = self.avg_pool3x3(x) + self.max_pool3x3(x)
        feature3x3 = feature3x3.view(p, n, c, s)
        feature3x3 = feature3x3 * scores3x3

        # Temporal Pooling
        ret = self.TP(feature3x1 + feature3x3, dim=-1)[0]  # [p, n, c]
        ret = ret.permute(1, 2, 0).contiguous()  # [n, p, c]
        return ret


class Gait_GaitPart(nn.Module):
    def __init__(self, model_cfg):
        super(Gait_GaitPart, self).__init__()
        
        self.build_network(model_cfg)

    def build_network(self, model_cfg):

        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        head_cfg = model_cfg['SeparateFCs']
        self.Head = SeparateFCs(**model_cfg['SeparateFCs'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.HPP = SetBlockWrapper(
            HorizontalPoolingPyramid(bin_num=model_cfg['bin_num']))
        self.TFA = PackSequenceWrapper(TemporalFeatureAggregator(
            in_channels=head_cfg['in_channels'], parts_num=head_cfg['parts_num']))


    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)

        del ipts
        out = self.Backbone(sils)  # [n, c, s, h, w]
        out = self.HPP(out)  # [n, c, s, p]
        out = self.TFA(out, seqL)  # [n, c, p]

        embed_1 = self.Head(out)  # [n, c, p]

        return embed_1

    def get_backbone(self, backbone_cfg):
        """Get the backbone of the model."""
        if is_dict(backbone_cfg):
            Backbone = get_attr_from([backbones], backbone_cfg['type'])
            valid_args = get_valid_args(Backbone, backbone_cfg, ['type'])
            return Backbone(**valid_args)
        if is_list(backbone_cfg):
            Backbone = nn.ModuleList([self.get_backbone(cfg)
                                      for cfg in backbone_cfg])
            return Backbone
        raise ValueError(
            "Error type for -Backbone-Cfg-, supported: (A list of) dict.")


class GaitApp_GaitPart(BaseModel):
    def build_network(self, model_cfg):
        self.gaitrecog = Gait_GaitPart(model_cfg)
        print("GaitApp_GaitPart")
        self.encoder = Encoder(model_cfg)
        self.decoder = Decoder(model_cfg)

    def forward(self, inputs, inputs_original=None):
        if inputs_original == None:
            ipts_test, labs_test, type_test, view_test, seqL_test = inputs

            test_embs = self.gaitrecog(inputs)

            sils_test = ipts_test[0]
            if len(sils_test.size()) == 4:
                sils_test = sils_test.unsqueeze(1)

            n, _, s, h, w = sils_test.size()
            del ipts_test            
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': test_embs, 'labels': labs_test}
                },
                'visual_summary': {
                    'image/sils': sils_test.view(n*s, 1, h, w)
                },
                'inference_feat': {
                    'embeddings': test_embs
                }
            }
            return retval
            
        else:

            ipts_cc, labs_cc, type_cc, view_cc, seqL_cc = inputs

            # # original input for recognition
            ipts_original, labs_original, type_original, view_original, seqL_original = inputs_original
            original_embs = self.gaitrecog(inputs_original)

            del ipts_original

            
            sils_cc = ipts_cc[0]
            if len(sils_cc.size()) == 4:
                sils_cc = sils_cc.unsqueeze(1)

            n, _, s, h, w = sils_cc.size()
            del ipts_cc
            
            str_outs, cl_outs = self.encoder(sils_cc) # str = [n, 96, s, 1 ,1] cl = [n, 32, s, 1, 1]       

            # branch 3 recycle consistent
            # # cross id recycle consistent
            cross_str_3 = str_outs.view(8, 2, -1, s, 1, 1) # [8, 2, 96, s, 1, 1]

            # NM -> CL and CL -> NM
            cl_change_3 = torch.flip(cl_outs.view(8, 2, -1, s, 1, 1), dims=[0, 1]) # [8, 2, 32, s, 1, 1]
            cross_cloth_3 = torch.cat([cross_str_3, cl_change_3], dim=2).view(n, -1, s, 1, 1) # [n, 128, s, 1, 1]

            cross_cloth_fake_recon_3 = self.decoder(cross_cloth_3) # [n, 1, s, h, w]

            # triplet loss
            cross_cloth_fake_recon_triplet = cross_cloth_fake_recon_3.view(8, 2, -1, s, h, w)[:, 0, :, :, :, :] # # [n//2, 1, s, h, w]
            labs_cc_triplet = labs_cc.view(8, 2)[:, 0]

            cross_cloth_fake_recon_detach = cross_cloth_fake_recon_triplet.squeeze(1).detach()
            cross_cloth_labs_cc_detach = labs_cc_triplet.detach()
            cross_cloth_fake_inputs = ([cross_cloth_fake_recon_detach], cross_cloth_labs_cc_detach, type_cc, view_cc, seqL_cc)


            cross_embs = self.gaitrecog(cross_cloth_fake_inputs)    


            # cross cloth visualization
            # # NM -> CL and CL -> NM
            visual_Source_CL_triplet = torch.flip(sils_cc.clone().view(8, 2, -1, s, h, w), dims=[0, 1])[:, 0, :, :, :, :] # [n // 2, 1, s, h, w]
            visual_Target_NM_triplet = sils_cc.clone().view(8, 2, -1, s, h, w)[:, 0, :, :, :, :] # [n // 2, 1, s, h, w]
            visual_triplet = torch.cat([visual_Target_NM_triplet, cross_cloth_fake_recon_triplet.clone(), visual_Source_CL_triplet], dim=2)       
         

            n, _, s, h, w = sils_cc.size()

            retval = {
                'training_feat': {
                    'triplet_original': {'embeddings': original_embs, 'labels': labs_original},
                    'triplet_cross_cloth': {'embeddings': cross_embs, 'labels': labs_cc_triplet}
                },
                'visual_summary': {
                    'image/visual_triplet': visual_triplet.view((n//2)*s*3, 1, h, w) # nrow = 10
                },
                'inference_feat': {
                    'embeddings': str_outs
                }
            }
            return retval     