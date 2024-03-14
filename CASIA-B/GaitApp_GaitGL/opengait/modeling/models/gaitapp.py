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


# GaitApp-GaitGL
# # define GaitGL
class GLConv(nn.Module):
    def __init__(self, in_channels, out_channels, halving, fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(GLConv, self).__init__()
        self.halving = halving
        self.fm_sign = fm_sign
        self.global_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
        self.local_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        gob_feat = self.global_conv3d(x)
        if self.halving == 0:
            lcl_feat = self.local_conv3d(x)
        else:
            h = x.size(3)
            split_size = int(h // 2**self.halving)
            lcl_feat = x.split(split_size, 3)
            lcl_feat = torch.cat([self.local_conv3d(_) for _ in lcl_feat], 3)

        if not self.fm_sign:
            feat = F.leaky_relu(gob_feat) + F.leaky_relu(lcl_feat)
        else:
            feat = F.leaky_relu(torch.cat([gob_feat, lcl_feat], dim=3))
        return feat


class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1)*p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)


class Gait_GaitGL(nn.Module):
    def __init__(self, model_cfg):
        super(Gait_GaitGL, self).__init__()

        self.build_network(model_cfg)

    def build_network(self, model_cfg):
        in_c = model_cfg['channels']
        class_num = model_cfg['class_num']
        dataset_name = "CASIA-B"

        if dataset_name in ['OUMVLP', 'GREW']:
            # For OUMVLP and GREW
            self.conv3d = nn.Sequential(
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
                BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
            )
            self.LTA = nn.Sequential(
                BasicConv3d(in_c[0], in_c[0], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(inplace=True)
            )

            self.GLConvA0 = nn.Sequential(
                GLConv(in_c[0], in_c[1], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[1], in_c[1], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
            self.MaxPool0 = nn.MaxPool3d(
                kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.GLConvA1 = nn.Sequential(
                GLConv(in_c[1], in_c[2], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[2], in_c[2], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
            self.GLConvB2 = nn.Sequential(
                GLConv(in_c[2], in_c[3], halving=1, fm_sign=False,  kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[3], in_c[3], halving=1, fm_sign=True,  kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
        else:
            # For CASIA-B or other unstated datasets.
            self.conv3d = nn.Sequential(
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            )
            self.LTA = nn.Sequential(
                BasicConv3d(in_c[0], in_c[0], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(inplace=True)
            )

            self.GLConvA0 = GLConv(in_c[0], in_c[1], halving=3, fm_sign=False, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            self.MaxPool0 = nn.MaxPool3d(
                kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.GLConvA1 = GLConv(in_c[1], in_c[2], halving=3, fm_sign=False, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            self.GLConvB2 = GLConv(in_c[2], in_c[2], halving=3, fm_sign=True,  kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = GeMHPP()

        self.Head0 = SeparateFCs(64, in_c[-1], in_c[-1])

        if 'SeparateBNNecks' in model_cfg.keys():
            self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
            self.Bn_head = False
        else:
            self.Bn = nn.BatchNorm1d(in_c[-1])
            self.Head1 = SeparateFCs(64, in_c[-1], class_num)
            self.Bn_head = True

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        sils = ipts[0].unsqueeze(1)
        del ipts
        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)

        outs = self.conv3d(sils)
        outs = self.LTA(outs)

        outs = self.GLConvA0(outs)
        outs = self.MaxPool0(outs)

        outs = self.GLConvA1(outs)
        outs = self.GLConvB2(outs)  # [n, c, s, h, w]

        outs = self.TP(outs, seqL=seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs = self.HPP(outs)  # [n, c, p]

        gait = self.Head0(outs)  # [n, c, p]

        if self.Bn_head:  # Original GaitGL Head
            bnft = self.Bn(gait)  # [n, c, p]
            logi = self.Head1(bnft)  # [n, c, p]
            embed = bnft
        else:  # BNNechk as Head
            bnft, logi = self.BNNecks(gait)  # [n, c, p]
            embed = gait

        return embed, logi, bnft


class GaitApp_GaitGL(BaseModel):
    def build_network(self, model_cfg):
        self.gaitrecog = Gait_GaitGL(model_cfg)
        print("GaitApp_GaitGL")
        self.encoder = Encoder(model_cfg)
        self.decoder = Decoder(model_cfg)

    def forward(self, inputs, inputs_original=None):
        if inputs_original == None:
            ipts_test, labs_test, type_test, view_test, seqL_test = inputs

            test_embs, test_logits, bnft = self.gaitrecog(inputs)

            sils_test = ipts_test[0]
            if len(sils_test.size()) == 4:
                sils_test = sils_test.unsqueeze(1)

            n, _, s, h, w = sils_test.size()
            del ipts_test            
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': test_embs, 'labels': labs_test},
                    'softmax': {'logits': test_logits, 'labels': labs_test}
                },
                'visual_summary': {
                    'image/sils': sils_test.view(n*s, 1, h, w)
                },
                'inference_feat': {
                    'embeddings': bnft
                }
            }
            return retval
            
        else:

            ipts_cc, labs_cc, type_cc, view_cc, seqL_cc = inputs

            # # original input for recognition
            ipts_original, labs_original, type_original, view_original, seqL_original = inputs_original
            original_embs, original_logits, original_bnft = self.gaitrecog(inputs_original)

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

            # # triplet loss
            # cross_cloth_fake_recon_triplet = cross_cloth_fake_recon_3.view(8, 2, -1, s, h, w)[:, 0, :, :, :, :] # # [n//2, 1, s, h, w]
            # labs_cc_triplet = labs_cc.view(8, 2)[:, 0]

            # triplet loss
            cross_cloth_fake_recon_triplet = cross_cloth_fake_recon_3.squeeze(1)
            labs_cc_triplet = labs_cc         

            cross_cloth_fake_recon_detach = cross_cloth_fake_recon_triplet.detach()
            cross_cloth_labs_cc_detach = labs_cc_triplet.detach()
            cross_cloth_fake_inputs = ([cross_cloth_fake_recon_detach], cross_cloth_labs_cc_detach, type_cc, view_cc, seqL_cc)


            cross_embs, cross_logits, cross_bnft = self.gaitrecog(cross_cloth_fake_inputs)


            # cross cloth visualization
            # # NM -> CL and CL -> NM
            n, _, s, h, w = sils_cc.size()
            visual_Source_CL_triplet = torch.flip(sils_cc.clone().view(8, 2, -1, s, h, w), dims=[0, 1]).view(n, _, s, h, w)
            visual_Target_NM_triplet = sils_cc.clone()
            visual_triplet = torch.cat([visual_Target_NM_triplet, cross_cloth_fake_recon_triplet.unsqueeze(1).clone(), visual_Source_CL_triplet], dim=2)        
         

            n, _, s, h, w = sils_cc.size()

            retval = {
                'training_feat': {
                    'triplet_original': {'embeddings': original_embs, 'labels': labs_original},
                    'softmax_original': {'logits': original_logits, 'labels': labs_original},
                    'triplet_cross_cloth': {'embeddings': cross_embs, 'labels': labs_cc_triplet},                   
                    'softmax_cross_cloth': {'logits': cross_logits, 'labels': labs_cc_triplet}
                },
                'visual_summary': {
                    'image/visual_triplet': visual_triplet.view(n*s*3, 1, h, w) # nrow = 10
                },
                'inference_feat': {
                    'embeddings': str_outs
                }
            }
            return retval     