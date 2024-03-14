










import torch
import torch.nn as nn

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, BasicConv2d


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

        outs_app = feature_outs[:, 96:, :, :, :]


        return feature_outs, outs_str, outs_app


class Decoder(nn.Module):
    def __init__(self, model_cfg):
        super(Decoder, self).__init__()
        in_c_decoder = model_cfg['Decoder']['in_channels']
        self.convTran = nn.Sequential(
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

        self.convTran = SetBlockWrapper(self.convTran)
        

    def forward(self, x):
        
        out = self.convTran(x)

        return out


class GaitApp_Generator(BaseModel):
    def build_network(self, model_cfg):
        print("GaitApp_Generator")
        self.encoder = Encoder(model_cfg)
        self.decoder = Decoder(model_cfg)

    def forward(self, inputs):

        ipts_at, labs_at, type_at, view_at, seqL_at = inputs

        
        sils_at = ipts_at[0]
        if len(sils_at.size()) == 4:
            sils_at = sils_at.unsqueeze(1)

        n, _, s, h, w = sils_at.size()
        del ipts_at
        
        compact_feature, str_outs, app_outs = self.encoder(sils_at) # str = [n, 96, s, 1 ,1] cl = [n, 32, s, 1, 1]

        ## Branch 1 (Cross-Frame Reconstruction) reconstruct operation within a sequence
        app_random_outs = app_outs[:, :, torch.randperm(app_outs.size(2)), :, :] # [n, 32, s, 1, 1]
        
        ipts_decoder_1 = torch.cat([str_outs, app_random_outs], 1) # [n, 128, s, 1, 1]

        fake_recon_1 = self.decoder(ipts_decoder_1) # [n, 1, s, h, w]
        real_1 = sils_at # [n, 1, s, h, w]


        ## Branch 2 (Pose Feature Similarity) Structural features of the same person between two sequences (NM, CL)
        str_nm_2 = str_outs.view(8, 2, -1, s, 1, 1)[:, 0, :, :, :, :] # [n//2, 96, s, 1, 1]
        mean_str_nm_2 = torch.mean(str_nm_2, dim=2) # [n//2, 96, 1, 1]

        str_cl_2 = str_outs.view(8, 2, -1, s, 1, 1)[:, 1, :, :, :, :] # [n//2, 96, s, 1, 1]
        mean_str_cl_2 = torch.mean(str_cl_2, dim=2)# [n//2, 96, 1, 1]

        ## Branch 3 (GEI Similarity) exchange NM to CL for the same person
        # NM -> CL
        str_nm_3 = str_outs.view(8, 2, -1, s, 1, 1)[:, 0, :, :, :, :] # NM [n//2, 96, s, 1, 1]
        app_cl_3 = app_outs.view(8, 2, -1, s, 1, 1)[:, 1, :, :, :, :] # CL [n//2, 32, s, 1, 1]
        ipts_cl_decoder_3 = torch.cat([str_nm_3, app_cl_3], dim=1) # [n//2, 128, s, 1, 1]
        fake_cl_recon_3 = self.decoder(ipts_cl_decoder_3) # [n//2, 1, s, h, w]
        fake_cl_gei_3 = torch.mean(fake_cl_recon_3, dim=2) # [n//2, 1, h, w]

        real_cl_3 = sils_at.view(8, 2, -1, s, h, w)[:, 1, :, :, :, :] # [n//2, 1, s, h, w]
        real_cl_gei_3 = torch.mean(real_cl_3, dim=2) # [n//2, 1, h, w]

        # CL -> NM
        str_cl_3 = str_outs.view(8, 2, -1, s, 1, 1)[:, 1, :, :, :, :] # [n//2, 96, s, 1, 1]
        app_nm_3 = app_outs.view(8, 2, -1, s, 1, 1)[:, 0, :, :, :, :] # [n//2, 32, s, 1, 1]

        ipts_nm_decoder_3 = torch.cat([str_cl_3, app_nm_3], dim=1) # [n//2, 128, s, 1, 1]
        fake_nm_recon_3 = self.decoder(ipts_nm_decoder_3) # [n//2, 1, s, h, w]
        fake_nm_gei_3 = torch.mean(fake_nm_recon_3, dim=2) # [n//2, 1, h, w]
        
        real_nm_3 = sils_at.view(8, 2, -1, s, h, w)[:, 0, :, :, :, :] # [n//2, 1, s, h, w]
        real_nm_gei_3 = torch.mean(real_nm_3, dim=2) # [n//2, 1, h, w]      


        ## Branch 4 (Gait Recycle Consistency) recycle consistent
        cross_str_4 = str_outs.view(8, 2, -1, s, 1, 1) # [8, 2, 96, s, 1, 1]

        # NM -> CL and CL -> NM
        cl_change_4 = torch.flip(app_outs.view(8, 2, -1, s, 1, 1), dims=[0, 1]) # [8, 2, 32, s, 1, 1]
        cross_cloth_4 = torch.cat([cross_str_4, cl_change_4], dim=2).view(n, -1, s, 1, 1) # [n, 128, s, 1, 1]

        cross_cloth_fake_recon_4 = self.decoder(cross_cloth_4) # [n, 1, s, h, w]

        # recycle consistent (NM -> CL -> NM and CL -> NM -> CL)
        recycle_compact_feature_4, recycle_str_outs_4, recycle_cl_outs_4 = self.encoder(cross_cloth_fake_recon_4) # recycle_str_outs_3 = [n, 96, s, 1, 1] recycle_cl_outs_3 = [n, 32, s, 1, 1]
        recycle_ipts_4 = torch.cat([recycle_str_outs_4, app_outs], dim=1) # [n, 128, s, 1, 1]
        recycle_fake_recon_4 = self.decoder(recycle_ipts_4) # [n, 1, s, h, w]
        real_recycle_4 = sils_at

        ## Branch 5 (Weak Identity Constraint) InfoNCE
        cross_id_query = cross_cloth_4.permute(0, 2, 1, 3, 4).contiguous().squeeze(-1).squeeze(-1).view(n*s, -1) # [n*s, 128]
        cross_id_positive_keys = compact_feature.permute(0, 2, 1, 3, 4).contiguous().squeeze(-1).squeeze(-1).view(n*s, -1)  # [n*s, 128]
        cross_id_negative_keys = torch.flip(compact_feature.view(8, 2, -1, s, 1, 1), dims=[0, 1]).view(n, -1, s, 1, 1).permute(0, 2, 1, 3, 4).contiguous().squeeze(-1).squeeze(-1).view(n*s, -1).unsqueeze(1) # [n*s, 1, 128]

        ## Visualization
        # visual branch 1
        visual_branch_1 = torch.cat([real_1.clone(), fake_recon_1.clone()], dim=2) # [n, 1, 2s, h, w]

        # visual branch 3
        visual_nm_branch_real_str_3 = sils_at.view(8, 2, -1, s, h, w)[:, 0, :, :, :, :] # [n//2, 1, s, h, w]
        visual_cl_branch_real_str_3 = sils_at.view(8, 2, -1, s, h, w)[:, 1, :, :, :, :] # [n//2, 1, s, h, w]
        # NM -> CL
        visual_cl_branch_3 = torch.cat([visual_nm_branch_real_str_3.clone(), fake_cl_recon_3.clone(), real_cl_3.clone()], dim=2) # [n//2, 1, 3s, h, w]
        # CL -> NM
        visual_nm_branch_3 = torch.cat([visual_cl_branch_real_str_3, fake_nm_recon_3.clone(), real_nm_3.clone()], dim=2) # [n//2, 1, 3s, h, w]

        # visual GEI branch 3
        # NM -> CL
        visual_cl_branch_gei_3 = torch.cat([real_cl_gei_3.clone(), fake_cl_gei_3.clone()], dim=0) # [n, 1, h, w]
        # CL -> NM
        visual_nm_branch_gei_3 = torch.cat([real_nm_gei_3.clone(), fake_nm_gei_3.clone()], dim=0) # [n, 1, h, w]

        # visual recycle branch 4
        # NM -> CL -> NM and CL -> NM -> CL
        visual_branch_4_recycle = torch.cat([real_recycle_4.clone(), cross_cloth_fake_recon_4.clone(), recycle_fake_recon_4.clone()], dim=2) # [n, 1, 3s, h, w]

        # cross cloth visualization
        # # NM -> CL and CL -> NM
        visual_source_cl_triplet = torch.flip(sils_at.clone().view(8, 2, -1, s, h, w), dims=[0, 1]).view(n, -1, s, h, w) # [n, 1, s, h, w]
        visual_target_nm_triplet = sils_at.clone() # [n, 1, s, h, w]
        visual_triplet = torch.cat([visual_target_nm_triplet, cross_cloth_fake_recon_4.clone(), visual_source_cl_triplet], dim=2)                    
        

        n, _, s, h, w = sils_at.size()
        retval = {
            'training_feat': {
                'branch_1_img_recon': {'source': fake_recon_1, 'target': real_1},
                'branch_2_pose_sim': {'source': mean_str_nm_2, 'target': mean_str_cl_2},
                'branch_3_cl_gei': {'source': fake_cl_gei_3, 'target': real_cl_gei_3},
                'branch_3_nm_gei': {'source': fake_nm_gei_3, 'target': real_nm_gei_3},
                'branch_4_recycle': {'source': recycle_fake_recon_4, 'target': real_recycle_4},
                'branch_5_id': {'query': cross_id_query, 'positive_key': cross_id_positive_keys, 'negative_keys': cross_id_negative_keys}
            },
            'visual_summary': {
                'image/visual_branch_1': visual_branch_1.view(n*s*2, 1, h, w), # nrow = 10
                'image/visual_branch_3_NM_to_CL': visual_cl_branch_3.view((n//2)*s*3, 1, h, w), # nrow = 10
                'image/visual_branch_3_CL_to_NM': visual_nm_branch_3.view((n//2)*s*3, 1, h, w), # nrow = 10
                'image/visual_branch_GEI_3_NM_to_CL': visual_cl_branch_gei_3.view(n, 1, h, w), # nrow = 8
                'image/visual_branch_GEI_3_CL_to_NM': visual_nm_branch_gei_3.view(n, 1, h, w), # nrow = 8
                'image/visual_branch_4_recycle': visual_branch_4_recycle.view(n*s*3, 1, h, w), # nrow = 10
                'image/visual_triplet': visual_triplet.view(n*s*3, 1, h, w) # nrow = 10
            },
            'inference_feat': {
                'embeddings': str_outs
            }
        }
        return retval