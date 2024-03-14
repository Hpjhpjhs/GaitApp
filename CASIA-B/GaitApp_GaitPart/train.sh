# # **************** For CASIA-B ****************
# GaitApp_GaitPart
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=45678 opengait/main_gaitapp_gaitpart.py --cfgs ./configs/gaitapp/gaitapp_gaitpart.yaml --phase train --log_to_file
