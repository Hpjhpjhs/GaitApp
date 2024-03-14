# # **************** For FVG ****************
# FVG_GaitSet
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=23456 opengait/main_gaitapp_gaitset_FVG.py --cfgs ./configs/gaitapp_gaitset_FVG/gaitapp_gaitset_FVG.yaml --phase test --log_to_file