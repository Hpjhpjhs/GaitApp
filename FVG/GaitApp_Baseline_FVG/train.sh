# # **************** For FVG ****************
# GaitApp_Baseline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=23456 opengait/main_gaitapp_baseline_FVG.py --cfgs ./configs/gaitapp_baseline_FVG/gaitapp_baseline_FVG.yaml --phase train --log_to_file
