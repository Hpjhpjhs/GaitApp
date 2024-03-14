# # **************** For CASIA-B ****************
# GaitApp_Baseline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 opengait/main_gaitapp_baseline.py --cfgs ./configs/gaitapp/gaitapp_baseline.yaml --phase test --log_to_file