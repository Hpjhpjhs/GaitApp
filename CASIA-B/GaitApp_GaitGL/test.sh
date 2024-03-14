# # **************** For CASIA-B ****************
# GaitApp_GaitGL
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=56789 opengait/main_gaitapp_gaitgl.py --cfgs ./configs/gaitapp/gaitapp_gaitgl.yaml --phase test --log_to_file
