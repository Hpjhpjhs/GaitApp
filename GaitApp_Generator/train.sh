# # **************** For CASIA-B ****************
# # GaitApp
# Generator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=5432 gaitapp/main_gaitapp_generator.py --cfgs ./configs/gaitapp/gaitapp_generator.yaml --phase train --log_to_file
