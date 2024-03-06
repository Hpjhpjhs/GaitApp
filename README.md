# GaitApp
Integral Pose Learning via Appearance Transfer for Gait Recognition

## First, training GaitApp_Generator on CASIA-B
cd GaitApp_Generator
```
bash train.sh
```

After GaitApp_Generator training, you will get the Encoder and Decoder model in 'GaitApp\GaitApp_Generator\output\CASIA-B\GaitApp_Generator\GaitApp_Generator\checkpoints'

## Second, traing the downstream gait models through original gait samples and generated samples by GaitApp_Generator
GaitApp_GaitSet as a example



