# GaitApp
Integral Pose Learning via Appearance Transfer for Gait Recognition

## Step 1 GaitApp_Generator
First, training GaitApp_Generator on CASIA-B
cd GaitApp_Generator
```
bash train.sh
```

After training, you will get Encoder and Decoder models in 'GaitApp\GaitApp_Generator\output\CASIA-B\GaitApp_Generator\GaitApp_Generator\checkpoints'

## Step 2 Downstream Gait Models
Second, traing the downstream gait models through original gait samples and generated samples by GaitApp_Generator
GaitApp_GaitSet as a example



