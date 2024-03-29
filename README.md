# GaitApp
Official code for "Integral Pose Learning via Appearance Transfer for Gait Recognition" (IEEE TIFS 2024).

## Step 1 GaitApp_Generator
Training GaitApp_Generator on CASIA-B

cd GaitApp_Generator
```
bash train.sh
```

After training, you will get Encoder and Decoder models in 

'GaitApp\GaitApp_Generator\output\CASIA-B\GaitApp_Generator\GaitApp_Generator\checkpoints'

## Step 2 Downstream Gait Models
Traing downstream gait models through original gait samples and generated samples by GaitApp_Generator

GaitApp_GaitSet on CASIA-B as a example

cd CASIA-B/GaitApp_GaitSet
```
bash train.sh
```

We build this project based on the open-source project [OpenGait](https://github.com/ShiqiYu/OpenGait).
