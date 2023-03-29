# PAL
This repository contains the code for our paper [Unsupervised Vehicle Re-identification with Progressive Adaptation]).

### 1.Train the reID model in source domain.
We have provided the pretrained model [data/pretrained_model.mat]. You can click to Download and put the file in the data directory. The model is trained on VehicleID dataset. We also provide a file [models/net-epoch-50.mat] that you can Download and put in the model directory.

|          **File**          |    **Download**                                       |
|----------------------------|-------------------------------------------------------|
| data/pretrained_model.mat  |    [Download](https://1drv.ms/u/s!AufmTFpX_6Tta9K0hmdOn4Ra_gY?e=uFVc8D)    |
| models/net-epoch-50.mat    |    [Download](https://1drv.ms/u/s!AufmTFpX_6Ttaq1UfQhf64VwuEM?e=zj3tBf)    |

### 2.Generate Pseudo Images (CycleGAN)
The second stage is to generate fake images by Cycle.
We used the code provided in 'https://github.com/junyanz/CycleGAN'.
After generating images, use [prepare_data_VehicleID] to generate file 'cycle_train_part.mat', which is the part of the training set.

### 3.Train the reID model in target domain.
The third stage is to combine the original data and generated data to train the network.
Run 'demo.m'.

### Test
We have provided the test reID model and test features of VeRi dataset in our paper in [features] and [models].
You can run 'evaluation/evaluation.m' to evluate the well-trained model.

### Compile Matconvnet
You need to uncomment and modify some lines in `gpu_compile.m` and run it in Matlab. Try it~

(The code does not support cudnn 6.0. You may just turn off the Enablecudnn or try cudnn5.1)

If you fail in compilation, you may refer to http://www.vlfeat.org/matconvnet/install/

### Dataset
Download [VeRi Dataset]
