# Image super-resolution
Code for Image super-resolution.


## Hardware

The following specs were used to create the original solution.

- Ubuntu 18.04 LTS
- Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz
- 1x GeForce RTX 2080 Ti

## Dataset

You can download dataset from [here](https://drive.google.com/drive/u/3/folders/1H-sIY7zj42Fex1ZjxxSC3PV1pK4Mij6x)
```
data
  +- training_hr_images
  | 291 high-resolution images
  
  +- tseting_lr_images
  |  14 low-resolution images

```



## Method

[VDSR](https://github.com/twtygqyy/pytorch-vdsr) and [msrresnet](https://github.com/cszn/KAIR)



## VDSR
To reproduct my submission without retrain, you need to do the following steps:

1. [clone](#clone)
2. [Download pre-trained model](#Download-pre-trained-model)
4. [Inference](#Testing)

### clone
```
git clone https://github.com/twtygqyy/pytorch-vdsr
```

### Download pre-trained model

You can download pretrained model from [here](https://drive.google.com/file/d/1G51gQs0vFngrsLtON_ScuOoKPQ3VyAd-/view?usp=sharing)


### Testing

```
python3 infer.py [your args]
```

#### Argument
* `--cuda`        use gpu 
* `--model`       your model path 
* `--scale`       my model is pre-trained with scale 3
* `--gpus`        whcih gpu you want to use 
* `--output`      output file name 
* `--path`  testing file path

Or you can just run 
```
sh mytest.sh
```
### Train VDSR 

You can train your own model 

```
python3 main_vdsr.py [your args]
```

#### Argument
* `--batchSize`   Training batch size
* `--nEpochs`     Number of epochs to train for
* `--lr`       Learning Rate. Default=0.1
* `--step`     Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10
* `--cuda`      Use cuda? 
* `--threads`  Number of threads for data loader to use, Default: 1
* `--weight-decay`  Weight decay, Default: 1e-4
* `--pretrained`  path to pretrained model (default: none)
* `--gpus`  gpu ids


```
sh mytrain.sh
```

If you want to train your own dataset, you need to use `data/generate_train.m` to generate new `train.h5` file.

## msrresnet
To reproduct my submission without retrain, you need to do the following steps:

1. [clone](#clone)
2. [Download pre-trained model](#Download-pre-trained-model)
4. [Inference](#Testing)

### clone
```
git clone https://github.com/cszn/KAIR
```

### Download pre-trained model

You can download pretrained model from [here](https://drive.google.com/drive/folders/1J6EGrNC6jbCm_VwDrRWmxosu45Vs1bmR?usp=sharing)


### Testing

```
python3 main_test_msrresnet.py 
```

### Train msrresnet

You can train your own model, you need to move training set to  `trainsets/trainH`, and modify parameter in `options/train_msrresnet_psnr`

```
python3 main_train_msrresnet_psnr.py 
```



## Reference

[VDSR code](https://github.com/twtygqyy/pytorch-vdsr)

[VDSR paper](https://cv.snu.ac.kr/research/VDSR/VDSR_CVPR2016.pdf)

[msrresnet code](https://github.com/cszn/KAIR)

[msrresnet paper](https://arxiv.org/abs/1809.00219)

## References
```
@InProceedings{wang2018esrgan, % ESRGAN, MSRResNet
    author = {Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Loy, Chen Change},
    title = {ESRGAN: Enhanced super-resolution generative adversarial networks},
    booktitle = {The European Conference on Computer Vision Workshops (ECCVW)},
    month = {September},
    year = {2018}
}
```