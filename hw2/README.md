# Digits detection
Code for object detection on the Street View House Numbers (SVHN) Dataset.


## Hardware

The following specs were used to create the original solution.

- Ubuntu 18.04 LTS
- Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz
- 1x GeForce RTX 2080 Ti

## SVHN Dataset

You can download dataset from [here](http://ufldl.stanford.edu/housenumbers/)
```
data
  +- train
  | digitStruct.mat
  |  +- 33402 images
  
  +- test
  |  +- 13068 images

```

## 
## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:

1. [Installation](#installation)
2. download data
3. download pre-trained model
4. [inference](#Testing)

### Installation
```
pip install -r requirements.txt
```
### Download data
Download data from [Kaggle](https://www.kaggle.com/c/cs-t0828-2020-hw1/data)

### Download pre-trained model

You can download pretrained model from [here](https://drive.google.com/file/d/1pegMyiUYyVpJiS6uagute-sJvBpdzW2f/view?usp=sharing)

Then, move the model weight to `./model/`

### Training 
```
python main.py
```
You can change hyperparameters in `parameter.py`
### Testing

```
python test.py
```
After testing, You can get prediction with .csv format, then you can submit the prediction and get your accuracy 


## Citation
```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```
## Reference

[SVHN dataset](http://ufldl.stanford.edu/housenumbers/)
[torchvision function](https://github.com/pytorch/vision/tree/master/references/detection)
[TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
[faster R-CNN tutorial](https://www.lagou.com/lgeduarticle/129222.html)
[mmdetection tutorial: Train with customized datasets](https://github.com/open-mmlab/mmdetection/blob/master/docs/2_new_data_model.md)