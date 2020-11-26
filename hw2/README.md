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

## Method

I use two object detection models on this dataset separately.
1. faster R-CNN provided by Pytorch
2. yolov3 provided by mmdetection



## Faster R-CNN
To reproduct my submission without retrainig, you need to do the following steps:

1. [Installation](#installation)
2. [Download torchvision function](#Download-torchvision-function)
3. [Download pre-trained model](#Download-pre-trained-model)
4. [Inference](#Testing)

### Installation
```
pip install -r requirements.txt
```
### Download torchvision function

In my model, I use some function, which is provided by torchvision, in `faster R-CNN/torchvision`, or you can download the file from [here](https://github.com/pytorch/vision/tree/master/references/detection).


### Download pre-trained model

You can download pretrained model from [here](https://drive.google.com/file/d/1Pk7vDXtx_Wxb18w8i4PVC-OAjopssFp5/view?usp=sharing)

Then, move the model weight to `./model/`

#### Modify model
The default backbone of faster R-CNN is ResNet50. The image size of SVHN dataset is relatively small, so I change the backbone to mobilenet_v2 which is a ligher model. I also modify size and aspect ratios of anchor so that the model can foucs on smaller objects.

```
AnchorGenerator(sizes=((16, 32, 64, 128, 256),),
               aspect_ratios=((0.5, 1.0, 1.5),))
```



### Testing

```
python test.py -v pre-trained-weight
```
And then, You can get prediction with .json format, then you can submit the prediction and get your accuracy.
If you download the dataset from website, you can calculate the mAP of the test result by yourself.

### Training 

If you want to train your own model, you should change the data format, or you can just use `load_data.py`

You can change hyperparameters in `parameter.py`

- `version`
- `Batch_size`
- `lr` 
- `epochs` 

Training your own model!
```
python main.py
```
## MMDetection

To reproduct my submission without retrainig, you need to do the following steps:

1. [Download MMDetection](#Download-MMDetection)
2. [Move file](#Move-file)
3. [COCO dataset format](#COCO-dataset-format)
4. [Inference](#Testing)

### Installation
```
pip install -r requirements.txt
```
### Download MMDetection

Before using yolov3, you should install [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) first.

### Move file
After you download MMDetection, you need to 
1. move the `svhn` folder under `configs` folder.
2. move the `test.py` to `mmdetection/tools/`
 
### COCO dataset format

You need to change dataset to [COCO dataset format](https://cocodataset.org/#format-data), or you can perform `svhntococo.py`, it can generate COCO dataset format ```svhn_coco.json``` for training and ```svhn_coco_test.json``` for testing.
And you need to move ```svhn_coco.json``` to ```./dataset/train/``` and ```svhn_coco_test.json``` to ```./dataset/test/```

### modify config

* select yolov3 version
``` _base_ = 'yolo/yolov3_d53_320_273e_coco.py' ```
* modify class number
``` 
model = dict(
    bbox_head=dict(
        num_classes=10,))
```
* add class name and dataset type
```
dataset_type = 'COCODataset'
classes = ('10', '1', '2', '3', '4', '5', '6', '7', '8', '9')
```
* add data
```
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        img_prefix='../dataset/train/',
        classes=classes,
        ann_file='../dataset/train/svhn_coco.json'),
    val=dict(
        img_prefix='../dataset/train/',
        classes=classes,
        ann_file='../dataset/train/svhn_coco.json'),
    test=dict(
        img_prefix='../dataset/test/',
        classes=classes,
        ann_file='../dataset/test/svhn_coco_test.json'))
```
* ```samples_per_gpu``` is batch size

### Training 

```
python3 tools/train.py configs/svhn/yolov3_svhn_v2.py --gpu-ids 0
```
#### Argument
* `--gpu-ids` the number of gpu 

### Testing

```
python3 tools/test.py configs/svhn/yolov3_svhn.py  work_dirs/yolov3_svhn/latest.pth --format-only --options "jsonfile_prefix=./prediction/yolov3_v1"
```
Then, you will get a json file, which contain the model prediction resul, and you can submit the prediction and get your accuracy.

perform `convert_prediction.py` to get prediction of each image.

```
python convert_prediction.py -t yolov3_v1.bbox.json -v version1
```
#### Argument
* `-t` origin prediction result of yolo
* `-v` output json file name

If you download the dataset from website, you can calculate the mAP by
```
python3 tools/test.py configs/svhn/yolov3_svhn.py  work_dirs/yolov3_svhn/latest.pth --eval bbox
```
**Notice: you need to modify svhn_coco_test.json, let it contain label of testing data** 
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
