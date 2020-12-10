# Tiny PASCAL VOC dataset
Code for instance segmentation on the Tiny PASCAL VOC dataset.


## Hardware

The following specs were used to create the original solution.

- Ubuntu 18.04 LTS
- Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz
- 1x GeForce RTX 2080 Ti

## Tiny PASCAL VOC Dataset

You can download dataset from [here](https://drive.google.com/drive/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK)
```
data
  +- train
  | pascal_train.json
  |  +- 1,349  images
  
  +- tset.json
  |  +- 100 images

```

## YACS
I use [YACS](https://github.com/rbgirshick/yacs) in my project. It can help me track model config and reproduct model. With python logging library, it can automatically save each version for me



## Method

Mask R-CNN from [torchvision](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)



## Mask R-CNN
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

if you use normalizae, you need to add normalize class form [here](https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py)
```
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
```

And add this function to `utils.py` for change the predcition mask to rle format
```
def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskutil.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
    return compressed_rle
```

### Download pre-trained model

You can download pretrained model from [here](https://drive.google.com/file/d/1Ol4S84vnXCU4wnFCpZjgfojtrzdGP_kZ/view?usp=sharing)


### Testing

```
python test.py [your args]
```

#### Argument
* `--gpu-ids`         the which gpu you want to use 
* `--config`          the name of config which you want to use 
* `--version`         whcih version you want to use
* `--epoch`           whcih epoch you want to use 
* `--model-path`      where is your model 
* `--output-version`  model output version
* `--mask-threshold`  threshold of your mask

Or you can just run 
```
bash test.sh
```
### Training 

If you want to train your own model, you should change the data format, or you can just use `load_data.py`

You can change hyperparameters in `config.yaml`

  - `HIDDEN_LAYER`
  - `CLASS_NUMBER`
  - `BATCH_SIZE`
  - `LR`
  - `EPOCHS`
  - `HIDDEN_LAYER`
  - `TRAINABLE`
  - `LR_STEP`
  - `LR_GAMMA`
  - `ISNORM`

#### Modify model
The default hidden layer of Mask R-CNN head is 256. You can change it in `config.yaml`


#### Training your own model!
```
python main.py [your args]
```

#### Argument
* `--gpu-ids`         which gpu you want to use 
* `--config`          the name of config which you want to use 
* `--version`         your model's version
* `--save-dir`        where you want to save your model and log 
* `--log-file-name`   what is your log name 


Or you can just run 
```
bash train.sh
```



If you use your own dataset, you need to calculate the mAP by yourself, or use coco to evaluate.


## Reference

[torchvision function](https://github.com/pytorch/vision/tree/master/references/detection)

[TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

[Mask R-CNN tutorial](https://blog.csdn.net/u013685264/article/details/100564660)

[yacs_config](https://github.com/rbgirshick/yacs)