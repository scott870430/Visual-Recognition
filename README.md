# kaggle-car-brand-classification
Code for car brand classification challenge.

[InClass Prediction Competition](https://www.kaggle.com/c/cs-t0828-2020-hw1/overview)

## Hardware

The following specs were used to create the original solution.

- Ubuntu 18.04 LTS
- Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz
- 1x GeForce RTX 2080 Ti

## Dataset

```
data
  +- training_lables.csv
  +- training_data
  |  +- 11,185 images
  +- testing_data
  |  +- 5000 images

 
```
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

### Training 
```
python main.py
```
### Testing

```
python test.py
```

