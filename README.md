# AIML Coding Challenge 2024 - Random Forest Rangers

## Overview
In this challenge, we try to predict land-cover classes for patches of multi-spectral satellite imagery with deep neural networks!

## Installation
The list of the libraries which should be installed to run the pretrained Transformer could be found in the [requirements.txt](./requirements.txt) file.

## DataSet Description

### Training Data
The coding challenge is built around the EuroSat dataset, which you can download from Github. This data includes the labels (via the directory structure) as well as 64x64 pixel Sentinel-2 image patches with 13 bands. Alternatively, you can find a mapping from image-paths to labels from the train.csv file below. The dataset also includes RGB versions of the same images in a separate directory. Note that the resolution of all bands has already been harmonized to 10m. In the lecture and coding labs you already saw different ways to process such data with machine learning techniques. Use the Eurosat dataset to train a model for single-label classification of the land-cover classes.

### Test Data
To compete on Kaggle, you also need to download the testset from this page. It consists of ~4000 Sentinel-2 images that are not part of Eurosat. Your goal in this challenge is to obtain the best possible accuracy in land-cover classification for those images. Note that there are some differences between the train and test data. Sentinel-2 data is distributed at different processing levels.

### Submissions
To upload your results, create a .csv file with the same structure as the sample_submission.csv below and upload it to Kaggle.

## Model Development

In the coding challenge we developed the model in the following way.
### 1. ResNet12 

File: [Resnet12_v2](./ResNet12_v2.ipynb)

As a first step, we set up the basis of the coding challenge with a simple ResNet12, initially focusing on data loading, 
data transforming, differentiating between train, validation and test loop, and implementing the submission.

To run the code, we just need to run the python notebook.


### 2. Vision Transformer (ViT)
In the next step, we took the basis of the ResNet12 code, exchanged the model and replaced it with a VisionTransformer.

File: [ViT_v2](./ViT_v2.ipynb)

FineTuning was done with various tests with different parameters:

- learning rate
- epoch
- cross-validation
- batch size
- etc.

To run the code, we just need to run the python notebook.
The test metrics will be printed during the run.

### 3. Pretrained Vision Transformer

For the last step in the coding challenge, we have used a pretrained Vision Transformer as a base for our model from this github repository (https://github.com/crlna16/pretrained-vision-transformer/)

Additional to the previous two developments we used the pytorch lightning library.

Folder: [pretrained](./pretrained)

The config of the pretrained vision transformer could be found here: [default.yaml](./pretrained/configs/default.yaml)
The python application could be found here: [run.py](./pretrained/run.py)
Our transform operations on the data is defined in the [eurosat_module.py](./pretrained/src/eurosat_module.py)
The different steps of the model are defined in the [vision_transformer.py](./pretrained/src/vision_transformer.py)

To run the model we could simply run the [run.py](./pretrained/run.py). The needed configs for the model will imported at the beginning of the application.

For the visualization of the test-loss, validation_loss, accuracy we used the visualization suite: tensorboard.

## Results
All .csv files could be found in the results folder [results](./results)






