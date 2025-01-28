# Deep Learning-Based Classification and Segmentation of Impacted Teeth from Dental X-rays  

## Overview  
This repository contains the implementation of deep learning models for the classification and segmentation of angular impaction from dental panoramic radiographs. The study leverages multiple neural network architectures, including CNN, Vision Transformer (ViT), CNN with ViT embedding, CLIP, and U-Net, to analyze dental X-rays and assist in identifying impacted teeth and their associated risks.  

## Table of Contents  
- [Overview](#overview)  
- [Dataset](#dataset)  
- [Methods](#methods)  
- [Requirements](#requirements)  

## Dataset  
The dataset comprises **693 panoramic radiographs**, split into training (80%), validation (10%), and test (10%) sets.  
- A **random oversampler** was applied to address class imbalance in the classification task.  
- The U-Net model was exclusively trained on **142 images of angular impaction data** for segmentation tasks.  
- Dataset annotations for segmentation include JSON files specifying the impacted teeth regions.  

## Methods  
The following deep learning models were applied to the dataset:  
1. **CNN**: Convolutional Neural Network for classification tasks.  
2. **ViT**: Vision Transformer for improved feature extraction and classification.  
3. **CNN with ViT embedding**: A hybrid model combining CNN's convolutional layers with ViT's embedding layers.  
4. **CLIP**: Explored for both zero-shot and fine-tuned classification tasks.  
5. **U-Net**: Applied for segmentation of angular impaction regions.   

## Requirements  
To run the models and reproduce the results, install the following dependencies:  
- Python 3.8+  
- TensorFlow 2.10+  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  
- Albumentations  

Install the requirements using:  
```bash  
pip install -r requirements.txt
