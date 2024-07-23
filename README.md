## Overview
Kidney stones are small, hard mineral and salt deposits that form within the kidneys or urinary tract. Detecting kidney stones is crucial for several reasons, and failure to do so can result in significant threats to the body's health. Nowadays many practitioners are involved in including automation in the field of medical learning, hence i decided to indulge in the particular field as i were intrigued to work for it. Hence, we are experimenting with deep neural networks with transfer learning in medical image analysis.

## Project Description

## What is done in this Project?

## TechStack, frameworks and libraries Requirements
- Tools: vs code
- Programming Language: Python
- Libraries: numpy, pandas, os, matplotlib, seaborn, opencv, sklearn-metrics, tensorflow, keras, random, glob, ssim, peak_signal_noise_ratio, mean_squared_error, scipy, skimage-restoration, models, layers, optimizers, metrics.

## Dataset Description and EDA
For this project, i utilized this dataset on Kaggle named as [CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone). The dataset consists of one folder inside which contains 12,446 unique data within which the cyst contains 3709, normal 5077, tumor 2283 and stone 1377 images, and a kidneyData.csv dataset which containes 6 columns that containes 2 integers value and 4 object/string value those were unnamed, image_id, path of file inside folder, diagnosis (tumour, cyst, stone, normal), target value, class.
![image](https://github.com/user-attachments/assets/6c552658-80fc-4a9c-8277-a316d6803a73)

## Image Preprocessing Techniques Used

### 1.) Image Segmentation
Image Segmentation is a technique in computer vision in extracting meaningful information from images or to separate and extract information of necessary object from other non-required objects in an image. In this project, image segmentation is useful in extracting important region such as stone, tumour or cyst from other region of kidneys from CT Scan images.
<br>Five methods were used for image segmentation techniques which were Threshold-based segmentation, Cluster-based segmentation, Region-based segmentation, Watershed segmentation, Edge-based segmentation.

Evaluation metrices used was Intersection over Union with "watershed segmentation" showing better results than other techniques which was 0.6789.

### 2.) Image Enhancement
Image Enhancement is a technique to improve image quality by Brightening, improving pixel value, sharpening and smoothening of image. CT Scan, X-Ray and MRI images could be blur, less-lightening hence difficult for medical profeszsional to identify & diagnose difference between stone or tumor. This technique is used in the project to improve the CT Scan, X-Ray and MRI images which are of old records of patients and to train our dataset it is important to improve their images quality.<br>
Four methods were used for image enhancement which were Histogram Equalization & Histogram Matching, Contrast Convolution & Correlation, Smoothening Spatial Filter & Sharpening Spatial Filter and High pass Frequency Domain & Low pass Frequency Domain.

Evaluation metrices used was Structural Similarity Index Measure (SSIM) with "histogram equalization" showing better results than other which was 0.939839.

Histogram Equalization shows good results for contrast of image. Similarly, convolution operation for Sharpening of image and laplacian filter(sharpening) & gaussian filter(smoothening) for spatial domain.

### 3.) Image Restoration
It is a technique to recover an old and degraded image back to its original form. such types of images containes several types noises which can be removed by specific image restoration techniques. This technique would help us to restore old records of patients diagnosed with kidney stone, tumor or cyst like structure. hence, it would be further more beneficial for more accurate training of our datasets.<br>
Eight methods were used for image restoration which were Gaussian noise, Impulse noise, Poisson noise, Exponential noise, Gamma noise, Rayleigh noise, Uniform noise and Periodic noise.

Evaluation metrics used was Peak Signal-to-Noise Ratio (PSNR) which was 33.766 dB after Gaussion, impulse, rayleigh and other noise being removed.

### 4.) Morphological Operations 
It is a broad set of image processing which can process image based on their shapes. It divides the images into no. of segments based on shapes present inside images.<br>
Two methods were used which were Dilation and Erosion
## Tranfer Learning Models Used

### 1.) CNN Architecture

### 2.) VGG_19

### 3.) MobileNet

## Class Balancing with SMOTE
## Results 
