## Overview
Kidney stones are small, hard mineral and salt deposits that form within the kidneys or urinary tract. Detecting kidney stones is crucial for several reasons, and failure to do so can result in significant threats to the body's health. Nowadays many practitioners are involved in including automation in the field of medical learning, hence i decided to indulge in the particular field as i were intrigued to work for it. Hence, we are experimenting with deep neural networks with transfer learning in medical image analysis.

## Project Description
The purpose is to create a technology that accelerates diagnostic procedure while simultaneously increasing kidney stone detection accuracy. This study focuses on automatic kidney stone identification because kidney stones offer substantial health concerns if left unnoticed and can improve medical diagnosis. The study integrates kidney stones, image segmentation, and image enhancement with deep neural networks and transfer learning to increase diagnostic accuracy. This project focuses on the use of Convolutional Neural Networks (CNNs) models in addition to well-known architectures like VGG16 and MobileNet. These models are skilled at diagnosing kidney stones by deciphering intricate patterns and features from medical imaging. Transfer Learning achieves high accuracy while using less large amounts of training information and computational power by using knowledge from pre-trained networks.

In medical imaging, data variability and quality pose a significant barrier. The project uses cutting-edge image processing methods for this. Kidney stones and other regions of interest are best separated from the surrounding tissue using image segmentation. This step is essential, in order to minimize background noise and direct the model's attention on the relevant features. Furthermore, picture upscaling techniques are employed to enhance the quality and detail of images, a crucial aspect of precise stone identification.

In this project, data augmentation is significant since it increases the amount and diversity of the dataset, both of which are necessary for training effective models. Several scenarios and conditions that the model might face in practical applications are simulated by means of techniques including translation, scaling, and rotation. This stage guarantees that the model is applicable to various kidney stone instances and is both accurate and generalizable.

It also uses the Synthetic Minority Oversampling Technique (SMOTE) to solve the issue of class imbalance in medical materials. This method can successfully identify various kidney stone sizes and types while also assisting in the creation of a balanced dataset and ensuring that the model does not lean towards the majority category. When assessing models, a number of criteria are taken into account, such as specificity, sensitivity, and accuracy. These measurements offer a thorough analysis of every model and its performance, revealing its advantages and disadvantages under different conditions.<br>
The best methods for kidney stone detection can be discovered by comparing models under variouspreprocessing settings.

## What is done in this Project?
- Data Augmentation: This step involved artificially increasing the size and diversity of the dataset to ensure robust training of the models.
- Image Preprocessing: Techniques such as image segmentation and enhancement were used to improve the quality of the input images.
- Class Balancing: SMOTE was used to address class imbalances by creating synthetic samples of the minority class (kidney stone images).
- Model Building: Created a CNN model architecture and worked on two transfer learning model VGG_16 & MobileNet on two types of dataset that is class imbalance and class balance dataset. Then compared the accuracy score of all three models on both these datasets.

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
The Convolutional Neural Network (CNN) model presented for kidney stone detection leverages a sequential architecture with a series of layers designed to extract features, reduce dimensionality, and ultimately classify the images into one of four categories. The model begins with a Conv2D layer that applies 32 convolution filters of size 3x3 to the input images of shape (150, 150, 3). This layer uses the ReLU (Rectified Linear Unit) activation function, which introduces non-linearity to help the network learn complex patterns.

Following this, a MaxPooling2D layer with a pool size of 2x2 is used to reduce the spatial dimensions (height and width) by a factor of 2. This down-sampling helps retain important features while reducing the computational load. The process is repeated with another Conv2D layer, which again applies 32 filters of size 3x3 and uses the ReLU activation function to introduce non-linearity. Another MaxPooling2D layer with a pool size of 2x2 further reduces the spatial dimensions.

After the convolutional and pooling layers, the data is flattened into a 1D vector using a Flatten layer. This transformation is essential for feeding the data into fully connected (dense) layers, which are better suited for classification tasks. The first Dense layer contains 128 neurons and uses the ReLU activation function to introduce non-linearity. This layer is followed by another Dense layer with 4 neurons, corresponding to the four classes for classification. This final layer uses the Softmax activation function to output a probability distribution over the four classes, making it suitable for multi-class classification.

The model is compiled using the Adam optimizer, which adjusts the learning rate dynamically during training, and the loss function used is Sparse Categorical Crossentropy, appropriate for multi-class classification problems with integer labels. The model's performance is evaluated using the accuracy metric. The training process involves running the model for 5 epochs, which refers to the number of complete passes through the training dataset.
![flc](https://github.com/user-attachments/assets/fada8593-743e-4432-af4c-18d19329ecd6)

### 2.) VGG_16
VGG16 architecture, a deep convolutional neural network pre-trained on the ImageNet dataset, to classify images into four classes. By employing transfer learning, the model benefits from the features learned by VGG16, enhancing the efficiency and accuracy of the image classification task.

#### a) Pre-trained VGG16 Model:
- The architecture starts with the pre-trained VGG16 model. The include_top=False argument excludes the fully connected layers at the top of the model, making it act as a feature extractor.
- The input shape is set to (150, 150, 3), matching the size of the input images.
- pooling='max' applies global max pooling to the output of the convolutional base, ensuring a fixed-size output.
- The weights are pre-trained on the ImageNet dataset.
  
#### b) Additional Layers:
- After the VGG16 base, a Flatten layer is added to convert the multi-dimensional output from VGG16 into a 1D vector.
- A Dense layer with 512 neurons and ReLU activation function follows, introducing non-linearity and enabling the model to learn complex patterns.
- Batch Normalization is included after the Dense layer to normalize the inputs, improving training stability and convergence speed.
- A Dropout layer with a dropout rate of 0.5 is used to prevent overfitting by randomly setting half of the input units to zero during training.
- The final Dense layer has 4 neurons (one for each class) with a Softmax activation function, producing a probability distribution over the classes.

#### c) Training Configuration:
- The pre-trained VGG16 model's weights are set to non-trainable (pretrained_model.trainable=False), ensuring that only the newly added layers are trained.
- The model is compiled using the Adam optimizer, which adjusts the learning rate dynamically during training.
- Sparse Categorical Crossentropy is used as the loss function, appropriate for multi-class classification with integer labels.
- Accuracy is used as the metric to evaluate the model's performance.
  
#### d) Training:
The model is trained for 5 epochs using the training and validation datasets. During each epoch, the model's weights are updated to minimize the loss and improve accuracy on the validation set.

![vgg16](https://github.com/user-attachments/assets/6b4ed00b-86a6-45bc-8d73-8dac63abd388)

### 3.) MobileNet
The architecture begins with the inclusion of the MobileNetV2 model, which is pre-trained on the ImageNet dataset. The include_top=False argument removes the final fully connected layer of MobileNetV2, allowing it to be used as a feature extractor. The input shape is specified as (150, 150, 3), corresponding to the size of the images used in this task. The pooling='max' argument ensures that the output of the MobileNetV2 model is a fixed-size vector by applying global max pooling.

Following the pre-trained MobileNetV2 model, the architecture is extended with additional layers to tailor it for the specific classification task. The output from MobileNetV2 is flattened using a Flatten layer, transforming the multi-dimensional output into a 1D vector. A Dense layer with 512 neurons and ReLU activation function is added next, introducing non-linearity and enabling the network to learn complex patterns. To improve training stability and accelerate convergence, a Batch Normalization layer is included, which normalizes the inputs of the Dense layer.

To prevent overfitting, a Dropout layer with a dropout rate of 0.5 is added, randomly setting half of the input units to zero during training. The final layer of the model is a Dense layer with 4 neurons, corresponding to the four classes for classification. This layer uses the Softmax activation function to output a probability distribution over the four classes.

The pre-trained MobileNetV2 model is set to non-trainable (pretrained_model.trainable=False), meaning its weights will not be updated during training. This approach allows the model to leverage the learned features from ImageNet while only training the newly added layers.

The model is compiled with the Adam optimizer, which dynamically adjusts the learning rate, and uses Sparse Categorical Crossentropy as the loss function, suitable for multi-class classification problems with integer labels. The model's performance is evaluated using accuracy. The training process is set for 5 epochs, where the model iteratively updates its weights to minimize the loss and improve accuracy on the validation dataset.

![Untitled Diagram drawio](https://github.com/user-attachments/assets/31c87785-8f84-462f-8061-f2ebaf0d4f31)

## Class Balancing with SMOTE
SMOTE(Synthetic Minority Over-sampling TEchnique) is a method used to address class imbalance in datasets. It works by generating synthetic instances of the minority class to balance the dataset. SMOTE identifies minority class samples and their nearest neighbors, then creates new synthetic samples by interpolating between these points. This technique helps improve model performance by providing a more balanced training set, which reduces bias towards the majority class and enhances the model's ability to detect and classify minority class instances accurately.  
I have applied SMOTE to avoid overfitting as dataset consist of imbalance class of data for stone, tumor, cyst and normal with 1834, 1101, 1242 and 4069 respectively.

## Results 
#### Application of models with preprocessing and without class balancing
- Training accuracy of CNN, VGG16 and MobileNet were 99.124%, 91.744% and 95.134%
- Validation accuracy of CNN, VGG16 and MobileNet were 100%, 93.138% and 97.042%
  
#### Application of models with preprocessing & class balancing
- Training accuracy of CNN, VGG16 and MobileNet were 100%, 97.33% and 98.57%
- Validation accuracy of CNN, VGG16 and MobileNet were 100%, 90.50% and 91.02%
