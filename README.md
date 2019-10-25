# Project description

This project is a part of:  
 [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Goal

Main goal of this project is to use Deep Learning to classify traffic signs from:  
[German Traffic Signs Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) dataset.


### Dataset

The pickled dataset used in this project was downloaded from :  
https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip

### Where to start

The code is written in Jupyter Notebook available [here](./TrafficSignsClassifier.ipynb)

# Data exploration

### Dataset image samples

In dataset there are :
- 34799 training samples
- 4410 validation samples
- 12630 testing samples 

Every photo has resolution of 32x32 pixels and has 3 channels: RGB.

There are 43 classes of signs: They can be viewed in [signnames.csv](signnames.csv)

![alt text](/images/output_19_0.png)

There are visible:
- location differences
- brightness chalanges

### EDA
![alt text](/images/output_21_0.png)

* The dataset is be unbalanced
* Train/Validation/Test datasets have similat distributions so they are ready to be trained without any additional data stratification.

# Training process

### Augmentation

One of techniques used to create more training samples is online data augmentation. Many transformations were applie to make the dataset more diverse:
- **brightness** modification
- **contrast** modification
- **shift** image to prevent overfitting
- **scale** to be able to recognize sing independently to size
- **rotation** and **shear** cause sings can be visible from many different perspectives
- **gaussian noise** to prevent overfitting

Here are random samples of augmentation usage:

![alt text](/images/output_28_0.png)

# Network architecture

### Activation function

One of recent activation function has been used in the model:  
**"Mish: A Self Regularized Non-Monotonic Neural Activation Function"**  
Link to paper: https://arxiv.org/abs/1908.08681v1  
Implementation for PyTorch aken from https://github.com/lessw2020/mish  

### Regularization

For generalisation of model and to prevent overfitting **dropout layer** with rate of 0.5 was used between two classification layers.

To generalize sampled during training **Batch normalization** layers were used to penalize outlying samples.

### Layers

**Feature extractions:**
Four blocks of feature extractors were used in configuration visible below:
- Block1:
    - Conv2d
    - MaxPooling2d
    - Mish activation
    - BatchNorm2d

- Block2:
    - Conv2d
    - Mish activation
    - BatchNorm2d

- Block3:
    - Conv2d
    - MaxPooling2d
    - Mish activation
    - BatchNorm2d

- Block4:
    - Conv2d
    - MaxPooling2d
    - Mish activation
    - BatchNorm2d

**Classification layers:**
- AdaptiveAvgPool2d
- Linear 
- Mish
- Dropout
- Linear (with output equal to number of classes)

*Size:*  
The materialized model after training weights only 610 KB


# Results

Training was performed using:
- Cross-Entropy loss
- Optimizer used: Stochastic Gradient Descent with learning rate of 0.01 with momentum 0.9
- Learning rate scheduler with factor of 0.1 fired every 10 epochs
- Batch size: 256
- Number of epochs: 35

### Training loss change
![alt text](/images/output_57_1.png)

### Training accuracy
![alt text](/images/output_58_1.png)

### Heatmap

After training classification on test dataset was performed with accuracy score of **95.8%**

Confusion matrix for traffic signs in test sets:

![alt text](/images/output_61_1.png)

## Sample external sign photos

Sample images from outside of dataset were used to ches real-life scenario.  
Images were cropped manually to be square-like. The testing scales image to 32x32 square and performs inference.

**Results:**

 Sign        | Classification result 
-------------|-------------
![alt text](/images/output_63_1.png) | Speed limit (50km/h)
![alt text](/images/output_63_3.png) | Yield 
![alt text](/images/output_63_5.png) | *Pedestrians* (!!!) 
![alt text](/images/output_63_7.png) | No entry
![alt text](/images/output_63_9.png) | Stop


**Comment:**

Even though sample signs haven't beed checked for similarity ith training dataset most of external signs were properly classified.

'Children crossing' was classified as 'Pedestrians'. This is the case that was not visible in confusion matrix of test dataset. The real-life difference between those two signs is that on first there are two people visible ond secon on there is only one person. Part of the sign was ocluded by the tree what may cause tis misclassification.

# Final word
- The dataset presents many real-world challanges with position/lighting and shape of signs

- Classification of traffic signs on this dataset with very tiny images at the accuracy of 95.8% is ok. The result using bigger images would be greater cause more unique features of sings could be extracted.

- Used model presents possibility of using deep learning to address the classification problem. I believe with more research with bigger model, more augmentation, hiper-parameters finetuning achieving 99+ accuracy would be possible.