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

# Network architecture

### Network architecture: Model searching 

There were tests done with state-of-the-art models yt there were inaccurate for that small input as 32x32 where most of them are trained and measured with 224x224 resolution or more. There were two options to choose:
- scale up images
- use simplier model  
Second option was used to keep training faster. 

Initial idea of search of architecture was to use:
- Convolution layers to extract image features
- Use non-zero paddings in convolutions not to lower features resolutions (input image is very slow and using convolution with poolings might very fase reduce resoluton to zero)
- Pooling layers to reduce train complexity
- Activation functions to intorduce non-linearity.

After starting training from 3 blocks of (Conv-Pool-Activate-BatchNorm) it quickly turned out that 4-block network gives much better results.

Used model is simple having (Conv,Pool,Activate, BatchNorm) sequences repeated 4 times.  
After those classification layers (2 Linear separated by Dropout) are used.

### Network architecture: Batch Norm
Most moders state-of-the art methods use batch norm with training in batches of 2+ samples to fight with outlying activation values

*Dropout*  
For generalisation of model and to prevent overfitting **dropout layer** with rate of 0.5 was used between two classification layers.

To generalize sampled during training **Batch normalization** layers were used to penalize outlying samples. 

### Network architecture: Activation function

One of recent activation function has been used in the model:  
**"Mish: A Self Regularized Non-Monotonic Neural Activation Function"**  
Link to paper: https://arxiv.org/abs/1908.08681v1  
Implementation for PyTorch aken from https://github.com/lessw2020/mish  

This activation function has more 'soft edges':

![alt text](/images/mish_landscape.png)  
(Image from https://arxiv.org/abs/1908.08681v1 )

 Mish is performing much better than ReLU on many benchmark datasets (detail in the paper)



### Network architecture: Layers
The input tensor size is : (-1,3,32,32)


Below output shapes are visible.   
*(Input shapes of next layer are the same as output shape of previous layer.)*

```bash
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 30, 30]           2,432
         MaxPool2d-2           [-1, 32, 15, 15]               0
              Mish-3           [-1, 32, 15, 15]               0
       BatchNorm2d-4           [-1, 32, 15, 15]              64
            Conv2d-5           [-1, 64, 15, 15]          18,496
              Mish-6           [-1, 64, 15, 15]               0
       BatchNorm2d-7           [-1, 64, 15, 15]             128
            Conv2d-8           [-1, 64, 15, 15]          36,928
         MaxPool2d-9             [-1, 64, 7, 7]               0
             Mish-10             [-1, 64, 7, 7]               0
      BatchNorm2d-11             [-1, 64, 7, 7]             128
           Conv2d-12            [-1, 128, 7, 7]          73,856
        MaxPool2d-13            [-1, 128, 3, 3]               0
             Mish-14            [-1, 128, 3, 3]               0
      BatchNorm2d-15            [-1, 128, 3, 3]             256
AdaptiveAvgPool2d-16            [-1, 128, 1, 1]               0
           Linear-17                  [-1, 128]          16,512
             Mish-18                  [-1, 128]               0
          Dropout-19                  [-1, 128]               0
           Linear-20                   [-1, 43]           5,547
================================================================
Total params: 154,347
Trainable params: 154,347
Non-trainable params: 0
```

where '-1' stands for any number equal to batch size.

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

# Results
The materialized model after training weights only 610 KB

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

No. | Sign        | Classification result 
----|---------|-------------
1 | ![alt text](/images/output_63_1.png) | Speed limit (50km/h)
2 | ![alt text](/images/output_63_3.png) | Yield 
3 | ![alt text](/images/output_63_5.png) | *Pedestrians* (!!!) 
4 | ![alt text](/images/output_63_7.png) | No entry
5 | ![alt text](/images/output_63_9.png) | Stop

**Details:**

Top 5 results for all signs are presented below.  
Value in firs bracket corresponds to probability of being in this class (softmax used here.)
Proper classes marked in bold.

Sign 1:  
**TOP0(prob=1.00): Speed limit (50km/h) (classid=2)**  
TOP1(prob=0.00): Speed limit (80km/h) (classid=5)   
TOP2(prob=0.00): Speed limit (70km/h) (classid=4)   
TOP3(prob=0.00): Speed limit (30km/h) (classid=1)   
TOP4(prob=0.00): Speed limit (120km/h) (classid=8)   

Sign 2:  
**TOP0(prob=1.00): Yield (classid=13)**  
TOP1(prob=0.00): Speed limit (60km/h) (classid=3)  
TOP2(prob=0.00): No vehicles (classid=15)  
TOP3(prob=0.00): Speed limit (30km/h) (classid=1)  
TOP4(prob=0.00): No passing (classid=9)  
  
Sign 3:  
TOP0(prob=0.40): Pedestrians (classid=27)   
TOP1(prob=0.31): Double curve (classid=21)   
TOP2(prob=0.09): Bicycles crossing (classid=29)   
**TOP3(prob=0.04): Children crossing (classid=28)**   
TOP4(prob=0.04): Wild animals crossing (classid=31)   

Sign 4:  
**TOP0(prob=1.00): No entry (classid=17)**   
TOP1(prob=0.00): Stop (classid=14)  
TOP2(prob=0.00): No passing for vehicles over 3.5 metric tons (classid=10)  
TOP3(prob=0.00): Vehicles over 3.5 metric tons prohibited (classid=16)  
TOP4(prob=0.00): Priority road (classid=12)  

Sign 5:  
**TOP0(prob=1.00): Stop (classid=14)**   
TOP1(prob=0.00): No entry (classid=17)   
TOP2(prob=0.00): Speed limit (20km/h) (classid=0)   
TOP3(prob=0.00): No passing for vehicles over 3.5 metric tons (classid=10)   
TOP4(prob=0.00): Road work (classid=25)   

**Comment:**

Even though sample signs haven't beed checked for similarity ith training dataset most of external signs were properly classified.

'Children crossing' was classified as 'Pedestrians'. This is the case that was not visible in confusion matrix of test dataset. The real-life difference between those two signs is that on first there are two people visible ond second on there is only one person. Part of the sign was ocluded by the tree what may cause tis misclassification. Thas uncertainty is also visible on softmax probability calculation - only for this one sample there are values 'fat from 1'.

# Final word
- The dataset presents many real-world challanges with position/lighting and shape of signs

- Classification of traffic signs on this dataset with very tiny images at the accuracy of 95.8% is ok. The result using bigger images would be greater cause more unique features of sings could be extracted.

- Used model presents possibility of using deep learning to address the classification problem. I believe with more research with bigger model, more augmentation, hiper-parameters finetuning achieving 99+ accuracy would be possible.

# Possible issues/discussion
Classifier is trained in very constrained environment. Lets take look on missclassified samples:

![alt text](/images/badly.png)

The issues that are visible and might be an issue in production systems are:

- very blury images (when car goes onto a bump) 

- overexposed images when color dissapears or seems while. Those extrme cases might be cused by any light source shining on sign with too much light

- uderexposed images - when there is not enough light to compensate camera ISO speeds (this is second option to blurry image: wneh expose time is too long images are blury, when exposure time is too fast they're underexposed). The colors are much less wisible then and all details dissappear

- ocluded images - sometimes cause classification to thin the sign is something else.

- bad resolution in number area - there might me an issue with speed limits when 60 looks like 80 and vice versa: that might be connected to exposure time, blurrines, image resolution, camera quality

- for or other envirnmental issue might cause too colors look mich different than they really are.

- very light background (sky) might cause camera sensor to not compensate light for the sign and it might appear black

- partial sun flares might introduce artifacts on signs that are causing sign to look different


**To address some of those issues solutions might be applied:**

- track signs and classify many times on couple frames before finaly judning hat is vissible (softmax output could be returned and averaged)

- super resolution algorightms might improve image resolution before classification

- augumentation with oclusion might be used to improve quality of partially visible signs