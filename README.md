# **Traffic Sign Recognition** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
This is the third project of [Udacity's Self Driving Car Nano Degree Program](https://www.udacity.com/drive). The goal is to recoginize and classify the traffic sign by using convolution neural network. The data is augmented and LeNet is customized to reach high accuracy in detection.
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./figures/30km_h.PNG "Speed Limit of 30km/h"
[image2]: ./figures/30km_h.PNG "Original"
[image3]: ./figures/30km_h_gray.PNG "Grayscale"
[image4]: ./figures/distribution_original.PNG "Original Dataset Distribution"
[image5]: ./figures/distribution_modified.PNG "Modified Dataset Distribution"
[image6]: ./my_traffic_sign/1.png "Traffic Sign 1"
[image7]: ./my_traffic_sign/2.png "Traffic Sign 2"
[image8]: ./my_traffic_sign/3.png "Traffic Sign 3"
[image9]: ./my_traffic_sign/4.png "Traffic Sign 4"
[image10]: ./my_traffic_sign/5.png "Traffic Sign 5"
[image11]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

Note that the actual size of training set and validation set is different from what is given above; their size are changed by data augmentation process.

#### 2. Include an exploratory visualization of the dataset.

The dataset we used is German Traffic Sign Dataset. Below is an example of traffic sign which indicates a speed limit of 30km/h.

![alt text][image1]

The distribution of dataset is also shown below:

![alt text][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because this is the suggested preprocessing approach in Pierre and Yann's paper. Also, my test shows the accuracy does improve when compared with RGB color space.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2] ![alt text][image3]

I've also done some affine transformation to increase the number of traffic signs. The dataset given is imbalanced; some types of traffic sings outnumber the others. This does impair the network performance quite a lot. Without any data augmentation, the accuracy of classifier can hardly surpass 90%. The mistakes made has a strong correlation with the number of samples for each type of sign. 

So, the main purpose of this step is simply to make dataset more balanced. This could be done by any types of affine transformation. A decent solution is provided by Ryein C. Goddard (https://github.com/Goddard/udacity-traffic-sign-classifier) where a few random alternation is done on the original figure. The solution is modified in my project so that the validation set also participates in this step, which is more reasonable IMO. After this is done, the dataset becomes more evenly-distributed, as is shown below:

![alt text][image5]

After the dataset is generated, normalization suggested in the lecture is done.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 1x1	    | 2x2 stride, valid padding, outputs 1x1x412    |
| RELU					|												|
| Fully connected		| input 412, output 122        									|
| RELU					|												|
| Dropout				| 50% keep        									|
| Fully connected		| input 122, output 84        									|
| RELU					|												|
| Dropout				| 50% keep        									|
| Fully connected		| input 84, output 43        									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

After the data augmentation, the dataset is spliited by sklearn and fed to the network. To train the model, I used Adam optimizer. The learning rate is reduced to 1e-4. The EPOCH is set to 50 and BATCH_SIZE is 128. You can see the trainning history in jupyter notebook.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.4%
* validation set accuracy of 98.8%
* test set accuracy of 92.1%

An iterative approach is taken and explained below:

***What was the first architecture that was tried and why was it chosen?***

I started with the LeNet model given in the lecture. I was only able to achieve around 90% accuracy no matter what parameters I changed.

***What were some problems with the initial architecture?***

The validation accuracy is lower than required; the test accuracy is roughly the same, which indicates a high bias(under fitting) scenario. So a larger model is needed to decrease the bias.

***How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.***

Since the model is under fitting, it is reasonable to make the model larger and deeper. I added a 1x1 convolution layer before the fully connected layer to increase the nonlinearity of the model. Then the model overfits the training/validation set. So dropout layer is added after each fully connected layer. With these two changes, the model is good enough to meet the requirement of project.

***Which parameters were tuned? How were they adjusted and why?***

The only hyper parameter tuned is the learning rate. It is decreased from 1e-3 to 1e-4 as the accuracy starts to decrease after only a few iterations with 1e-3. Decreasing it solves the problem.

***What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?***

Convolution layer is used as it is very common in computer vision task. It shares the parameters in each filter so that the number of parameters is reasonable. Also, spatial correlation can be learned from convolutional layer. Dropout layer is used to prevent over-fitting. The softmax layer is used as it is the most commonly used for classification.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy road      		| Bumpy road   									| 
| No vehicles     			| Speed limit (100km/h) 										|
| Go straight or left					| Go straight or left											|
| Speed limit (30km/h)	      		| Speed limit (30km/h)					 				|
| Ahead only			| Ahead only      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 92.1%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in last few cells of Jupyter notebook. The certainty is quite high for the prediction. The result is shown below:

For the first image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Bumpy road   									| 
| 0.00     				| Speed limit (20km/h) 										|
| 0.00					| Speed limit (30km/h)											|
| 0.00	      			| Speed limit (50km/h)					 				|
| 0.00				    | Speed limit (60km/h)      							|


For the second image:


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Speed limit (100km/h)   									| 
| 0.00     				| Speed limit (50km/h) 										|
| 0.00					| Speed limit (20km/h)											|
| 0.00	      			| Speed limit (30km/h)					 				|
| 0.00				    | Speed limit (60km/h)      							|


For the third image:


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Go straight or left   									| 
| 0.00     				| Speed limit (20km/h) 										|
| 0.00					| Speed limit (30km/h)											|
| 0.00	      			| Speed limit (50km/h)					 				|
| 0.00				    | Speed limit (60km/h)      							|

For the fourth image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Speed limit (30km/h)   									| 
| 0.00     				| Speed limit (20km/h) 										|
| 0.00					| Speed limit (50km/h)											|
| 0.00	      			| Speed limit (60km/h)					 				|
| 0.00				    | Speed limit (70km/h)      							|

For the fifth image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Ahead only   									| 
| 0.00     				| Speed limit (20km/h) 										|
| 0.00					| Speed limit (30km/h)											|
| 0.00	      			| Speed limit (50km/h)					 				|
| 0.00				    | Speed limit (60km/h)      							|
