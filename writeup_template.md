# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./output_images/flip.png "Flipped Image"
[image7]: ./output_images/loss_epoch.png "Training Epoch"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths of 6.

Data is normalized in the model using a Keras lambda layer.

Tried to add RELU layers to introduce nonlinearity, but the validation error becomes larger. 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Because of time limit, I did not use training mode to collect data. Instead, I just used the provided dataset.

I am fully aware the model can be trained better with more driving data, such a combination of center lane driving, recovering from the left and right sides of the road and getting more data from track 2. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the one used to train MNIST. I thought this model might be appropriate because in the lecture, it is mentioned that LeNet is a good place to start. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:
1. Normalizing layer 
2. Cropping layer focused on the driving lane 
3. 6 filters of 5*5 Convolutional layer
4. MaxPooling layer ( pooling size by default)
5. Another layer of 6 5*5 Convolutional layer
6. Another MaxPooling layer
7. Flatten Layer
8. 3 fully connected layer

#### 3. Creation of the Training Set & Training Process

Because of time limit, I did not use training mode to collect data. Instead, I just used the provided dataset.
To augment the data sat, I also flipped images and angles thinking that this would relieve the problem of overfitting.  
![alt text][image6]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs of 5 is good enough as evidenced by the figure below. After 5 epochs, the loss change is really small. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image7]
