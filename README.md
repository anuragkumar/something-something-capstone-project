something something

## Introduction

Common-sense video understanding entails fine-grained recognition of actions, objects, spatial temporal relations, and physical interactions, arguably well beyond the capabilities of current techniques. Neural Networks that are specifically trained to predict a class will never account for the changes in pose, position, or distance. However, these fine minute details are very important to get accurate information about the action that is being performed. For example, there is a very subtle difference between _showing the cup_ , _showing the pen in the cup_, or _pretending to show the cup_

## Problem Statement

Visual object classification has made major advances with the help of datasets like ImageNet which have been trained on neural networks. However, most of the datasets that are used are in the form of images. These images inhibit the networks to account for more deeply complex scene scenes and situations. Unlike images, videos consist of more information and can aid to understand which action is being performed. However, labeled video dataset can convey high-level concepts but fail to represent detailed physical aspects since the neural networks used for the classification of such dataset lack common-sense knowledge about the physical environment, unlike humans.

## Objective

The primary goal of this project is to develop a model which can account for the common sense factor with respect to the physical world while predicting the correct category of action class for each video. A user-friendly end-to-end application has also been developed to demonstrate this model. In addition to this we have added one more feature where if the prediction done by the model is wrong then the user can input the correct class and improve the accuracy of the model.

## Dataset

The 20BN-something-something dataset is a large collection labeled video clips which depicts humans performing pre-defined basic actions with basic objects.The version 2 of the dataset is available in the following website The video data is provided as one large TGZ archive, split into parts of 1 GB max. The total download size is 19.4 GB. The archive contains webm-files using the VP9 codec. Files are numbered from 1 to 220847. It consists of a total of 220,847 videos in .webm format and the length of videos range between 2 to 6 seconds.  
[![123](https://i.ibb.co/6FMp6Gv/123.png)](https://imgbb.com/)  
These videos comprise one of the 174 action categories. Furthermore, the videos as well as the labels are divided into three training, validation, and test sets where the training set. Each video has a respective label which describes the action. For example “Dropping \[something\] into \[something\]” where something represents the objects. There are four json files

1.  174 action categories labels - Eg : “Putting \[something\] into \[something\]”
2.  Dictionary for training set
    -   ID : Video Id (eg 123456)
    -   Label: Consisting of entire action (eg : Tipping perfume bottle over)
    -   Placeholders: Object (eg perfume bottle)
    -   Template: Action categories (eg: Tipping \[something\] over)
3.  Dictionary for validation set : Same subset as training set
4.  Dictionary for test set : Consist only the id of the videos.  
    [![training](https://i.ibb.co/YBh3m4C/training.png)](https://imgbb.com/)  
    [![Validation](https://i.ibb.co/kKLc9dF/Validation.png)](https://imgbb.com/)

# Models

## VGG-16

The initial model that was used was VGG-16. In this model the data pre-processing was done slightly differently. Instead of considering the entire video raw image frames were extracted and 24fps and with then they were resized initially by 84x84 and later 124x124.We initially grouped all the video into 9 handpicked classes which was followed by the extraction of the frames from the videos. To extract the raw frame parallel processing was used which in turn made the process comparatively faster. For the VGG-16 model we used NVIDIA GPU of 32GB RAM. Due to the limitations of RAM that was initially available we could not process the entire data set. Thus, we set a maximum limit of 7000 frames per class training dataset. The number 7000 was selected in order to prevent bias among the classes since 7000 was the minimum number of frames per class. After resizing the image the next step was to normalize the raw frames. Furthermore, the labels in the json files were also encoded using one hot encoder technique. The concept of transfer learning is used to develop this model.

_Transfer learning_ is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. Here we have used a pre-trained model of VGG-16. The weights that are used for this model are those that belong to the Imagenet dataset. The 3 fully-connected layers at the top of the network is not included here and the input shape of (124 , 124, 3) is given where in 3 stands or RGB image and (124, 124) is the size of the image. Later, the bottleneck features are defined which will be used to develop our output layer to sync it with the pre-trained vgg-16 model.

For the output layer, the dense output layer uses rectification linearity function (reclu) which makes sure no negative values pass through the layers. The dropout layer is used to prevent overfitting. The categorical cross entropy loss is used here since there are more than 2 classes.  
The first model is the one without any form of image augmentation

[![vgg](https://i.ibb.co/1LPKK6K/vgg.png)](https://ibb.co/6mTbbBb)

Later, ImageDataGenerator is used for the image augmentation wherein the frames are rotated and flipped horizontally. One can observe a good improvement in the accuracy of the same.

[![vgg-16](https://i.ibb.co/85gbkGr/vgg-16.png)](https://ibb.co/9THhPBb)

The last approach for this model was to use not only image augmentation but also to use fine tuning as well which comparatively improved the accuracy  
[![vgg-16p](https://i.ibb.co/fFr3k5t/vgg-16p.png)](https://ibb.co/QQYWPRn)

## 2D CNN

A sequential 2D CNN model with a very similar approach of data processing with the only exception of using training features instead of bottleneck features while keeping the target variable same that is the one hot encoded labels. A pre trained 2D CNN model is used where the layers had a similar configuration as the previous model. However, the model had poor accuracy.  
[![poor-acc](https://i.ibb.co/Fg1VQ6k/poor-acc.png)](https://ibb.co/hDNfGsx)

### Limitations of VGG-16 and 2D CNN

The GPU available was not strong enough to handle the processing of the entire raw frames data. There was a limitation of 7000 frames from each class. The frames that were selected were not necessarily related and as a main limitation of training a model on limited images of the action performed there was insufficient and possibly inaccurate information of the action obtained from the images.

# Main Model

## 3D-CNN for 4 classes

One of the biggest hurdles to get an accurate class prediction model in VGG-16 and 2D-CNN was how the models were unable to identify the relations between the frames because they were unable to determine the relationship between them. In addition to that the basic 3D CNN model was using 3D frames which was not helping the accuracy of the model. Thus, it was necessary to modify the architecture of the model to get better results.

One of the biggest modifications of the entire 3D-CNN model was to cut down the number of classes. There are two main reasons for this modification, one was because there was not a powerful GPU available to process the entire data and second cutting down the classes enabled us to take an entire data as video for all four defined classes.

### **Pre-processing for the json files**

There are a total of 4 json files where one consists of all the labels , and the rest 3 are dictionaries for training , validation and test dataset respectively consisting of the information as mentioned above. From the 174 defined classes we narrowed it down to 4 classes namely: ‘Dropping \[something\]’,‘Poking \[something\]’, ‘Tearing \[something\]’, ‘Holding \[something\]’. The json files were modified based on these classes. The training and validation labels json file was modified into a simplified form which consisted only of the id, label, template and placeholders. For eg: (“id”:“107014”,“label”:“dropping a card in front of a coin”,“template”:“Dropping \[something\]”,“placeholders”:\[“a card”,“a coin”\]).

### Pre-processing for video files

There are a total of approximately 220k videos present. All the videos were cropped from a specific randomly selected location. The desired size of the output from this was 84x84 since it was easier for the GPU to process this size of videos. These videos were later converted into tensor and then normalized using different data augmentation techniques by randomly flipping our videos horizontally by a default value of 0.5 , reversing the video frames in time by a default value of 0.5, randomly rotating the video by 15 degrees.

### Main Model

Since the entire videos were used we used a MultiColumn network which is useful when the video size is too long and it has to be split into multiple clips. It processes 3d-CNN on each clip and then averages resulting features across clipping before passing it to the classification layer.we have constructed a 3D CNN model. This model has proven to give us the maximum accuracy. A 3D CNN with 11 layers has been used and the kernel size is kept 3 for all three dimensions (time , H,W) except for the first layer. Time dimension is preserved with padding =1 and stride =1 and is averaged at the end. The cross entropy loss was used as a loss function. For optimizer stochastic gradient descent momentum is used because it aids in accelerating the gradient vectors in the right directions thus leading to faster convergence. The momentum is defined as 0.9. There is total of 30 epochs that has be run on the entire data

[![4class](https://i.ibb.co/tckdCDj/4class.png)](https://ibb.co/XpNwJt0)

## Object Detection

After predicting the correct class the next important was to correctly predict the object used while performing the action. The selective search method is been performed on the entire video for object detection. Here the selective search method algorthmically selects either fast or slow method. We have used a fast search method and after that we got a list of all the list of bounding boxes that represent where an object could exist.

A pretrained model of Resnet50 is loaded with the weights of the imagenet dataset is used as an image classifier. After selctive search we get our proposals and boxes. Proposals consist of list that will hold sufficiently large pre-processed ROIs from the input image, which we will feed into our ResNet classifier and boxes consist of bounding box coordinates corresponds to our proposals which usually consist of large areas.

The entire video stream is being processed and filters out smalls boxes which would’nt contain the boxes and extracts the ROI of the proposals and later updates the proposal and boxes list. W e filtered out the weak confidence interference and then converted and stored the (x,y) coordinates of the bounding box. The output is collated in the labels and then aplly non-maxima suppression in order to suppress the weak overlapping bounding boxes resulting in a single object detection.

[![detect](https://i.ibb.co/fSgzyKB/detect.png)](https://ibb.co/b30MCyq)

## Model Enhancements

### 3D CNN for 9 Classes

The biggest limitations of the previous three models was the lack of GPU capability to handle a huge dataset. To overcome this issue we used Northeastern University’s resource of a Discovery cluster.Discovery is a high performance computing (HPC) resource for the Northeastern University research community Discovery cluster enabled us to use a p100 GPU which has 512gb RAM which aided us in not only reducing our computational time but also to cover all 9 classes for the entire data. In addition to that we MobaXterm was the terminal that was used to access the cluster and the entire cluster was processed in srun mode that is the interactive mode. Also checkpoint is used since the data is huge and computational power is more. In addition to that the second limitation that we dealt with was how uncorrelated the frames were. Thus we instead of using the frames which gave us limited information, the entire video was used.
<a href="https://imgbb.com/"><img src="https://i.ibb.co/Br9HmnP/3d-cnn-final.png" alt="3d-cnn-final" border="0"></a>
<a href="https://imgbb.com/"><img src="https://i.ibb.co/1RqGRcz/3d-cnn-final-2.png" alt="3d-cnn-final-2" border="0"></a>

# Web Architecture

We have used Angular App and flask in order to build our UI. In our UI there are two most important features for the users.

### 1\. Classification

Here, the user input the video for which it wants to detect the action to the server from and then it will load the data from the file server. Furthermore, the file server will send the input video to the frame extractor which will have two functions. One would be to detect the action in the video and second would be to detect the object in the video. The output of both the detection would be sent back to the server which will in turn send back to the client.  
[![web](https://i.ibb.co/C1tkLMx/web.png)](https://ibb.co/SsxhSdq)

### 2\. Improving the model using user feedback

The user can either upload a new video with correct information about the action and object or it can also just correct the label for an existing video. Once the server receives the feedback it will create a trigger which will start the training of the dataset on cloud services and at the same time it will check for sufficient new data on the file server. Once the training is done with the new data the data from cloud will go back to the server so that it can be used next time to detect the labels of such actions and objects.  
[![web2](https://i.ibb.co/nQBFz79/web2.png)](https://ibb.co/9yh7sb6)

> Written with [StackEdit](https://stackedit.io/).
