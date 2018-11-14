# **Traffic Sign Recognition**

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

[image1]: ./images/dataset.png "Visualization"
[image2]: ./images/distribution.png "Distribution"
[image3]: ./images/translated.png "Translated"
[image4]: ./images/scaled.png "Scaled"
[image5]: ./images/rotated.png "Rotated"
[image6]: ./images/distribution_augmented.png "Distribution After Augmentation"
[image7]: ./images/sermanet.png "SermaNet Model Architecture"
[image8]: ./images/found.png "New Images"
[image9]: ./images/prediction.png "Classification Result of New Images"
[image10]: ./images/top5.png "Top 5 Softmax"
[image11]: ./images/confusion_test.png "Confusion Matrix for Test Data"
[image12]: ./images/confusion_new.png "Confusion Matrix for New Images"
[image13]: ./images/precision_recall_test.png "Precision & Recall for Test Data"
[image14]: ./images/precision_recall_new.png "Precision & Recall for New Images"
[image15]: ./images/image1.png "Image 1"
[image16]: ./images/image1_conv1.png "Image 1"
[image17]: ./images/image1_conv2.png "Image 1"
[image18]: ./images/image2.png "Image 2"
[image19]: ./images/image2_conv1.png "Image 1"
[image20]: ./images/image2_conv2.png "Image 1"
[image21]: ./images/image22.png "Image 22"
[image22]: ./images/image22_conv1.png "Image 1"
[image23]: ./images/image22_conv2.png "Image 1"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted

#### 1. Submission Files

3 required files are provided:
- Ipython notebook with code: `Traffic_Sign_Classifier.ipynb`
- HTML output of the code: `Traffic_Sign_Classifier.html`
- A writeup report (either pdf or markdown): This file (`writeup.md`)

### Dataset Exploration

#### 1. Dataset Summary

First of all, I loaded all data set files including `train.p`, `valid.p` and `test.p`. Then, I used numpy library to summarize data set. Here is the summarization output:

```
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```

#### 2. Exploratory Visualization

I plotted 15 randomly chosen traffic signs per class to generally understand the dataset in below image.

![image1]

Then, a bar chart was used to visualize the number of samples of class for the train data. The bar chart is in the following figure.

![image2]

It is obvious that data distribution is deviated and should be uniformed.

### Design and Test a Model Architecture

#### 1. Preprocessing

As mentioned in above section, data distribution is nonuniform. For more accurate model, number of samples in each class should be approximately equal. In addition, generating slightly changed data may also make the model more accurate because augmentation makes model more robust to natural distractions. By generating fake data (a.k.a. augmenting dataset), data distribution was uniformed at the same time.

In [this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), Semanet at. al. used 3 methods for augmentation and I also used the same methods. For this purpose, 4 functions were implemented in the Ipython notebook (`randTranslation()`, `randScale()`, `randRotate()`, `generateAugmentedData()`).

`randTranslation()` function takes image and returns a randomly translated version of the image. The translation amount is chosen randomly `[-2, 2]` for both x and y dimensions. Here is an example output of the function:

![image3]

`randScale()` function scales an image and returns it. The scaling factor is between `[0.9, 1.1]`. Below is the an example image:

![image4]

`randRotate()` function rotates an image between `[-15, 15]` and returns it. The below image is an example output:

![image5]

`generateAugmentedData()` function takes X, y and max_count and generates augmented data until the numbers of samples per groups increases to max_count.

Augmentation is applied only for train data. The data distribution after augmentation is seen below figure.

![image6]

For normalization, at first, I used the scaling method e.g. scaling all pixel values in `[-1, 1]`. However, the validation accuracy was not good as expected. Then, I followed the preprocessing procedure explained in [this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) and [this article](http://cs231n.github.io/neural-networks-2/).

All images converted in `YUV` color space and only `Y` channel used for next steps because as discussed in the paper, feeding only `Y` channel increases model accuracy in contrast to expectations. Then, `Y` channel values were zero-centered by subtracting mean and normalized by dividing standard deviation. Mean augmentation ratio is `6.2` e.g. about 6 images were created by augmenting each image.

#### 2. Model Architecture

My final model architecture is a similar arhitecture with the one described in Sermanet at. al.'s paper. Following is a figure of general view of the algorithm taken from the paper.

![image7]

I provides the details of the layers in terms of types and sizes of each layer below.

##### 1st Stage

| Layer           | Size            | Description                           |
|-----------------|-----------------|---------------------------------------|
| Input           | (?, 32, 32, 1)  | Normalized `Y` channel of images      |
| Convolution     | (?, 28, 28, 32) | 5x5 kernel, 1x1 stride, valid padding |
| ReLU            | (?, 28, 28, 32) |                                       |
| Max pooling     | (?, 14, 14, 32) | 2x2 kernel, 2x2 stride, same padding  |
| Convolution     | (?, 10, 10, 64) | 5x5 kernel, 1x1 stride, valid padding |
| ReLU            | (?, 10, 10, 64) |                                       |
| Max pooling     | (?, 3, 3, 64)   | 4x4 kernel, 4x4 stride, same padding  |

##### 2nd Stage

| Layer           | Size            | Description                           |
|-----------------|-----------------|---------------------------------------|
| Input           | (?, 32, 32, 1)  | Normalized `Y` channel of images      |
| Convolution     | (?, 28, 28, 32) | 5x5 kernel, 1x1 stride, valid padding |
| ReLU            | (?, 28, 28, 32) |                                       |
| Max pooling     | (?, 14, 14, 32) | 2x2 kernel, 2x2 stride, same padding  |
| Convolution     | (?, 10, 10, 64) | 5x5 kernel, 1x1 stride, valid padding |
| ReLU            | (?, 10, 10, 64) |                                       |
| Max pooling     | (?, 5, 5, 64)   | 2x2 kernel, 2x2 stride, same padding  |
| Convolution     | (?, 1, 1, 128)  | 5x5 kernel, 1x1 stride, valid padding |
| ReLU            | (?, 1, 1, 128)  |                                       |

##### Classifier

| Layer           | Size      | Description                              |
|-----------------|-----------|------------------------------------------|
| Input           | (?, 704)  | Concatenated outputs of 1. and 2. stages |
| Fully connected | (?, 256)  |                                          |
| ReLU            | (?, 256)  |                                          |
| Fully connected | (?, 128)  |                                          |
| ReLU            | (?, 128)  |                                          |
| Dropout         | (?, 128)  | 0.70 keep ratio                          |
| Logits          | (?, 43)   |                                          |

#### 3. Model Training

Best found hyperparameter setting is given here:

| Parameter           | Value |
|---------------------|------:|
| Epochs              | 10    |
| Batch size          | 128   |
| Learning rate       | 0.001 |
| Dropout factor      | 0.30  |
| L2 scale            | 0.001 |
| Learning rate decay | 0.8   |

I chose to use Adam optimizer which is able to scale learning rate automatically. However, I saw that learning rate should have explicit upper limit when I started to train because after some epoch loops, train accuracy started to jump around.

#### 4. Solution Approach

As we know that neighbor pixels of images are related to each other and they creates some shapes. Because of this fact, CNN architecture really fits to image classification problem just like this problem e.g. traffic sign classification.

Thus, my solution steps are as follows:
1. Finding an average solution with a well known CNN architecture like `LeNet` with original train data
1. Applying data augmentation and normalization to train data
1. Developing the model architecture
1. Tuning hyperparameters

At first, I used LeNet architecture with a `0.30` dropout regularization at the first fully connected layer and got about `%93` validation accuracy while train accuracy was about `%99`. I thought that model was overfitting on train data so I added `L2` regularization with a rate `0.001`. I got about `%95` validation accuracy.

Further more, I implemented SermaNet architecture because uses intermediate features are also used as input of classifier in this architecture in addition to output of last convolution layer. I thought that intermediate features also helps classifier to be more accurate. As a result, I got `%97` validation accuracy with `%99` train accuracy by using SermaNet architecture.

In addition, I dig around hyperparameters and tested some different epochs, batch sizes, dropout factors, learning rates, etc. The best found combination is given above.

Finally, I got `%98` validation accuracy with `%99` train accuracy. The accuracies were close and this says that there is not much overfitting. I decided to stop there and run the model on test data and got `%95.8` test accuracy.

### Test a Model on New Images

#### 1. Acquiring New Images

I acquired 28 new traffic sign images in total. Some of them are found on web and the remaining are cropped from Google street view. The following figure shows them. Each image is given with a title indicating their label.

![image8]

Some of the new signs are blurred, some of them have reflection and some are taken under low lightning. Further more, some of them are rotated, some have affine transformation. I did not applied any correction or distraction. I just scaled them to 32x32.

#### 2. Performance on New Images

The below figure shows the found images with a title which has two number. The first number is the true label of the corresponding image and the second one is the prediction of the model. Red titles indicates that the model misclassified that image.

![image9]

The model misclassified only two of them. One of the misclassified image is `Speed limit (100km/h)` which is classified as `Speed limit (30km/h)` and the other is `Speed limit (120km/h)` which is classified as `No passing for vehicles over 3.5 metric tons`.

For the first misclassification, both of them have thick red border and black number in the red border but no other similarities. Maybe the model needs fine tuning. For the second misclassification, they both also have thick red border but the `Speed limit (120km/h)` image has a sun reflection on its top right edge. If model has the color information, it may classify correctly but color information decreases overall performance.

The model has accuracy of `0.929` on new images. This accuracy is lower than the accuracy of test data which is `0.958`. I think that 28 images are not enough for a reliable accuracy calculation since even one misclassification has effect accuracy too much.

#### 3. Model Certainty - Softmax Probabilities

The following figure illustrates the labels which have top 5 softmax probabilities. True image labels ordered on y axis and x axis indicates probability order. Each cell has label and its probability in the parenthesis.

![image10]

Almost all top 1 softmax probabilities is about 1.0 which indicates the model is really sure about its predictions except 2., 14., 17. rows. The image at the second row is misclassified and the correct classification is at 3. column. The image at row 14 is correctly classified but model is not quite sure about that. The image at row 17 is also correctly classified but this time, model is relatively more confident. The worst issue about the result is that model confidently misclassified the image at row 27 and the correct label which must be 8 is not in the top 5 label. Thus, this image must be really hard one to classify correctly.

### Suggestions to Make Your Project Stand Out!

#### 1. Augmenting The Training Data

This is discussed above in detail.

#### 2. Analyze New Image Performance in More Detail

The result for test data and newly found images are presented on [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) which are given in two figures below. Confusion matrix is a square matrix, rows and columns are indicates labels and each cell has an integer value indicating the number of images classified as the label for its column label which actually have true label of row label. We expect a [diagonal matrix](https://en.wikipedia.org/wiki/Diagonal_matrix) from a perfect model because all cell values of a diagonal matrix are zero expect main diagonal.

![image11]

![image12]

One can use confusion matrix to calculate [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) easily. Dividing the main diagonal values with sum of column values gives us precisions and dividing again main diagonal with this time sum of row values gives us recalls. Thus, precision and recall measures are calculated using the above confusion matrices data and presented as bar graphs for both test data and newly found images and given below.

![image13]

![image14]

#### 3. Create Visualizations of The Softmax Probabilities

A graph is created and given in related section.

#### 4. Visualize Layers of The Neural Network

The visualization of the activations of the first and the second convolutions for newly found 3 images are given below. The first example image is selected from misclassified images, second one is correctly classified and has a shape of lozenge and the third one is also correctly classified and has an inverted triangle shape.

| Original Image | First Convolutional Layer | Second Convolutional Layer |
|----------------|---------------------------|----------------------------|
| ![image15]     | ![image16]                | ![image17]                 |
| ![image18]     | ![image19]                | ![image20]                 |
| ![image21]     | ![image22]                | ![image23]                 |

We can see that some of the feature maps looks for edges having different directions. For example, top right feature map for the first convolutional layer looks for edges with a direction about -45 degree and the feature map at 3. row 2. column for again first convolutional layer looks for 45 degreed edges.
