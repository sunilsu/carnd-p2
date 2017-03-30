# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[explore1]: ./images/explore1.png "Visualization"
[explore2]: ./images/explore2.png "Visualization"
[augmented]: ./images/aug.png "Augmented"
[pp1]: ./images/pp1.png "Preprocessing"
[pp2]: ./images/pp2.png "Preprocessing"
[download]: ./images/web_downloaded.png "Test Images from Web"
[hist_web]: ./images/hist_web_img.png "Histograms"
[featuremap1]: ./images/featuremaps1.png "Featuremaps"
[featuremap]: ./images/featuremaps.png "Featuremaps"
[featuremap3]: ./images/featuremaps3.png "Featuremaps"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Used numpy to get required summary.

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the **exploratory visualization** section of the IPython notebook.  

Here is an exploratory visualization of the data set.
* First is a sample image from every traffic sign class.
* Second is a bar chart showing how many training images are present for each class. The training images are highly skewed towards few classes.

![Samples from all classes][explore1]
![Counts for each class][explore2]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained within the **Pre-process the Data Set** section of the IPython notebook.

I considered 2 different methods,
* In one, I decided to convert the images to *grayscale* because most of the information about the traffic sign can be inferred from the image intenesity. I then *rescaled* the pixels intensities between the 2nd and 98th percentiles. I followed this with *Contrast Limited Adaptive Histogram Equalization* with a clip limit of 0.03. This also normalized the image between 0 and 1 because it helps in the training of deep nets.
  * Here is an example of a 50 km/h traffic sign image before and after this preprocessing step. ![Prep Processing using CLAHE][pp2]
* In second, I converted the image to YUV color space and then applied histogram equalization to the Y channel. I reconverted it back to RGB space and normalized the image channels between 0 and 1.
  * Here is an example of Right turn traffic sign image before and after this preprocessing step. ![Prep Processing using YUV][pp1]

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The provided dataset is already split into train/validation/test sets.

The **Pre-process the Data Set** section of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because the training images distribution was highly imbalanced. Also, augmenting data is a good strategy to improve accuracy of convnets. To add more data to the the data set, I used the following techniques,
* Apply a random rotation of 8 degrees, clockcwise or anti-clockwise
* Apply a random scaling of image between 0.9 and 1.1
* Apply a random translation between 0 to 2 pixels along x,y
* Add training samples so that the classes are balanced.
* I used skimage tranform api for this.
  * Here is an example of an original image and an augmented image:

 ![Augmented][augmented]

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the **Model Architecture** section of ipython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|		|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU          | |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16				
| Dropout             | |
| Fully connected		| input 400 (5x5x16), output  120 |
| RELU          | |
| Dropout             | |
| Fully connected		| input 120 , output  84 |
| RELU          | |
| Dropout             | |
| Fully connected		| input 84 , output  43 |
| Softmax				|   |


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the **Train, Validate and Test** section of ipython notebook.

To train the model, I used an Adam optimizer with default learning rate (0.01), a batch size of 128 and 30 epochs

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of **0.991**
* validation set accuracy of **0.959**
* test set accuracy of **0.948**

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  * I started with Lenet-5 architecture. I chose this since this network was designed to capture patterns in handwritten digits and traffic sign classification is a  similar classification problem. I changed the input convolution filters to deal with 3 channels of color. I tried this with *YUV* preprocessing step outlined above. I tried it with just the training data first and then with augmented data.
  * Next I tried Lenet-5 without color channel. I tried this with *CLAHE* preprocessing outlined above. Again I tried with just the give training data and with augmented data.
* What were some problems with the initial architecture?
  * The network was in the overfit area. I tried to control overfit by adding dropouts, reducing learning rate and changing epochs. I could not get the validation accuracy above 0.9.
  * I tried simpler network next, with just grayscale input and without data augmentation. The network was overfitting quickly, but adding dropouts helped to increase validation accuracy. So I chose this architecture.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  * Some of the adjustments I tried with Lenet-5 were 1) changing *conv-1* to operate on 3 channels of color input and 2) adding dropout layers to maxpool layer and fully connected layers.
* Which parameters were tuned? How were they adjusted and why?
  * I adjusted the learning rate and number of epochs. I tried learning rates of 0.01, 0.001 and 0.0001 and also few around these. I adjusted the number of epochs depending on how the training/validation loss changed with it.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  * The convolution layer helps to extract the patterns in the image that are relevant for classification. For example, different shapes of the traffic signs.
  * The dropout layer helped in reducing overfitting on the training data.

If a well known architecture was chosen:
* What architecture was chosen?
  * Lenet-5 with droputs to *conv2* *maxpool* and *fully connected* layers.
* Why did you believe it would be relevant to the traffic sign application?
  * Lenet-5 was able to capture patterns in the handwritten dataset and classify them into 1 of 10 classes. Traffic sign data is similar size and number of classes is slightly more. So, I believe this architecture could be used for traffic sign classification problem.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  * The accuracy on the training set is high. The accuracy on validation/test is similar and high. So, the model is working well.


### Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Web Images][download]

Here are the histograms of their intenesity

![Histogram][hist_web]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the **Test Model on New Images** section of the Ipython notebook.

Looking at the histograms of the 5 web images, the 2nd image (Right of way) is on the bright side, while the 4th image (30 km/h is on the darker side). Also the 50 km/h sign is much smaller with lot of background information. I would guess that the model would probably not do well on the 4th and 5th images.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| General caution      		| General caution   									|
| Right-of-way at the next intersection | Right-of-way at the next intersection |
| Keep right					| Keep right										|
| Speed limit (30km/h)	 | Speed limit (30km/h)	|
| Speed limit (50km/h) | Speed limit (50km/h)	|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.8%.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the **Test Model on New Images** section of the Ipython notebook.

For the first image, the model is relatively sure that this is a General Caution  (probability of 0.83), and the image does contain a General Caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .83         			| General Caution  									|
| .16     				| Traffic Signals 										|
| .00					| Pedestrians											|
| .00	      			| Road narrows on the right				 				|
| .00				    | Right-of-way at the next intersection |


For the second image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .96         			| Right-of-way at the next intersection |
| .04     				| Beware of ice/snow |
| .00					| Pedestrians											|
| .00	      			| Double curve	|
| .00				    | Traffic Signals |

For the third image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .97         			| Keep right |
| .03     				| Yield |
| .00					| Turn left ahead |
| .00	      			| Priority road |
| .00				    | No entry |

For the fourth image. model is not as certain as others, but still good.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .79         			| Speed limit (30km/h) |
| .08     				| Speed limit (50km/h) |
| .08					| Speed limit (80km/h) |
| .02	      			| Speed limit (60km/h) |
| .01				    | Speed limit (20km/h) |

For the fifth image, model uncertainity is higher as compared to other images.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .47         			| Speed limit (50km/h) |
| .14     				| Speed limit (30km/h) |
| .07					| Speed limit (80km/h) |
| .05	      			| Speed limit (60km/h) |
| .04				    | Keep right) |

### Feature Maps

Here is the featuremaps for 1) General Caution, 2) Keep right and 3) Speed limit 30 km/h

![feature maps][featuremap1]
![feature maps][featuremap]
![feature maps][featuremap3]
