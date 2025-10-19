Image classification using Convolutional neural network (CNN) with CIFAR 10 Dataset
 
Purpose:
The primary aim of image classification with the CIFAR-10 dataset using CNNs is to arrange (categorize) the images into one of the ten predefined classes accurately (e.g., airplane, automobile, bird, cat, etc.).
What are convolutional neural networks?

Convolutional neural networks (CNNs) are a type of neural network that uses deep learning algorithms to process and analyze visual data such as images, video or graphics.
 
When CNNs process visual data they recognize, extract and learn meaningful information. Using convolutional layers and filters that detect features from image data, CNNs can recognize and extract patterns. CNNs learns features at different levels of complexity starting from simple, low-level details to more abstract, higher -level details.
 
 
LeNet – 5 Architecture

Yann LeCun designed LeNet-5 in the late 1990s primarily for digit recognition tasks. The LeNet-5 architecture consists of the input layers, convolutional layers, pooling layers, and fully connected layers. All these layers connects at the end with a SoftMax classifier, which performs the classification. This architecture laid the foundation for modern CNNs, and the significance lies in its innovative use of these layers.
The LeNet-5 architecture systematically breaks down an image for classification into smaller, manageable pieces, which allows the neural network to identify simple features first. These features combines in deeper layers to recognize complex patterns and shapes, which lead to accurate identification of images.

 
The CIFAR-10 Dataset 
This dataset commonly and widely find its use for image classification tasks. It consists of 60,000 32 x 32-color images in 10 classes with 60,000 images per class. The images in the CIFAR -10 dataset is has both training and test images in 50,000 and 10,000 images respectively.
During training of the model, it is advisable to split the training sets into validation sets. These validation sets will allow for the evaluation of the model’s performance on unseen data, it will detect overfitting and provide a checkpoint to tune hyper-parameters. They will also adjust the model before testing on the final test sets.




Convolutional neural network mechanism
If a convolutional neural network is made to slide on an image with dimension 32 x32 x 3 representing the width, height and depth (RGB channels) in a CIFAR -10 DATASET, the input layer will receive the input image which can be an image or sequence of images. 
Pixels are the smallest element in a digitally displayed image. These pixels are composed of sub pixels in the colors of Red, Green and Blue (RGB) and each pixel is an image corresponding to a value of 0 to 255 with zero (0) being black and 255 white.
The convolutional layer extracts the features from the input dataset using a set of learnable filters (kernels) in a convolutional operation. The convolution operation involves sliding the filter through the input image step by step where each step is called a stride. The stride can be one (1), two (2), and three (3) but it is preferable to work with smaller strides.
During the sliding process, a multiplication of a grid from the original input image with the grid from the filter generates a value. Carrying out the average of these values produces the feature map. Smaller filters and smaller strides extract more features from the input image. A filter size of 3x3 and a stride of one (1) is usually preferred. 
Two other operations performed in a convolutional neural network architecture include the addition of activation functions and pooling. Activation functions such as ReLu, sigmoid, leaky ReLu brings non-linearity into the mathematical equation of the model making it useful for binary classification task. Using ReLu as the activation function changes all the negative numbers on a grid to zero (0) and when the number is positive and more than zero, it leaves the number as it is.
The application of filters and activation function (ReLu) does not change the output volume of the feature map, pooling makes this possible. The pooling layer also makes the computation faster, reduces memory and prevents overfitting. Max pooling and Average pooling are two most common types of pooling. The max-pooling layer makes use of the maximum number in each stride while average pooling takes the average of the numbers in each stride.
A flattening of the resulting feature maps into a one-dimensional vector after the convolution and pooling layers takes place so that they can be passed into a completely linked layer for categorization.
The fully connected layers take the input from the previous layer and computes the final classification. The output from the fully connected layers is then fed into a logistic function such as Softmax or sigmoid for classification task. This logistic function converts the output of each class into the probability score of each class.



