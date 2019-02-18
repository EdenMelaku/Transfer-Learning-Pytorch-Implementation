# Transfer-Learning-Pytorch-Implmentation
Transfer learning in torchvision implemtation with different models
## Transfer Learning
Transfer learning is the improvement of learning in a new
task through the transfer of knowledge from a related task that has al-
ready been learned. While most machine learning algorithms are designed
to address single tasks, the development of algorithms that facilitate
transfer learning is a topic of ongoing interest in the machine-learning
community.

Fine tuning - starts with a pretrained model and update all of the model’s parameters for our new task, 
in essence retraining the whole model. 

Feature extraction- starts with a pretrained model and only update the final layer weights from which we derive predictions. It is called feature extraction 
because we use the pretrained CNN as a fixed feature-extractor, and only change the output layer.

refer - http://cs231n.github.io/transfer-learning/
      - https://ruder.io/transfer-learning/
 ## pretrainied models
 ### 1- AlexNet (2012)
In 2012, AlexNet significantly outperformed all the prior competitors and won the challenge by reducing the top-5 error from 26% to 15.3%.
The second place top-5 error rate, which was not a CNN variation, was around 26.2%. see http://cvml.ist.ac.at/courses/DLWT_W17/material/AlexNet.pdf
### VGGNet (2014)
The runner-up at the ILSVRC 2014 competition is dubbed VGGNet by the community and was developed by Simonyan and Zisserman . VGGNet consists of 16 convolutional layers and is very appealing because of its very uniform architecture. 
Similar to AlexNet, only 3x3 convolutions, but lots of filters. Trained on 4 GPUs for 2–3 weeks. see https://arxiv.org/pdf/1409.1556.pdf
### ResNet(2015)
ILSVRC 2015, the so-called Residual Neural Network (ResNet) by Kaiming He et al introduced 
anovel architecture with “skip connections” and features heavy batch normalization.They were able to train a NN with 152 layers while still having lower complexity than VGGNet. 
It achieves a top-5 error rate of 3.57% which beats human-level performance on this dataset. see https://arxiv.org/abs/1512.03385

## Data Set
### Mnist Data

The MNIST dataset is one of the most common datasets used for image classification and accessible from many different sources.
The MNIST database contains 60,000 training images and 10,000 testing images taken from American Census Bureau employees and American high school students
see http://yann.lecun.com/exdb/mnist/

## Quick start 
 
 1- **install python3** 
 
 2- **install requirements**:
  ```
  pip install -r requirements.txt
  ```
   
 3- **Extract dataset**:
 ```
   curl https://raw.githubusercontent.com/EdenMelaku/Transfer-Learning-Pytorch-Implmentation/master/mnist_png.tar.gz | xzC /temp
 ```
   
 ### 4- Running examples
 #### 1- feature extraction with alexNet:
 ```
                 cd feature\ extraction
                 python3 minst_alexnet.py --dataFolder /temp/mnist_png
```
 #### 1- fine tuning with alexNet:
```
                 cd  fine\ tuning
                 python3 minstDataset+AlexNet.py --dataFolder /temp/mnist_png
              
```
                  
   ## References
   1-https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
   
   2- https://ruder.io/transfer-learning/
   
   3-https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5
   
   4-http://cs231n.github.io/transfer-learning/
      

