![alt text](https://github.com/Orolol/sat2plan/blob/main/sat2plan/interface/icone/satellite_lewagon.jpg?raw=true)

# SAT2PLAN
Project developed for the **Data Science & AI** training course at [Le Wagon Paris](https://www.lewagon.com/fr/paris/data-science-course).

The sat2plan project aims to retrieve a satellite image from Google Maps for a given address or GPS coordinates, then reconstruct maps using a generative antagonistic neural network (GAN).

## Context

The sat2plan project is a group initiative as part of the **Data Science & AI** course offered by [Le Wagon Paris](https://www.lewagon.com/fr/paris/data-science-course). It falls within the field of computer vision and artificial intelligence, with the aim of creating a system capable of reconstructing 2D maps from satellite images retrieved from Google Maps.


![alt text](https://github.com/Orolol/sat2plan/blob/main/schema/schema-1.jpg?raw=true)


## Features

- **Image collection :** Around 100,000 pairs of images have been collected from the Google Maps API to feed the project. These images are essential for training and evaluating the generative adversarial neural network (GAN) model.

- **Geographical data :** The images collected cover a variety of American and European cities, with a satellite view and map of around 32 cities. The data is distributed so as to cover an area of approximately 1000 km² around each city, providing significant geographical diversity for the development of the model.

- **Data cleaning :** Prior to use in the project, a data cleaning process was carried out. This includes the removal of duplicates and irrelevant plans. This cleaning ensures that only relevant, quality data is used in the project, improving the efficiency and accuracy of the GAN model.

## Generative adversarial networks


![alt text](https://github.com/Orolol/sat2plan/blob/main/schema/schema-2.jpg?raw=true)

GANs were invented by Ian Goodfellow in 2014 and first described in the paper [Generative Adversarial Nets](https://proceedings.neurips.cc/paper_files/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf).

They consist of two distinct models:
- **The generator :** Its aim is to generate 'false' images that resemble the training images.
- **The discriminator :** Its aim is to examine an image and determine whether it is a genuine training image or a false image from the generator.

### How gan works
During training, the generator constantly tries to outwit the discriminator by producing increasingly false images, while the discriminator strives to become a better detective and correctly classify true and false images. The equilibrium of this game is reached when the generator generates perfect fakes that appear to come directly from the training data, and the discriminator must always guess, with 50% confidence, whether the generator's output is real or fake <sup>[1](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)</sup>.

### Alternative models
Three models are used in this project:

- **U-net :** It consists of a contracting path (left side) and an expansive path (right side). The contracting path follows the typical architecture of a convolutional network. It consists of the repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step we double the number of feature channels. Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution (“up-convolution”) that halves the
number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border pixels in
every convolution. At the final layer a 1x1 convolution is used to map each 64 component feature vector to the desired number of classes. In total the network has 23 convolutional layers <sup>[2](https://arxiv.org/pdf/1505.04597.pdf)</sup>.

- **UVC-GAN :** It is an improved method to perform an unpaired image-to-image style transfer based on a CycleGAN framework. Combined with a new hybrid generator architecture UNet-ViT (UNet-Vision Transformer) and a self-supervised pre-training, it achieves state-of-the-art results on a multitude of style transfer benchmarks <sup>[3](https://arxiv.org/pdf/2203.02557.pdf)</sup> <sup>[4](https://github.com/ls4gan/uvcgan)</sup>.

- **SAM-GAN :** This model aims to train generators to learn mapping relationships between source and target domains. The SAM-GAN model is divided into two main parts: a generator and a discriminator, where the generator consists of a content encoder, a style encoder, and a decoder <sup>[5](https://www.mdpi.com/2220-9964/12/4/159)</sup>.


![alt text](https://www.mdpi.com/ijgi/ijgi-12-00159/article_deploy/html/images/ijgi-12-00159-g002.png?raw=true)

## Graphic interface


## Licence
Ce projet est placé sous la licence MIT. Voir [LICENSE](https://opensource.org/license/mit) pour plus d'informations.

<span style="color:grey"><font size=”0.5em”>*This project is licensed under the MIT licence. See [LICENSE](https://opensource.org/license/mit) for more information.*</font></span>
