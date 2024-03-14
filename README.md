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



## Licence

This project is licensed under the MIT licence. See the [LICENSE] file (https://opensource.org/license/mit) for more information.
