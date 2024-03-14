![alt text](https://github.com/Orolol/sat2plan/blob/main/sat2plan/interface/icone/satellite_lewagon.jpg?raw=true)

# SAT2PLAN
Project developed for the **Data Science & AI** training course at [Le Wagon Paris](https://www.lewagon.com/fr/paris/data-science-course).

The sat2plan project aims to retrieve a satellite image from Google Maps for a given address or GPS coordinates, then reconstruct maps using a generative antagonistic neural network (GAN).

## Context

The sat2plan project is a group initiative as part of the **Data Science & AI** course offered by [Le Wagon Paris](https://www.lewagon.com/fr/paris/data-science-course). It falls within the field of computer vision and artificial intelligence, with the aim of creating a system capable of reconstructing 2D maps from satellite images retrieved from Google Maps.

![alt text](https://github.com/Orolol/sat2plan/blob/main/temp/42622e7d-982d-4e40-9e03-c8228d934e5c.jpg?raw=true)

## Features

- **Image collection :** Around 100,000 pairs of images have been collected from the Google Maps API to feed the project. These images are essential for training and evaluating the generative adversarial neural network (GAN) model.

- **Geographical data :** The images collected cover a variety of American and European cities, with a satellite view and map of around 32 cities. The data is distributed so as to cover an area of approximately 1000 km² around each city, providing significant geographical diversity for the development of the model.

- **Data cleaning :** Prior to use in the project, a data cleaning process was carried out. This includes the removal of duplicates and irrelevant plans. This cleaning ensures that only relevant, quality data is used in the project, improving the efficiency and accuracy of the GAN model.

## Generative adversarial networks

![alt text](https://github.com/Orolol/sat2plan/blob/main/temp/47e254f2-0e70-4d4e-b844-762c312c349a.jpg?raw=true)








## Licence

This project is licensed under the MIT licence. See the [LICENSE] file (https://opensource.org/license/mit) for more information.
