![alt text](https://github.com/Orolol/sat2plan/blob/main/sat2plan/interface/icone/satellite_lewagon.jpg?raw=true)

# SAT2PLAN
Projet développé dans le cadre de la formation **Data Science & AI** de [Le Wagon Paris](https://www.lewagon.com/fr/paris/data-science-course).

Le projet **SAT2PLAN** vise à récupérer une image satellite à partir de Google Maps pour une adresse ou des coordonnées GPS données, puis à reconstruire des cartes à l'aide d'un réseau de neurones antagonistes génératif (GAN).

-----------------------------------------------------------------------------

*Project developed for the **Data Science & AI** training course at [Le Wagon Paris](https://www.lewagon.com/fr/paris/data-science-course).*

*The **SAT2PLAN** project aims to retrieve a satellite image from Google Maps for a given address or GPS coordinates, then reconstruct maps using a generative antagonistic neural network (GAN).*

## Contexte / Context

Le projet **SAT2PLAN** est une initiative collective dans le cadre de la formation **Data Science & AI** proposée par [Le Wagon Paris](https://www.lewagon.com/fr/paris/data-science-course). Il s'inscrit dans le domaine de la vision par ordinateur et de l'intelligence artificielle, avec pour objectif de créer un système capable de reconstruire des cartes en 2D à partir d'images satellites récupérées sur Google Maps.

-----------------------------------------------------------------------------

*The **SAT2PLAN** project is a group initiative as part of the **Data Science & AI** course offered by [Le Wagon Paris](https://www.lewagon.com/fr/paris/data-science-course). It falls within the field of computer vision and artificial intelligence, with the aim of creating a system capable of reconstructing 2D maps from satellite images retrieved from Google Maps.*


![alt text](https://github.com/Orolol/sat2plan/blob/main/schema/schema-1.jpg?raw=true)


## Caractéristiques / Features

- **Collecte d'images :** Environ 100 000 paires d'images ont été collectées à partir de l'API Google Maps pour alimenter le projet. Ces images sont essentielles pour l'entraînement et l'évaluation du modèle de réseau neuronal antagoniste génératif (GAN).

- **Données géographiques :** Les images collectées couvrent une variété de villes américaines et européennes, avec une vue satellite et une carte d'environ 32 villes. Les données sont réparties de manière à couvrir une zone d'environ 1000 km² autour de chaque ville, offrant ainsi une diversité géographique significative pour le développement du modèle.

- **Nettoyage des données :** Avant d'être utilisées dans le cadre du projet, les données ont été nettoyées. Ce processus comprend la suppression des doublons et des plans non pertinents. Ce nettoyage garantit que seules des données pertinentes et de qualité sont utilisées dans le projet, améliorant ainsi l'efficacité et la précision du modèle GAN.

-----------------------------------------------------------------------------

*- **Image collection :** Around 100,000 pairs of images have been collected from the Google Maps API to feed the project. These images are essential for training and evaluating the generative adversarial neural network (GAN) model.*

*- **Geographical data :** The images collected cover a variety of American and European cities, with a satellite view and map of around 32 cities. The data is distributed so as to cover an area of approximately 1000 km² around each city, providing significant geographical diversity for the development of the model.*

*- **Data cleaning :** Prior to use in the project, a data cleaning process was carried out. This includes the removal of duplicates and irrelevant plans. This cleaning ensures that only relevant, quality data is used in the project, improving the efficiency and accuracy of the GAN model.*

## Réseaux antagonistes génératifs / Generative adversarial networks


![alt text](https://github.com/Orolol/sat2plan/blob/main/schema/schema-2.jpg?raw=true)

Les GAN ont été développés par Ian Goodfellow en 2014 et décrits pour la première fois dans l'article [Generative Adversarial Nets](https://proceedings.neurips.cc/paper_files/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf).

Ils se composent de deux modèles distincts :
- **Le générateur :** Son but est de générer de "fausses" images qui ressemblent aux images d'apprentissage.

- **Le discriminateur :** Son but est d'examiner une image et de déterminer s'il s'agit d'une véritable image d'apprentissage ou d'une fausse image provenant du générateur.

-----------------------------------------------------------------------------

*GANs were developed by Ian Goodfellow in 2014 and first described in the paper [Generative Adversarial Nets](https://proceedings.neurips.cc/paper_files/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf).*

*They consist of two distinct models:*

*- **The generator :** Its aim is to generate 'false' images that resemble the training images.*

*- **The discriminator :** Its aim is to examine an image and determine whether it is a genuine training image or a false image from the generator.*

### Comment fonctionne le GAN / How GAN works
Pendant l'entraînement, le générateur essaie constamment de déjouer le discriminateur en produisant des images de plus en plus fausses, tandis que le discriminateur s'efforce de devenir un meilleur détective et de classer correctement les vraies et les fausses images. L'équilibre de ce jeu est atteint lorsque le générateur produit des faux parfaits qui semblent provenir directement des données d'entraînement, et que le discriminateur doit toujours deviner, avec un degré de confiance de 50 %, si les résultats du générateur sont vrais ou faux

-----------------------------------------------------------------------------

*During training, the generator constantly tries to outwit the discriminator by producing increasingly false images, while the discriminator strives to become a better detective and correctly classify true and false images. The equilibrium of this game is reached when the generator generates perfect fakes that appear to come directly from the training data, and the discriminator must always guess, with 50% confidence, whether the generator's output is real or fake <sup>[1](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)</sup>.*

### Modèles alternatifs / Alternative models
Trois modèles sont utilisés dans ce projet :

- **U-net :** Il se compose d'un chemin contractuel (côté gauche) et d'un chemin expansif (côté droit). Le chemin de contraction suit l’architecture typique d’un réseau convolutif. Il consiste en l'application répétée de deux convolutions 3x3 (convolutions non rembourrées), chacune suivie d'une unité linéaire rectifiée (ReLU) et d'une opération de pooling max 2x2 avec pas de 2 pour le sous-échantillonnage. Chaque étape du chemin expansif consiste en un suréchantillonnage de la carte de fonctionnalités suivi d'une convolution 2x2 (« convolution ascendante ») qui réduit de moitié le nombre de canaux de fonctionnalités, une concaténation avec la carte de fonctionnalités recadrée en conséquence du chemin de contraction, et deux 3x3 convolutions, chacune suivie d'un ReLU. Le recadrage est nécessaire en raison de la perte de pixels de bordure à chaque circonvolution. Au niveau de la couche finale, une convolution 1x1 est utilisée pour mapper chaque vecteur de caractéristiques de 64 composants au nombre souhaité de classes. Au total, le réseau comporte 23 couches convolutives <sup>[2](https://arxiv.org/pdf/1505.04597.pdf)</sup>.

- **UVC-GAN :** Il s'agit d'une méthode améliorée pour effectuer un transfert de style image à image non apparié basé sur un framework CycleGAN. Associé à une nouvelle architecture de générateur hybride UNet-ViT (UNet-Vision Transformer) et à une pré-formation auto-supervisée, il permet d'obtenir des résultats de pointe sur une multitude de benchmarks de transfert de style <sup>[3](https://arxiv.org/pdf/2203.02557.pdf)</sup><sup>[4](https://github.com/ls4gan/uvcgan)</sup>.

- **SAM-GAN :** Ce modèle vise à entraîner les générateurs à apprendre les relations de mappage entre les domaines source et cible. Le modèle SAM-GAN est divisé en deux parties principales : un générateur et un discriminateur, le générateur étant constitué d'un encodeur de contenu, d'un encodeur de style et d'un décodeur. Le discriminateur guide le générateur pendant la formation en apprenant la distribution des images dans les domaines source et cible, ce qui permet au générateur de générer une carte plus réaliste <sup>[5](https://www.mdpi.com/2220-9964/12/4/159)</sup>.

Nous avons eu recours à la librairie PyTorch pour la programmation et le fonctionnement de l'algorithme.


-----------------------------------------------------------------------------

*Three models are used in this project:*

*- **U-net :** It consists of a contracting path (left side) and an expansive path (right side). The contracting path follows the typical architecture of a convolutional network. It consists of the repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling. Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution (“up-convolution”) that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border pixels in every convolution. At the final layer a 1x1 convolution is used to map each 64 component feature vector to the desired number of classes. In total the network has 23 convolutional layers <sup>[2](https://arxiv.org/pdf/1505.04597.pdf)</sup>.*

![alt text](https://vitalab.github.io/article/images/unet/unet.jpg?raw=true)

*- **UVC-GAN :** It is an improved method to perform an unpaired image-to-image style transfer based on a CycleGAN framework. Combined with a new hybrid generator architecture UNet-ViT (UNet-Vision Transformer) and a self-supervised pre-training, it achieves state-of-the-art results on a multitude of style transfer benchmarks <sup>[3](https://arxiv.org/pdf/2203.02557.pdf)</sup> <sup>[4](https://github.com/ls4gan/uvcgan)</sup>.*

![alt text](https://www.catalyzex.com/_next/image?url=https%3A%2F%2Fai2-s2-public.s3.amazonaws.com%2Ffigures%2F2017-08-08%2F905cf9ac767133510c90f5bc7c49bbb147e29ca7%2F2-Figure1-1.png&w=640&q=75)

*- **SAM-GAN :** This model aims to train generators to learn mapping relationships between source and target domains. The SAM-GAN model is divided into two main parts: a generator and a discriminator, where the generator consists of a content encoder, a style encoder, and a decoder. The discriminator guides the generator during training by learning the distribution of images in the source and target domains, thus allowing the generator to generate a more realistic map. <sup>[5](https://www.mdpi.com/2220-9964/12/4/159)</sup>.*


![alt text](https://www.mdpi.com/ijgi/ijgi-12-00159/article_deploy/html/images/ijgi-12-00159-g002.png?raw=true)


We used the PyTorch library to program and run the algorithm.

## Interface graphique / Graphic interface

L'utilisation du modèle se fait par l'intermédiaire d'une interface graphique codé en python avec la librairie Streamlit.
L'utilisateur peut choisir de rentrer l'une des deux options suivantes:
- **Les coordonnées GPS (latitude, longitude)**
- **L'adresse ou le nom du lieu**

Pour la première option, la latitude et la longitude sont envoyé vers l'API qui se charge de récupérer l'image satellite et le plan sur le site Google Maps et affiche l'image satellite sur l'interface pour illustration dans l'onglet "Import Google Maps".

Pour la deuxième option, l'adresse ou le nom du lieu est envoyé vers une fonction du code qui procède à une requête que le site de Google Maps pour récupérer les coordonnées GPS. La suite de la procédure est analogue à la première option.

L'onglet "Cartographie GAN" permet de voir la comparaison entre le résultat obtenu par le réseau de neurones antagoniste génératif (à gauche) et le plan de Google Maps (à droite).

-----------------------------------------------------------------------------

*The model is used via a graphical interface coded in Python with the Streamlit library.*
*The user can choose to enter one of the following two options:*
*- **GPS coordinates (latitude, longitude)***
*- **The address or name of the place***

*For the first option, the latitude and longitude are sent to the API which is responsible for retrieving the satellite image and the map from the Google Maps site and displays the satellite image on the interface for illustration in the "Import Google Maps" tab.*

*For the second option, the address or place name is sent to a code function that queries the Google Maps site to retrieve the GPS coordinates. The rest of the procedure is similar to the first option.*

![alt text](https://github.com/Orolol/sat2plan/blob/main/schema/interface-1.JPG?raw=true)

The "GAN mapping" tab shows a comparison between the results obtained by the generative adversarial neural network (left) and the Google Maps map (right).

![alt text](https://github.com/Orolol/sat2plan/blob/main/schema/interface-2.JPG?raw=true)

## Licence
Ce projet est placé sous la licence MIT. Voir [LICENSE](https://opensource-org.translate.goog/license/mit?_x_tr_sl=en&_x_tr_tl=fr&_x_tr_hl=fr&_x_tr_pto=wapp) pour plus d'informations.

-----------------------------------------------------------------------------

*This project is licensed under the MIT licence. See [LICENSE](https://opensource.org/license/mit) for more information.*
