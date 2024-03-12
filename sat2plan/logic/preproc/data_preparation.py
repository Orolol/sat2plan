from torchvision.io import read_image
from torchvision import transforms
import os

#data_path = os.path.join('.', '.', '.', '.', 'data', 'data-1k')

def spliting_image(path):

    images = os.listdir(path)

    X = []
    y = []

    for image in images:
        img_path = os.path.join(path, image)
        img = read_image(img_path)
        X.append(img[:,:,:512])
        y.append(img[:,:,512:])

    return X, y

def resize(image,pourcentage=1):
    """
    Cette fonction accepte en entrée une liste de tenseur d'images.
    La modification de la taille s'effectue à l'aide de la variable "pourcentage" qui est de 50% par défaut
    On obtient en sortie une liste de tenseur d'images
    """
    # Initialisation de la liste de sortie
    image_resize = []

    # Boucle itérative du post-traitement
    for img in image:
        image_resize.append(transforms.Resize(int(img.size()[1] * pourcentage))(img))

    return image_resize
