import requests
import os
import kornia
import torch
from torchvision import transforms
from torchvision.utils import save_image

from PIL import Image
import shutil
from sat2plan.scripts.params import API_KEY
from sat2plan.interface.main import pred


def conversion_adresse_coordonnees_gps(adresse):
    """
    Cette fonction convertit une adresse postale ou le nom d'un lieu en coordonnées GPS
    Pour un bon fonctionnement, il est nécessaire d'avoir une clé API Google Maps
    Une commande "input" permet d'entrer l'adresse ou le mieu directement dans le terminal
    """

    # Remplacement des espaces sous le format Google
    adresse = adresse.replace(' ', '%20')

    # Adresse de la requête
    url = f'https://maps.googleapis.com/maps/api/geocode/json?address={adresse}&zoom=17&size=512x512&key={API_KEY}'

    # Récupération du dictionnaire sous format json
    response = requests.get(url).json()

    # Contrôle du dictionnaire pour éviter les erreurs
    if response['status'] == "OK":
        # Latitude
        lat = response['results'][0]['geometry']['location']['lat']

        # Longitude
        lon = response['results'][0]['geometry']['location']['lng']

        return lat, lon

    else:
        return None


def get_images(loc):
    """
    Cette fonction récupère l'image satellite pour des coordonnées GPS données
    Elle stocke l'image satellitaire dans un répertoire "adresse" qui sera située dans le répertoire "data"
    """

    # # Supression du répertoire "adresse" ancien
    # shutil.rmtree('data/adresse', ignore_errors=True)

    # Création du répertoire adresse qui contiendra l'image satellite de l'adresse donnée
    path = os.path.join(os.getcwd(), "data/adresse/", f"{loc[0]}_{loc[1]}")
    os.makedirs(path, exist_ok=True)

    format = ['roadmap', 'satellite']

    repertoire = {}

    for carte in format:
        # Préparation du fichier de sortie
        out = Image.new('RGB', (512, 512))

        # Adresse de la requête
        url = "https://maps.googleapis.com/maps/api/staticmap?center={},{}&zoom=17&size=640x640&maptype={}&style=feature:all%7Celement:labels%7Cvisibility:off&key={}".format(
            loc[0], loc[1], carte, API_KEY)

        # Lecture de l'image
        im = Image.open(requests.get(url, stream=True).raw)

        # Cadrage de l'image
        im = im.crop((0, 0, 512, 512))

        # Collage image dans fichier de sortie
        out.paste(im, (0, 0))
        convert_tensor = transforms.ToTensor()
        y = convert_tensor(out)
        y_saturated: torch.Tensor = kornia.enhance.adjust_contrast(
            y, 0.3)

        # Sauvegarde du fichier de sortie
        repertoire[carte] = os.path.join(path, f"adresse_{carte}.png")
        save_image(y_saturated, repertoire[carte], normalize=True)
        # y_saturated.save(repertoire[carte])
    pred(path)
    repertoire['generee'] = os.path.join(path, f"adresse_generee.png")

    return repertoire


def test_api():
    adresse = "tour eiffel"
    loc = conversion_adresse_coordonnees_gps(adresse)
    print(loc)
    repertoire = get_images(loc)
    print(repertoire)
