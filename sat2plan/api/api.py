import requests
import os
from PIL import Image
import shutil
# from sat2plan.scripts.params import API_KEY

def coordonnees_gps(ville):
    """
    Cette fonction se base sur l'application du Wagon pour obtenir la lattitude et la longitude d'une ville donnée
    """

    # Adresse URL de l'application du Wagon
    url = f'https://weather.lewagon.com/geo/1.0/direct?q={ville}'

    # Récupération du dictionnaire sous format json
    response = requests.get(url).json()

    # Latitude
    lat = response[0]['lat']

    # Longitude
    lon = response[0]['lon']

    return lat, lon


def conversion_adresse_coordonnees_gps():
    """
    Cette fonction convertit une adresse postale ou le nom d'un lieu en coordonnées GPS
    Pour un bon fonctionnement, il est nécessaire d'avoir une clé API Google Maps
    Une commande "input" permet d'entrer l'adresse ou le mieu directement dans le terminal
    """

    # Entrée de l'adresse postale ou du nom d'un lieu et remplacement des espaces sous le format Google
    adresse = input("Entrer une adresse ou le nom d'un lieu:")
    adresse = adresse.replace(' ','%20')

    # Adresse de la requête
    url = f'https://maps.googleapis.com/maps/api/geocode/json?address={adresse}&key={API_KEY}'

    # Récupération du dictionnaire sous format json
    response = requests.get(url).json()

    # Latitude
    lat = response['results'][0]['geometry']['location']['lat']

    # Longitude
    lon = response['results'][0]['geometry']['location']['lng']

    return lat, lon


def get_images(loc):
    """
    Cette fonction récupère l'image satellite pour des coordonnées GPS données
    Elle stocke l'image satellitaire dans un répertoire "adresse" qui sera située dans le répertoire "data"
    """

    # Préparation du fichier de sortie
    out = Image.new('RGB', (512, 512))

    # Adresse de la requête
    url = "https://maps.googleapis.com/maps/api/staticmap?center={},{}&zoom=17&size=1280x1280&maptype=satellite&style=feature:all%7Celement:labels%7Cvisibility:off&key={}".format(
            loc[0], loc[1], API_KEY)

    # Lecture de l'image
    im = Image.open(requests.get(url, stream=True).raw)

    # Cadrage de l'image
    im = im.crop((0, 0, 512, 512))

    # Collage image dans fichier de sortie
    out.paste(im, (0, 0))

    ## Préparation du répertoire
    # Changement de répertoire - Direction data
    os.chdir('../../data')
    repertoire_data = os.getcwd()

    # Supression du répertoire "adresse" ancien
    shutil.rmtree('adresse',ignore_errors=True)

    # Création du répertoire adresse qui contiendra l'image satellite de l'adresse donnée
    path = os.path.join(repertoire_data,"adresse")
    os.mkdir(path, 0o777)
    os.chdir('adresse')

    # Sauvegarde du fichier de sortie
    repertoire = os.path.join(os.getcwd(), "adresse.jpg")
    out.save(repertoire)
    return repertoire


loc = conversion_adresse_coordonnees_gps()
repertoire = get_images(loc)
print(repertoire)
