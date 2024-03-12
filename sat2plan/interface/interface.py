import streamlit as st
import time
import os
from pathlib import Path
from PIL import Image
from streamlit_image_comparison import image_comparison
# from sat2plan.api.api import conversion_adresse_coordonnees_gps, get_images


# test



def barre_defilement(texte,temps):
    """
    Cette fonction est en charge des barres de défilement pour faire patienter l'utilisateur
    Elle prend en argement le texte à afficher et le temps de défilement
    """

    # Initialisation de la barre
    my_bar = st.progress(0, texte)

    # Boucle d'itération pour le défilement
    for percent_complete in range(100):
        time.sleep(temps)
        my_bar.progress(percent_complete + 1, texte)

    # Fin du défilement
    time.sleep(1)
    my_bar.empty()


def comparaison_image(satellite,cartographie):
    """
    Cette fonction permet de facilier la comparaison entre une image satellite et une cartographie générée par le GAN
    La variable "satellite" est le chemin d'accès à l'image satellite
    La variable "cartographie" est le chemin d'accès à la cartographie générée
    """
    # Fonction
    st.write("# Comparaison de l'image satellite et de la cartographie issue de l'IA")
    image_comparison(img1=satellite,
                    label1="Image satellite",
                    img2=cartographie,
                    label2='Cartographie',
                    starting_position=50)


def image_cote_a_cote(satellite,cartographie):
    """
    Autre façon de présenter les résultats
    Les images sont côte à côte
    """

    # Définition des colonnes
    col1, col2 = st.columns(2)

    # Colonne de gauche
    with col1:
        st.header(f"Image satellite")
        st.image(satellite)

    # Colonne de droite
    with col2:
        st.header(f"Cartographie")
        st.image(cartographie)


def onglet(titre,icone):
    """
    Cette fonction configure l'onglet du navigateur internet
    La variable "titre" configure le titre de l'onglet
    La variable "icone" est le lien d'accès au fichier icone
    """

    st.set_page_config(page_title=titre,page_icon=icone,initial_sidebar_state="collapsed")


def coordonnee_GPS(lat,lon):
    """
    Cette fonction récupère la latitude et la longitude pour récupérer l'image satellite
    et fait appel au réseau de neurones pour récupérer la cartographie produite.
    """
    loc = (lat,lon)
    # Récupération de l'image satellite
    # satellite = get_images(loc)
    satellite = '/home/louishenri/code/TsaoTsao1/08-Projet/sat2plan/data/adresse/adresse.jpg'

    # Image satellite récupérée
    barre_defilement("Récupération image satellite sur Google Maps. Merci de patientier",0.001)
    st.write("# Image satellite récupérée sur Google Maps")
    st.image(satellite)

    # Traitement par le réseau de neurones
    ###
    ### Appel fonction dédié au traitement cartographique
    ###

    # Récupération des arborescences
    cartographie = '/home/louishenri/code/TsaoTsao1/08-Projet/sat2plan/data/adresse/adresse.jpg'
    barre_defilement("Traitement image satellite. Merci de patientier",0.001)

    return satellite, cartographie


def adresse():
    """
    Cette fonction fait appel à une autre fonction qui convertit une adresse postale ou le nom d'un lieu en coordonnées GPS
    Elle retourne la latitude et la longitude
    """

    # Appel de la fonction
    loc = conversion_adresse_coordonnees_gps()

    return loc[0], loc[1]

# Configuration de l'onglet du navigateur
titre = "SAT2PLAN"
icone = os.path.join(os.getcwd(),"icone","satellite_lewagon.png")
onglet(titre,icone)

col1, col2 = st.columns(2)

# Logo site internet
with col1:

    st.image(icone,width=200)

# Titre
with col2:

    st.title("""
            SAT2PLAN
            """)


st.write("""
        Cette page internet est une démonstration d'un réseau de neurones qui transforme les images satellites en cartographie
         """)

# Paramètres d'entrée pour le relevé de l'image satellite
choix = st.radio("type d'entrée",['coordonnées GPS','adresse'])

if choix == "coordonnées GPS":

    # Entrée utilisateur
    lat = st.text_input("Latitude")
    lon = st.text_input("Longitude")

    if lat and lon:

        # Récupération des arborescences
        satellite , cartographie = coordonnee_GPS(lat,lon)

        # Illustration comparative
        comparaison_image(satellite,cartographie)

else:

    # Entrée utilisateur
    adresse = st.text_input("Adresse")

    if adresse:
        # Conversion adresse en coordonnées GPS
        # loc = adresse()
        loc = (0,0)

        # Récupération des arborescences
        satellite , cartographie = coordonnee_GPS(loc[0],loc[1])

        # Illustration comparative
        comparaison_image(satellite,cartographie)
