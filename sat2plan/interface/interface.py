import streamlit as st
import time
import os
from streamlit_image_comparison import image_comparison
from sat2plan.api.api import conversion_adresse_coordonnees_gps, get_images


def barre_defilement(texte, temps):
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


def comparaison_image(satellite, cartographie):
    """
    Cette fonction permet de facilier la comparaison entre une image satellite et une cartographie générée par le GAN
    La variable "satellite" est le chemin d'accès à l'image satellite
    La variable "cartographie" est le chemin d'accès à la cartographie générée
    """
    # Barre de défilement de la cartographie GAN
    barre_defilement("Traitement image satellite. Merci de patientier", 0.01)

    # Fonction
    # st.write("# Cartographie GAN")
    image_comparison(img1=satellite,
                     label1="Image générée par le GAN",
                     img2=cartographie,
                     label2='Cartographie',
                     starting_position=50)


def image_cote_a_cote(satellite, cartographie):
    """
    Autre façon de présenter les résultats
    Les images sont côte à côte
    """

    # Définition des colonnes
    col1, col2 = st.columns(2)

    # Colonne de gauche
    with col1:
        st.header(f"Image générée par le GAN")
        st.image(satellite)

    # Colonne de droite
    with col2:
        st.header(f"Cartographie")
        st.image(cartographie)


def onglet(titre, icone):
    """
    Cette fonction configure l'onglet du navigateur internet
    La variable "titre" configure le titre de l'onglet
    La variable "icone" est le lien d'accès au fichier icone
    """

    st.set_page_config(page_title=titre, page_icon=icone,
                       initial_sidebar_state="collapsed")


def coordonnee_GPS(lat, lon):
    """
    Cette fonction récupère la latitude et la longitude pour récupérer l'image satellite
    et fait appel au réseau de neurones pour récupérer la cartographie produite.
    """

    if (float(lat) >= -90 and float(lat) <= 90) and (float(lon) >= -180 and float(lon) <= 180):
        loc = (lat, lon)
        # Récupération de l'image satellite
        satellite = get_images(loc)['satellite']

        # Image satellite récupérée
        barre_defilement(
            "Récupération image satellite sur Google Maps. Merci de patientier", 0.001)
        # st.write("# Image satellite")
        st.image(satellite)

        # Traitement par le réseau de neurones
        ###
        # Appel fonction dédié au traitement cartographique
        ###
        ###

        # Récupération de l'arborescence pour la cartographie
        # cartographie =
        cartographie = get_images(loc)['roadmap']
        generee = get_images(loc)['generee']

        return generee, cartographie
    else:
        st.error("Vos coordonnées sont absurdes, revoyez votre géographie", icon="🚨")
        st.error("La latitude est comprise entre -90 et +90", icon="🌐")
        st.error("La longitude est comprise entre -180 et +180", icon="🌐")

        return None, None


def adresse(adresse):
    """
    Cette fonction fait appel à une autre fonction qui convertit une adresse postale ou le nom d'un lieu en coordonnées GPS
    Elle retourne la latitude et la longitude
    """

    # Appel de la fonction
    loc = conversion_adresse_coordonnees_gps(adresse)

    try:
        return loc[0], loc[1]
    except:
        st.error(
            "Veuillez vérifier les paramètres d'entrées (adresse, nom de lieu, orthographe ...)", icon="🚨")
        return None


def main():
    """
    Fonction principale de l'interface
    """
    # Configuration de l'onglet du navigateur
    titre = "SAT2PLAN"
    icone = os.path.join(
        os.getcwd(), "sat2plan/interface/icone", "satellite_lewagon.jpg")
    icone_inverse = os.path.join(
        os.getcwd(), "sat2plan/interface/icone", "satellite_lewagon_inverse.jpg")
    onglet(titre, icone)

    col1, col2, col3 = st.columns(3)

    # Logo site internet
    with col1:

        st.image(icone, width=200)

    # Titre
    with col2:

        st.title("""
                SAT2PLAN
                """)

    with col3:

        st.image(icone_inverse, width=200)

    # Descriptif
    st.write("""
            Démonstration d'un réseaux antagonistes génératifs (GAN) qui transforme les images satellites en cartographie
            """)

    # Paramètres d'entrée pour le relevé de l'image satellite
    choix = st.radio("type d'entrée", [
                     "Coordonnées GPS", "Adresse / Nom d'un lieu"])

    if choix == "Coordonnées GPS":
        # Entrée utilisateur
        lat = st.text_input("Latitude")
        lon = st.text_input("Longitude")

        if lat and lon:
            # Création des onglets
            tab1, tab2 = st.tabs(['Import Google Maps', 'Cartographie GAN'])

            with tab1:
                # Récupération des arborescences
                satellite, cartographie = coordonnee_GPS(lat, lon)

            with tab2:
                # Illustration comparative
                if satellite == None or cartographie == None:
                    st.error(
                        "GAN en attente de cartographie satellitaire", icon="🧠")
                else:
                    comparaison_image(satellite, cartographie)
    else:
        # Entrée utilisateur
        entree_adresse = st.text_input("Adresse / Nom d'un lieu")

        if entree_adresse:
            # Création des onglets
            tab1, tab2 = st.tabs(['Import Google Maps', 'Cartographie GAN'])

            with tab1:
                # Conversion adresse en coordonnées GPS
                loc = adresse(entree_adresse)

                # Récupération des arborescences
                if loc != None:
                    satellite, cartographie = coordonnee_GPS(loc[0], loc[1])

            with tab2:
                # Illustration comparative
                if loc != None:
                    comparaison_image(satellite, cartographie)
                else:
                    st.error(
                        "GAN en attente de cartographie satellitaire", icon="🧠"
                    )


if __name__ == '__main__':
    main()
