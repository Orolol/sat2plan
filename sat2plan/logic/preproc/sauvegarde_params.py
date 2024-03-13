import json
import getpass

from sat2plan.logic.configuration.config import Global_Configuration

def export_params_txt():
    """
    Création d'un fichier qui résume les paramètres de calcul en txt
    """

    # Ouverture du fichier
    fichier = open("parametres_code.txt", "w")
    fichier.write(
        "#################################################################\n")
    fichier.write(
        "#################################################################\n")
    fichier.write(
        "#---------------------PARAMETRES DU CALCUL----------------------#\n")
    fichier.write(
        "#################################################################\n")
    fichier.write(
        "#################################################################\n")

    # Nom opérateur
    fichier.write(f"OPERATEUR: {getpass.getuser()}\n")

    # Nom modèle
    fichier.write(f"MODELE: UNET\n")

    # Mode de calcul
    fichier.write(f"DEVICE: {Global_Configuration().device}\n")
    fichier.write(f"TRAIN_DIR: {Global_Configuration().train_dir}\n")
    fichier.write(f"VAL_DIR: {Global_Configuration().val_dir}\n")

    # Hyperparamètres
    fichier.write(f"LEARNING_RATE: {Global_Configuration().learning_rate}\n")
    fichier.write(f"BETA 1: {Global_Configuration().beta1}\n")
    fichier.write(f"BETA 2: {Global_Configuration().beta2}\n")
    fichier.write(f"NB CPU: {Global_Configuration().n_cpu}\n")
    fichier.write(f"BATCH_SIZE: {Global_Configuration().batch_size}\n")
    fichier.write(f"NB_EPOCHS: {Global_Configuration().n_epochs}\n")
    fichier.write(f"SAMPLE_INTERVAL: {Global_Configuration().sample_interval}\n")
    fichier.write(f"IMAGE_SIZE: {Global_Configuration().image_size}\n")
    fichier.write(f"CHANNELS_IMG: {Global_Configuration().channels_img}\n")
    fichier.write(f"STRIDE: {Global_Configuration().stride}\n")
    fichier.write(f"PADDING: {Global_Configuration().padding}\n")
    fichier.write(f"KERNEL_SIZE: {Global_Configuration().kernel_size}\n")
    fichier.write(f"NB_WORKERS: {Global_Configuration().num_workers}\n")
    fichier.write(f"L1_LAMBDA: {Global_Configuration().l1_lambda}\n")
    fichier.write(f"LAMBDA_GP: {Global_Configuration().lambda_gp}\n")

    # Sauvegarde modèle
    fichier.write(f"CHARGEMENT_MODELE: {Global_Configuration().load_model}\n")
    fichier.write(f"SAUVEGARDE_MODELE: {Global_Configuration().save_model}\n")

    # Fichiers
    fichier.write(f"CHECKPOINT_DISC: {Global_Configuration().checkpoint_disc}\n")
    fichier.write(f"CHECKPOINT_GEN: {Global_Configuration().checkpoint_gen}\n")
    fichier.close()
    pass


def ouverture_fichier_json(nom):
    """
    Création d'un fichier json avec nom au choix
    """
    fichier = open(f"{nom}.json", mode="w",encoding='UTF-8')
    return fichier


def export_loss(fichier, epoch, batch, loss_l1, loss_g, loss_d, configuration):
    """
    Export des données dans un fichier .json
    Le terme 'fichier' permet d'indiquer sur quel fichier .json il faut écrire
    Le terme 'epoch' est le numéro de l'epoch
    Le terme 'batch' est le numéro du batch
    Le terme 'loss_l1' est la loss L1
    Le terme 'loss_g' est la loss du générateur
    Le terme 'loss_d' est la loss du discriminateur
    Le terme 'configuration' récupère tous les paramètres et hyperparamètres
    On récupère aussi le nom de l'opérateur
    """

    # Stockage des paramètres dans un dictionnaire
    dictionary = {
        "Epoch": epoch,
        "Batch": batch,
        "Loss_L1": loss_l1,
        "Loss_G": loss_g,
        "Loss_D": loss_d,
        "Device": configuration.device,
        "Train_dir": configuration.train_dir,
        "Val_dir": configuration.val_dir,
        "N_cpu": configuration.n_cpu,
        "Batch_size": configuration.batch_size,
        "N_epochs": configuration.n_epochs,
        "Sample_interval": configuration.sample_interval,
        "Image_size": configuration.image_size,
        "Channels_img": configuration.channels_img,
        "N_workers": configuration.num_workers,
        "L1_lambda": configuration.l1_lambda,
        "Lambda_gp": configuration.lambda_gp,
        "Load_model": configuration.load_model,
        "Save_model": configuration.save_model,
        "Checkpoint_disc": configuration.checkpoint_disc,
        "Checkpoint_gen": configuration.checkpoint_gen,
        "Auteur": getpass.getuser()
    }

    # Mise en forme des dictionnaires dans le fichier
    json_object = json.dumps(dictionary, indent=4)


    # Écriture dans le fichier .json
    fichier.write(json_object)

    pass
