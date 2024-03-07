#!/bin/bash

# Mise à jour du système
sudo apt-get update
sudo apt-get upgrade -y

# Installation de Python 3.8 et pip
sudo apt-get install -y git python3-pip python3-pip python3-dev python3-opencv libglib2.0-0 make gsutil

# Installation de CUDA Toolkit 11.1
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2004-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4 cuda nvidia-cuda-toolkit

#gcloud init
sudo apt-get -y install apt-transport-https ca-certificates gnupg curl
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update && sudo apt-get -y install google-cloud-cli




# Installation de PyTorch avec CUDA 11.1
pip3 install --upgrade pip
pip3 install . 
pip install --upgrade numpy

cp .env.sample .env

# Vérification de l'installation
python3 -c "import torch; print(torch.cuda.is_available())"