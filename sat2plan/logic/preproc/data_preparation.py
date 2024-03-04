import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torch import tensor
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
