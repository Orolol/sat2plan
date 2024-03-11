# Script to create the config class for the params of the GAN

import torch
import torch.nn as nn
import torch.optim as optim

class Configuration:

    def __init__(self):
        """
        Declare types but do not instanciate anything
        """
        self.device = torch.device('mps')
        self.learning_rate = 2e-4
        self.betas = (0.5, 0.999)
        self.batch_size = 256
        self.img_size = 64
        self.channels_img = 1
        self.z_dim = 100
        self.num_epochs = 5
        self.features_disc = 64
        self.features_gen = 64

    def init(self, **kwargs):
        pass
