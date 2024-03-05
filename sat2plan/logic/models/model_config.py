class Configuration:

    def __init__(self):

        #Hyperparamètres
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.n_cpu = 6
        self.batch_size = 16
        self.n_epochs = 5
        self.sample_interval = 10
        self.img_size = 256
        self.channels = 3
        self.latent_dim = 100
        self.n_blocks = 4
        self.stride = 1
        self.padding = 1
        self.kernel_size = 3

    def init(self, **kwargs):
        pass
