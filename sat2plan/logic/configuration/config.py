import torch


class Model_Configuration:

    def __init__(self):
        self.learning_rate_ViT = 1e-3
        self.learning_rate_G = 1e-4
        self.learning_rate_D = 1e-4
        self.learning_rate = 1e-4
        self.beta1 = 0.5
        self.beta2 = 0.999

        self.stride = 1
        self.padding = 1
        self.kernel_size = 3

    def items(self):
        return self.__dict__.items()

    def init(self, **kwargs):
        pass


class Global_Configuration:

    def __init__(self):
        # Hyperparamètres
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_dir = "./data/split/train"
        self.val_dir = "./data/split/val"
        self.data_bucket = 'data-full'

        self.n_cpu = 6

        self.batch_size = 16
        self.n_epochs = 200
        self.sample_interval = 10

        self.image_size = 256
        self.channels_img = 3

        self.num_workers = 2
        self.l1_lambda = 10.0
        self.lambda_gp = 0.1

        self.load_model = True
        self.save_model = True

        self.checkpoint_disc = "disc.pth.tar"
        self.checkpoint_gen = "gen.pth.tar"

    def init(self, **kwargs):
        pass
