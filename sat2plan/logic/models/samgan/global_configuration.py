import torch


class Global_Configuration:

    def __init__(self):
        # Hyperparamètres
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_dir = "./data/split/train"
        self.val_dir = "./data/split/val"

        self.n_cpu = 6

        self.batch_size = 1
        self.n_epochs = 200
        self.sample_interval = 50

        self.image_size = 512
        self.channels_img = 3

        self.num_workers = 2
        self.l1_lambda = 100
        self.lambda_gp = 10

        self.load_model = False
        self.save_model = True

        self.checkpoint_disc = "disc.pth.tar"
        self.checkpoint_gen = "gen.pth.tar"

    def init(self, **kwargs):
        pass