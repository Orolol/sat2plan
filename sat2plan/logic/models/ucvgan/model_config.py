import torch


class Model_Configuration:

    def __init__(self):

        self.learning_rate_G = 1e-4
        self.learning_rate_D = 1e-4
        self.beta1 = 0.5
        self.beta2 = 0.999

        self.stride = 1
        self.padding = 1
        self.kernel_size = 3

    def items(self):
        return self.__dict__.items()

    def init(self, **kwargs):
        pass
