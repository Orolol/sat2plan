import torch


class Model_Configuration:

    def __init__(self):

        self.learning_rate = 2e-4
        self.beta1 = 0.5
        self.beta2 = 0.999

        self.stride = 1
        self.padding = 1
        self.kernel_size = 3

    def items(self):
        return self.__dict__.items()

    def init(self, **kwargs):
        pass
