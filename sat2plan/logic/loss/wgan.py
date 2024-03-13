import torch
from torch.autograd import Variable
from torch import autograd


def calculate_gradient_penalty(real_images, fake_images):
    eta = torch.FloatTensor(self.batch_size, 1, 1, 1).uniform_(0, 1)
    eta = eta.expand(self.batch_size, real_images.size(
        1), real_images.size(2), real_images.size(3))
    if self.cuda:
        eta = eta.cuda(self.cuda_index)
    else:
        eta = eta

    interpolated = eta * real_images + ((1 - eta) * fake_images)

    if self.cuda:
        interpolated = interpolated.cuda(self.cuda_index)
    else:
        interpolated = interpolated

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = self.D(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(
                                  prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                  prob_interpolated.size()),
                              create_graph=True, retain_graph=True)[0]

    # flatten the gradients to it calculates norm batchwise
    gradients = gradients.view(gradients.size(0), -1)

    grad_penalty = ((gradients.norm(2, dim=1) - 1)
                    ** 2).mean() * self.lambda_term
    return grad_penalty
