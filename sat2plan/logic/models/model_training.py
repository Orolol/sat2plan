from model_building import Generator, Discriminator, weights_init_normal
from model_config import Configuration

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import os

#Import des hyperparamètres
CFG = Configuration()

device = CFG.device
train_dir = CFG.train_dir
val_dir = CFG.val_dir

learning_rate = CFG.learning_rate
beta1 = CFG.beta1
beta2 = CFG.beta2

n_cpu = CFG.n_cpu

batch_size = CFG.batch_size
n_epochs = CFG.n_epochs
sample_interval = CFG.sample_interval

image_size = CFG.image_size
channels_img = CFG.channels_img

stride = CFG.stride
padding = CFG.padding
kernel_size = CFG.kernel_size

num_workers = CFG.num_workers
l1_lambda = CFG.l1_lambda
lambda_gp = CFG.lambda_gp

load_model = CFG.load_model
save_model = CFG.save_model

checkpoint_disc = CFG.checkpoint_disc
checkpoint_gen = CFG.checkpoint_gen

from_scratch = True
cuda = True if torch.cuda.is_available() else False


netD = Discriminator(in_channels=3).cuda()
netG = Generator(in_channels=3).cuda()
OptimizerD = torch.optim.Adam(netD.parameters(),lr=learning_rate,betas=(beta1,beta2))
OptimizerG = torch.optim.Adam(netG.parameters(),lr=learning_rate,betas=(beta1,beta2))
BCE_Loss = nn.BCEWithLogitsLoss()
L1_Loss = nn.L1Loss()

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

torch.backends.cudnn.benchmark = True
Gen_loss = []
Dis_loss = []

os.makedirs("images", exist_ok=True)
os.makedirs("data", exist_ok=True)

dataloader = torch.utils.data.DataLoader(
    datasets.ImageFolder("data/", transform=transforms.Compose([
        # transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])),
    batch_size=batch_size,
    shuffle=True,
)


"""if load_model:
    load_checkpoint(
        CHECKPOINT_GEN,netG,OptimizerG,LEARNING_RATE
    )
    load_checkpoint(
        CHECKPOINT_DISC,netD,OptimizerD,LEARNING_RATE
    )"""

""" train_dataset = Satellite2Map_Data(root=TRAIN_DIR)
train_dl = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=True)
# G_Scaler = torch.cuda.amp.GradScaler()
# D_Scaler = torch.cuda.amp.GradScaler()
val_dataset = Satellite2Map_Data(root=VAL_DIR)
val_dl = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=True) """


""" for epoch in range(NUM_EPOCHS):
    train(
        netG, netD, train_dl, OptimizerG, OptimizerD, L1_Loss, BCE_Loss
    )
    #Generator_loss.append(g_loss.item())
    #Discriminator_loss.append(d_loss.item())
    if SAVE_MODEL and epoch%50==0:
        save_checkpoint(netG, OptimizerG, filename=CHECKPOINT_GEN)
        save_checkpoint(netD, OptimizerD, filename=CHECKPOINT_DISC)
    if epoch%2==0:
        save_some_examples(netG, val_dl, epoch, folder="evaluation") """


for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            sat = F.interpolate(imgs[:, :, :, :512],
                                size=(image_size, image_size)).cuda()
            plan = F.interpolate(imgs[:, :, :, 512:],
                                 size=(image_size, image_size)).cuda()

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(
                1.0), requires_grad=False).cuda()
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(
                0.0), requires_grad=False).cuda()

            # Configure input
            real_imgs = plan

            ############## Train Discriminateur ##############
            #with torch.cuda.amp.autocast():
            y_fake = netG(sat)
            D_real = netD(sat, real_imgs)
            D_real_loss = BCE_Loss(D_real, torch.ones_like(D_real))
            D_fake = netD(sat,y_fake.detach())
            D_fake_loss = BCE_Loss(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss)/2

            netD.zero_grad()
            Dis_loss.append(D_loss.item())
            D_loss.backward()
            #D_Scaler.scale(D_loss).backward()
            OptimizerD.step()
            #D_Scaler.step(OptimizerD)
            #D_Scaler.update()

            ############## Train Generateur ##############
            #with torch.cuda.amp.autocast():
            D_fake = netD(sat, y_fake)
            G_fake_loss = BCE_Loss(D_fake, torch.ones_like(D_fake))
            L1 = L1_Loss(y_fake,real_imgs) * l1_lambda
            G_loss = G_fake_loss + L1

            OptimizerG.zero_grad()
            Gen_loss.append(G_loss.item())
            G_loss.backward()
            #G_Scaler.scale(G_loss).backward()
            #G_Scaler.step(OptimizerG)
            OptimizerG.step()
            #G_Scaler.update()


            print(
               "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch+1, n_epochs, i+1, len(dataloader), D_loss.item(), G_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                concatenated_images = torch.cat(
                    (y_fake[:-5], sat[:-5], real_imgs[:-5]), dim=2)

                save_image(concatenated_images, "images/%d.png" %
                       batches_done, nrow=5, normalize=True)
