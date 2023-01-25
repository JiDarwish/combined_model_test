import os, re, torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms
from skimage.color import rgb2gray
from shutil import copytree, copy
from pix import rgb_to_lab, lab_to_rgb
from pix_pytorch import make_dataloaders
from pix2pix_model import Generator, Discriminator, init_weights, DiscriminatorLoss
from tqdm import tqdm

# parameters related to the images, model and training
IM_SIZE = 256
CROP_SIZE = 256
BATCH_SIZE = 64

MODEL = 'Pix2Pix_RGB'

# dataset directory
dir = 'data/coco/train2014/class/'

# directories to save logs and checkpoints to restart training
dir_summary = 'checkpoints_colorizer_3'
dir_model = os.path.join(dir_summary, MODEL)
log_path = os.path.join(dir_model, 'logs')
checkpoint_dir = 'checkpoints_colorizer'

# directories to save images during training
img_dir = os.path.join(dir_model, 'images_' + str(BATCH_SIZE))
img_train = os.path.join(img_dir, 'from_training')
img_test = os.path.join(img_dir, 'from_test')

device = 'cuda'


def lab_to_rgb_pytorch(L, ab):
    """lab_to_rgb takes a numpy stack [B, H, W ,C] as input and
     return a numpy stack in the same format"""
    L = L.permute(0, 2, 3, 1).cpu().numpy()
    ab = ab.permute(0, 2, 3, 1).detach().cpu().numpy()
    return torch.from_numpy(lab_to_rgb(L, ab)).permute(0, 3, 1, 2)


generator = make_dataloaders(batch_size=BATCH_SIZE, im_size=IM_SIZE, crop_size=CROP_SIZE, split='Train',
                             paths='data/coco/train2014', n_workers=0)
track_train = make_dataloaders(batch_size=5, im_size=IM_SIZE, crop_size=CROP_SIZE, split='Test',
                               paths='data/coco/val2014', n_workers=0, shuffle=False)
track_test = make_dataloaders(batch_size=5, im_size=IM_SIZE, crop_size=CROP_SIZE, split='Test', paths='data/coco/test2014',
                              n_workers=0, shuffle=False)

net_G = Generator(3).to(device)
net_D = Discriminator().to(device)

opt_G = optim.Adam(net_G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(net_D.parameters(), lr=2e-4, betas=(0.5, 0.999))

# load the latest model if it finds checkpoint files in the checkpoint directory
if os.listdir(checkpoint_dir):
    nums = [int(re.split('\-|\.', f)[1]) for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    cpkt = torch.load(os.path.join(checkpoint_dir, 'cp-'+str(max(nums))+'.pth'), map_location=device)
    net_G.load_state_dict(cpkt['G_state_dict'])
    net_D.load_state_dict(cpkt['D_state_dict'])
    opt_G.load_state_dict(cpkt['optimizerG_state_dict'])
    opt_D.load_state_dict(cpkt['optimizerD_state_dict'])
    epoch = cpkt['epoch']
    loss_G = cpkt['loss_G']
    loss_D = cpkt['loss_D']
    net_G.train()
    net_D.train()
    initial_epoch = epoch+1
else:
    net_G = net_G.apply(init_weights).train()
    net_D = net_D.apply(init_weights).train()
    initial_epoch = 0
print(initial_epoch)


track_train_batch = next(iter(track_train))
track_test_batch = next(iter(track_test))

GANcriterion = DiscriminatorLoss(device)
criterion = nn.L1Loss()
lambda1 = 100.
writer = SummaryWriter(log_dir=log_path)

for epoch in range(initial_epoch, 100):
    running_loss_D = 0.0
    running_loss_G = 0.0
    for i, data in tqdm(enumerate(generator)):
        L, ab = data[0]['L'].to(device), data[0]['ab'].to(device)
        fake_color = net_G(L).cuda()
        real_image = torch.cat([L, ab], dim=1).cuda()

        fake_image = fake_color.cuda()
        rgb = data[1].to(device)

        # train discriminator
        opt_D.zero_grad()
        # train on real images
        real_preds = net_D(real_image).cuda()
        loss_D_real = GANcriterion(real_preds, True).cuda()
        # train on fake images
        fake_preds = net_D(fake_image.detach()).cuda()
        loss_D_fake = GANcriterion(fake_preds, False).cuda()
        # total loss for D
        loss_D = ((loss_D_fake + loss_D_real) * 0.5).cuda()
        loss_D.backward()
        opt_D.step()

        # train generator
        opt_G.zero_grad()
        # train G using GAN criterion
        fake_preds = net_D(fake_image).cuda()
        loss_G_GAN = GANcriterion(fake_preds, True).cuda()

        # cycle GAN _ same training as for autoencoder times hyperparameter
        loss_G_L1 = (criterion(fake_color, rgb) * lambda1).cuda()
        # total loss for G
        loss_G = (loss_G_GAN + loss_G_L1).cuda()
        loss_G.backward()
        opt_G.step()

        running_loss_D += loss_D.item()
        running_loss_G += loss_G.item()

    running_loss_D = running_loss_D / (i + 1)
    running_loss_G = running_loss_G / (i + 1)
    writer.add_scalar('loss_D', running_loss_D, epoch)
    writer.add_scalar('loss_G', running_loss_G, epoch)

    # print statistics [epoch, number of steps, loss_G, loss_D]
    print('[%d, %5d] loss: %.3f %.3f' %
          (epoch, i + 1, running_loss_G, running_loss_D))

    checkpoint_path = os.path.join(checkpoint_dir, 'cp-{}.pth'.format(epoch))
    torch.save({'epoch': epoch,
                'G_state_dict': net_G.state_dict(),
                'D_state_dict': net_D.state_dict(),
                'optimizerG_state_dict': opt_G.state_dict(),
                'optimizerD_state_dict': opt_D.state_dict(),
                'loss_G': loss_G,
                'loss_D': loss_D
                }, checkpoint_path)

    color_train = net_G(track_train_batch[0]['L'].to(device)) * 255
    color_test = net_G(track_test_batch[0]['L'].to(device)) * 255

    color_train_grid = vutils.make_grid(color_train.to(device), padding=2, normalize=True, nrow=5).cpu()
    color_test_grid = vutils.make_grid(color_test.to(device), padding=2, normalize=True, nrow=5).cpu()

    writer.add_image('train_images', color_train_grid, epoch)
    writer.add_image('test_images', color_test_grid, epoch)

    train_path = os.path.join(img_train, 'img-{}.png'.format(epoch))
    test_path = os.path.join(img_test, 'img-{}.png'.format(epoch))
    save_image(color_train_grid, train_path)
    save_image(color_test_grid, test_path)

    plt.figure(figsize=(16, 16))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(color_test_grid, (1, 2, 0)))

writer.close()
