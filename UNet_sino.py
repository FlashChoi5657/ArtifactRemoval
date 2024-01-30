import torch
import pytorch_lightning as pl

from UNet_model import UNet
from torchvision import transforms
from utils import Custom_Dataset, CT_Dataset
from torch.utils.data import DataLoader
from physics.ct import CT

import numpy as np
from torchvision.utils import make_grid, save_image
import random, datetime, os
import matplotlib.pyplot as plt


image_size = 256
batch_size = 64
device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

workdir='/nas/users/minhyeok/CMH/sde'
time_stamp = datetime.datetime.now().strftime('%m_%d')
date_root = f'{workdir}/result/{time_stamp}/sinogram'
if not os.path.exists(date_root):  os.makedirs(date_root, exist_ok=True)

transform = transforms.Compose([
        # transforms.CenterCrop(448),
        transforms.Resize(image_size),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
        ])
train_set = CT_Dataset('/nas/dataset/users/minhyeok/LDCT', transform=transform, train=False)
train_dl = DataLoader(dataset=train_set, batch_size=batch_size, drop_last=True, shuffle=True)
radon_all = CT(img_width=image_size, radon_view=180, circle=False, device=device)
pl.seed_everything(0)

unet = UNet(in_channel=1, dim_feature=64, out_channel=1, bilinear=True).to(device)
unet = torch.nn.DataParallel(unet)
criterion = torch.nn.MSELoss(reduction='sum')
optim = torch.optim.Adam(unet.parameters(), lr=1e-4)

# ckpt
# unet.load_state_dict(torch.load('/nas/users/minhyeok/CMH/sde/result/11_13/sinogram/model_ckpt_150.pth'))

unet = unet.train()

val_loss = np.zeros(500)

for i in range(1, 302):
    loss_unet = list()
    for j, (high, low) in enumerate(iter(train_dl)):
        high = high.to(device)
        low = low.to(device)
        
        h_sinogram = radon_all.A(high)
        l_sinogram = radon_all.A(low)
        
        label = h_sinogram - l_sinogram
        
        prediction = unet(l_sinogram)
        
        optim.zero_grad()
        loss = criterion(prediction, label)
        loss.backward()
        optim.step()
        loss_unet.append(loss.item())
    
    print(i, val_loss[i])
    val_loss[i] = np.mean(loss_unet)
    if (i%100) == 0:
        rand = random.randrange(batch_size - 11)
        grid = make_grid(prediction[rand:rand+9], nrow=3, padding=2, normalize=True)
        save_image(grid, f'{date_root}/prediction_{i}.png')
        plt.plot(val_loss[:i], color='red', label='loss')
        plt.savefig(f'{date_root}/loss_{i}.png')
        torch.save(unet.state_dict(), f'{date_root}/model_ckpt_{i}.pth')

torch.save(unet.state_dict(), f'{date_root}/model_ckpt_last.pth')
print('finished')

