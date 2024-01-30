import os, datetime
import numpy as np

from models import ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
# import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import datasets

import sde_lib
from absl import flags
import torch
from torch import nn
from torchvision.utils import make_grid, save_image
from utils import Custom_Dataset, save_checkpoint, restore_checkpoint, CT_Dataset
from default_lsun_configs import get_default_configs
import matplotlib.pyplot as plt

def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.name = 'size256_Bone_1000scales'
    training.batch_size = 16
    training.sde = 'vesde'
    training.snapshot_sampling = False
    training.continuous = True
    training.epochs = 152
    training.retrain = False

    # sampling
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'reverse_diffusion'
    sampling.corrector = 'langevin'

    # data
    data = config.data
    data.dataset = 'streak_artifact'
    data.root = '/ssd1/dataset'
    data.image_size = 256
    data.is_multi = False
    data.is_complex = False
    data.center_crop = 448

    # model
    model = config.model
    model.name = 'ncsnpp'
    model.scale_by_sigma = True
    model.num_scales = 1000
    model.ema_rate = 0.999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'none'
    model.progressive_input = 'residual'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.
    model.fourier_scale = 16
    model.conv_size = 3

    return config


def train(config, workdir='/nas/users/minhyeok/CMH/sde'):

    score_model = mutils.create_model(config) # model load
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate) # moving average : 지역적인 평균 , E는 최근에 높은 가중치를 준다.
    optimizer = losses.get_optimizer(config, score_model.parameters()) 
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0) 
    initial_step = int(state['step'])

    # continue learning?
    if config.training.retrain:
        ckpt_filename = '/nas/users/minhyeok/CMH/sde/result/10_25/size128_ckpt_100.pth'
        state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True, skip_optimizer=True)

    # make folder
    time_stamp = datetime.datetime.now().strftime('%m_%d')
    date_root = f'{workdir}/result/{time_stamp}'
    if not os.path.exists(date_root):  os.makedirs(date_root, exist_ok=True)
    
    # data load
    transform_train = transforms.Compose([
            # transforms.CenterCrop(config.data.center_crop),
            transforms.Resize(config.data.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, ), (0.5, )) # [-1, 1]
            ])
    train_set = CT_Dataset(config.data.root, transform=transform_train, ww=1800, wl=400)
    train_dl = DataLoader(dataset=train_set, batch_size=config.training.batch_size, drop_last=True, shuffle=True)

    scaler = datasets.get_data_scaler(config) # lambda x: x
    inverse_scaler = datasets.get_data_inverse_scaler(config) # lambda x: x

    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5

    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                        reduce_mean=reduce_mean, continuous=continuous,
                                        likelihood_weighting=likelihood_weighting)

    val_loss = np.zeros(config.training.epochs)

    if config.training.snapshot_sampling:
        sampling_shape = (config.training.batch_size, config.data.num_channels,
                            config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    for epoch in range(1, config.training.epochs):
        val_loss_l = list()
        for step, (batch, _) in enumerate(train_dl, start=1):
            batch = scaler(batch.to(config.device))
            loss = train_step_fn(state, batch)
            if step % config.training.log_freq == 0:
                print("epcoh: %dth, step: %d, training_loss: %.8e" % (epoch, step, loss.item()))
            val_loss_l.append(loss.item())
        val_loss[epoch] = np.mean(val_loss_l)
        
        # if config.training.snapshot_sampling:
        #     if epoch % 50 == 0:
        #         ema.store(score_model.parameters())
        #         ema.copy_to(score_model.parameters())
        #         sample, n = sampling_fn(score_model)
        #         ema.restore(score_model.parameters())

        #         nrow = int(np.sqrt(sample.shape[0]))
        #         image_grid = make_grid(sample, nrow, padding=2)
        #         sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                
        #         save_image(image_grid, f'{date_root}/sample_{epoch}.png')
        if epoch % 50 == 0:    
            save_checkpoint(f'{date_root}/{config.training.name}_ckpt_{epoch}.pth', state)
            print('check point')
            plt.plot(val_loss[:epoch], color='red', label='loss')
            plt.legend()
            plt.title('Loss')
            plt.savefig(f'{date_root}/{config.training.name}_Loss.png')
    save_checkpoint(f'{date_root}/{config.training.name}_ckpt_last.pth', state)
    print('all finished')
    
if __name__ == '__main__':
    config = get_config()
    workdir = '/nas/users/minhyeok/CMH/sde'
    train(config, workdir)