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
import torch, torchvision
from torch import nn
from torchvision.utils import make_grid, save_image
from utils import Custom_Dataset, save_checkpoint
from default_lsun_configs import get_default_configs

def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.batch_size = 32
    training.sde = 'vesde'
    training.continuous = True

    # sampling
    sampling = config.sampling
    sampling.solver = 'song'
    sampling.name = 'standard'
    sampling.method = 'pc'
    sampling.predictor = 'reverse_diffusion'
    sampling.corrector = 'langevin'

    # data
    data = config.data
    data.dataset = 'streak_artifact'
    data.root = '/nas/dataset/users/minhyeok/LDCT'
    data.image_size = 128
    data.is_multi = False
    data.is_complex = False
    data.centercrop = 448

    # model
    model = config.model
    model.name = 'ncsnpp'
    model.scale_by_sigma = True
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

from utils import restore_checkpoint, show_samples_gray, clear, clear_color, \
    lambda_schedule_const, lambda_schedule_linear
from physics.ct import CT
from sampling import ReverseDiffusionPredictor, LangevinCorrector
import controllable_generation
import matplotlib.pyplot as plt

def test(config, workdir='/nas/users/minhyeok/CMH/sde'):
    
    config_name = config.sampling.name
    solver = config.sampling.solver
    sde = 'VESDE'
    num_scales = config.model.num_scales
    ckpt_num = 185
    N = num_scales

    root = './samples/CT'

    # Parameters for the inverse problem
    sparsity = 6
    num_proj = 180 // sparsity  # 180 / 6 = 30
    det_spacing = 1.0
    size = config.data.image_size
    det_count = int((size * (2*torch.ones(1)).sqrt()).ceil()) # ceil(size * \sqrt{2})

    schedule = 'linear'
    start_lamb = 1.0
    end_lamb = 0.6

    num_posterior_sample = 6

    if schedule == 'const':
        lamb_schedule = lambda_schedule_const(lamb=start_lamb)
    elif schedule == 'linear':
        lamb_schedule = lambda_schedule_linear(start_lamb=start_lamb, end_lamb=end_lamb)
    else:
        NotImplementedError(f"Given schedule {schedule} not implemented yet!")

    freq = 1

    ckpt_filename = '/nas/users/minhyeok/CMH/sde/result/10_29/size128_ckpt_199_last.pth'
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sde.N = N
    sampling_eps = 1e-5
        
    config.eval.batch_size = 4
    probability_flow = False
    snr = 0.16
    n_steps = 1
    random_seed = 0
    

    sigmas = mutils.get_sigmas(config)
    score_model = mutils.create_model(config)
    data_dir    = '/nas/dataset/users/minhyeok/LDCT'
    transforms_train = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(448),
        torchvision.transforms.Resize(128),
        # torchvision.transforms.RandomCrop(32, padding=4),
        # torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, ), (0.5, ))
        ])
    batch = 4
    train_set = Custom_Dataset(data_dir, transform=transforms_train)
    test_set = Custom_Dataset(data_dir, transform=transforms_train, train=False)
    dl_train = DataLoader(dataset=train_set, batch_size=batch, drop_last=True, shuffle=True)
    dl_test = DataLoader(dataset=test_set, batch_size=batch, drop_last=True, shuffle=True)
    print(len(train_set))

    image, _ = next(iter(dl_train))
    plt.imshow(torchvision.utils.make_grid(image, normalize=True).permute(1,2,0))

    # image, low = next(iter(dl_test))
    
if __name__ == '__main__':
    config = get_config()

    test(config)