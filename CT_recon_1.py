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
from utils import Custom_Dataset, save_checkpoint
from default_lsun_configs import get_default_configs

def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.batch_size = 30
    training.sde = 'vesde'
    training.continuous = True

    # sampling
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'reverse_diffusion'
    sampling.corrector = 'langevin'

    # data
    data = config.data
    data.dataset = 'streak_artifact'
    data.root = '/nas/dataset/users/minhyeok/recon'
    data.image_size = 128
    data.is_multi = False
    data.is_complex = False

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

from utils import restore_checkpoint, show_samples_gray, clear, clear_color, lambda_schedule_const, lambda_schedule_linear
from physics.ct import CT
from sampling import ReverseDiffusionPredictor, LangevinCorrector
import controllable_generation
import matplotlib.pyplot as plt

def test(config, workdir='/nas/users/minhyeok/CMH/sde'):
    
    solver = 'eend'
    config_name = 'MCG'
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

    ckpt_filename = '/nas/users/minhyeok/CMH/sde/result/10_02/ckpt_500.pth'
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sde.N = N
    sampling_eps = 1e-5
        
    config.eval.batch_size = 6
    probability_flow = False
    snr = 0.16
    n_steps = 1
    random_seed = 0
    
    sigmas = mutils.get_sigmas(config)
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    time_stamp = datetime.datetime.now().strftime('%m_%d')
    date_root = f'{workdir}/result/{time_stamp}/{config_name}'
    if not os.path.exists(date_root):  os.makedirs(date_root, exist_ok=True)
    
    state = dict(step=0, model=score_model, ema=ema)

    state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True, skip_optimizer=True)
    ema.copy_to(score_model.parameters())
    
    transform = transforms.Compose([
            # transforms.CenterCrop(config.center_crop),
            transforms.Resize(config.data.image_size),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
            ])
    test_set = Custom_Dataset(config.data.root, transform=transform, train=False)
    test_dl = DataLoader(dataset=test_set, batch_size=config.eval.batch_size, drop_last=True, shuffle=False)
    high, img = next(iter(test_dl))
    high_sample = make_grid(high, nrow=2, padding=2)
    save_image(high_sample, f'{date_root}/label_image.png')
    img_grid = make_grid(img, nrow=2, padding=2)
    save_image(img_grid, f'{date_root}/artifact.png')

    angles = np.linspace(0, np.pi, 180, endpoint=False)
    radon = CT(img_width=size, radon_view=num_proj, circle=False, device=config.device)
    radon_all = CT(img_width=size, radon_view=180, circle=False, device=config.device)
    mask = torch.zeros([config.eval.batch_size, 1, det_count, 180]).to(config.device)
    mask[..., ::sparsity] = 1
    
    img = img.to(config.device)
    # Dimension Reducing (DR)
    sinogram = radon.A(img)

    # Dimension Preserving (DP)
    sinogram_full = radon_all.A(img) * mask
    
    # FBP
    fbp = radon_all.A_dagger(sinogram)
    fbp_grid = make_grid(fbp, nrow=2, padding=2)
    save_image(fbp_grid, f'{date_root}/fbp.png')
    
    predictor = ReverseDiffusionPredictor
    corrector = LangevinCorrector
    if solver == 'MCG':
        pc_MCG = controllable_generation.get_pc_radon_MCG(sde,
                                                        predictor, corrector,
                                                        inverse_scaler,
                                                        snr=snr,
                                                        n_steps=n_steps,
                                                        probability_flow=probability_flow,
                                                        continuous=config.training.continuous,
                                                        denoise=True,
                                                        radon=radon,
                                                        radon_all=radon_all,
                                                        weight=0.1,
                                                        save_progress=False,
                                                        save_root=date_root,
                                                        lamb_schedule=lamb_schedule,
                                                        mask=mask)
        x = pc_MCG(score_model, scaler(img), measurement=sinogram)
    elif solver == 'song':
        pc_song = controllable_generation.get_pc_radon_song(sde,
                                                            predictor, corrector,
                                                            inverse_scaler,
                                                            snr=snr,
                                                            n_steps=n_steps,
                                                            probability_flow=probability_flow,
                                                            continuous=config.training.continuous,
                                                            save_progress=True,
                                                            save_root=date_root,
                                                            denoise=True,
                                                            radon=radon_all,
                                                            lamb=0.7)
        x = pc_song(score_model, scaler(img), mask, measurement=sinogram_full)
    else:
        pass
        
    x_grid = make_grid(x, nrow=2, padding=2)
    save_image(x_grid, f'{date_root}/last_image.png')
    
if __name__ == '__main__':
    config = get_config()

    test(config)