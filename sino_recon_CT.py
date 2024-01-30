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
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from utils import Custom_Dataset, save_checkpoint, CT_Dataset, CT_testset, window_setting
from default_lsun_configs import get_default_configs

def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.batch_size = 64
    training.sde = 'vesde'
    training.continuous = True

    # sampling
    sampling = config.sampling
    sampling.solver = 'cmh'
    sampling.name = 'std'
    sampling.method = 'pc'
    sampling.predictor = 'reverse_diffusion'
    sampling.corrector = 'langevin'

    # data
    data = config.data
    data.dataset = 'streak_artifact'
    data.root = '/nas/dataset/users/minhyeok/LDCT'
    data.image_size = 256
    data.is_multi = False
    data.is_complex = False
    data.center_crop = 448

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
from sampling import ReverseDiffusionPredictor, LangevinCorrector, AnnealedLangevinDynamics
import controllable_generation
import matplotlib.pyplot as plt
from UNet_model import UNet
from PIL import Image

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

    # ckpt_filename = '/nas/users/minhyeok/CMH/sde/result/10_29/size128_ckpt_199_last.pth'
    ckpt_filename = '/nas/users/minhyeok/CMH/sde/result/11_22/size128_M_ckpt_200.pth'
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sde.N = N
    sampling_eps = 1e-5
        
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
    date_root = f'{workdir}/result/{time_stamp}/{solver}'
    if not os.path.exists(date_root):  os.makedirs(date_root, exist_ok=True)
    # date_root = f'{workdir}/result/{time_stamp}/MCG'
    # if not os.path.exists(date_root):  os.makedirs(date_root, exist_ok=True)
    # date_root = f'{workdir}/result/{time_stamp}/cmh'
    # if not os.path.exists(date_root):  os.makedirs(date_root, exist_ok=True)

    state = dict(step=0, model=score_model, ema=ema)

    state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True, skip_optimizer=True)
    ema.copy_to(score_model.parameters())
    
    config.eval.batch_size = 2
    transform = transforms.Compose([
            # transforms.CenterCrop(config.data.center_crop),
            transforms.Resize(config.data.image_size),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, ), (0.5, ))
            ])
    # test_set = Custom_Dataset(config.data.root, transform=transform, train=False)
    test_set = CT_testset(transform=transform, train=False)
    test_dl = DataLoader(dataset=test_set, batch_size=config.eval.batch_size)
    high, img = next(iter(test_dl))
    
    high_sample = make_grid(high, nrow=1, padding=2, normalize=True)
    save_image(high_sample, f'{date_root}/label_image.png')
    img_sample = make_grid(img, nrow=1, padding=2, normalize=True)
    save_image(img_sample, f'{date_root}/artifact_image.png')

    angles = np.linspace(0, np.pi, 180, endpoint=False)

    # UNet
    unet = UNet(in_channel=1, dim_feature=32, out_channel=1, bilinear=True).to(config.device)
    unet.load_state_dict(torch.load('/nas/users/minhyeok/CMH/sde/result/11_10/sinogram/model_ckpt_80.pth'))
    unet = unet.eval()

    # image degradation
    radon = CT(img_width=size, radon_view=num_proj, circle=False, device=config.device)
    radon_all = CT(img_width=size, radon_view=180, circle=False, device=config.device)
    img = img.to(config.device)
    high = high.to(config.device)

    mask = torch.zeros([config.eval.batch_size, 1, det_count, 180]).to(config.device)
    
    sinogram = radon.A(img)
    sinogram_full = radon_all.A(img)
    # FBP
    fbp = radon_all.A_filter(sinogram)
    fbp_grid = make_grid(fbp, nrow=int(np.sqrt(config.eval.batch_size)), padding=2, normalize=True)
    save_image(fbp_grid, f'{date_root}/fbp.png')
         
    mask[..., ::sparsity] = 1
    
    predictor = ReverseDiffusionPredictor
    corrector = LangevinCorrector
    
    if solver == 'mcg':
        with torch.no_grad():
            diff = unet(sinogram_full)
        mask = F.normalize(diff, dim=0)  
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
                                                        save_progress=True,
                                                        save_root=date_root,
                                                        lamb_schedule=lamb_schedule,
                                                        mask=mask)
        x = pc_MCG(score_model, scaler(img), scaler(high), measurement=sinogram)
        x_grid = make_grid(x, nrow=int(np.sqrt(config.eval.batch_size)), padding=2, normalize=True)
        save_image(x_grid, f'{date_root}/last_image.png')
        
    elif solver == 'arti':
        with torch.no_grad():
            diff = unet(sinogram_full)
        mask = F.normalize(diff, dim=0)
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'navy', 'violet', 'purple']
        for k in range(1,9):
            pc_MCG = controllable_generation.get_pc_radon_Artifact(sde, predictor, corrector,
                                                            inverse_scaler, snr=snr,
                                                            continuous=config.training.continuous,
                                                            radon=radon, radon_all=radon_all,
                                                            save_root=date_root,  name=config_name,
                                                            lamb_schedule=lamb_schedule, mask=mask,
                                                            )
            x, psnr, ssim = pc_MCG(score_model, scaler(img), scaler(high), measurement=sinogram, jump=k)
            x_grid = make_grid(x, nrow=int(np.sqrt(config.eval.batch_size)), padding=2, normalize=True)
            save_image(x_grid, f'{date_root}/{k}_last_image.png')
            plt.figure(1, figsize=(10,8))
            plt.title('PSNR according to k times timestep')
            plt.plot(psnr, color=colors[k-1], label=f'{k}times')
            plt.xlabel('epochs')
            plt.ylabel('PSNR')
            plt.legend()
            plt.savefig(f'{date_root}/PSNR_setting.png')
            
            plt.figure(2, figsize=(10,8))
            plt.plot(ssim, color=colors[k-1], label=f'{k}times')
            plt.title('SSIM according to k times timestep')
            plt.xlabel('epochs')
            plt.ylabel('SSIM')
            plt.legend()
            plt.savefig(f'{date_root}/SSIM_setting.png')
        
    elif solver == 'song':
        pc_song = controllable_generation.get_pc_radon_song(sde,
                                                            predictor, corrector,
                                                            inverse_scaler,
                                                            snr=0.209,
                                                            n_steps=n_steps,
                                                            probability_flow=probability_flow,
                                                            continuous=config.training.continuous,
                                                            save_progress=True,
                                                            save_root=date_root, name=config_name,
                                                            denoise=True,
                                                            radon=radon_all,
                                                            lamb=0.227)
        x = pc_song(score_model, scaler(img), mask, measurement=sinogram_full)
        x_grid = make_grid(x, nrow=int(np.sqrt(config.eval.batch_size)), padding=2, normalize=True)
        save_image(x_grid, f'{date_root}/{config_name}_last_image.png')
    elif solver == 'cmh':
        with torch.no_grad():
            diff = unet(sinogram_full)
        mask = F.normalize(diff, dim=0)
        # corrector = AnnealedLangevinDynamics
        pc_MCG = controllable_generation.get_pc_radon_UNet_plus(sde, predictor, corrector,
                                                        inverse_scaler, snr=snr,
                                                        continuous=config.training.continuous,
                                                        radon=radon, radon_all=radon_all,
                                                        save_root=date_root,  name=config_name,
                                                        lamb_schedule=lamb_schedule, mask=mask,
                                                        )
        x = pc_MCG(score_model, scaler(img), scaler(high), measurement=sinogram, jump=1)
        x_grid = make_grid(x, nrow=1, padding=2)
        save_image(x_grid, f'{date_root}/last_image.png')

        x_grid = make_grid(x, nrow=1, padding=2, normalize=True, scale_each=True)
        save_image(x_grid, f'{date_root}/last_normalized_image.png')

        x_gen = Image.fromarray(x_grid.mul_(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
        plt.imsave(f'{date_root}/last_image_high.png', x_gen)

    elif solver == 'dataset':
        transforms_train = transforms.Compose([transforms.CenterCrop(448),
                                               transforms.Resize(size),])
        train_set = CT_Dataset(config.data.root, transform=transforms_train, train=False)
        dl_train = DataLoader(dataset=train_set, batch_size=config.eval.batch_size, drop_last=True, shuffle=True)
        high, low = next(iter(dl_train))
        high = high.to(config.device)
        low = low.to(config.device)
        sinogram_full = radon_all.A(low)
        sinogram = radon.A(low)
        with torch.no_grad():
            diff = unet(sinogram_full)
        mask = F.normalize(diff, dim=0)
        name = 'lung'
        pc_MCG = controllable_generation.get_pc_radon_Artifact(sde, predictor, corrector,
                                                        inverse_scaler, snr=snr,
                                                        continuous=config.training.continuous,
                                                        radon=radon, radon_all=radon_all,
                                                        save_root=date_root,  name=name,
                                                        lamb_schedule=lamb_schedule, mask=mask)
        x, psnrl = pc_MCG(score_model, scaler(low), scaler(high), measurement=sinogram)
        x_grid = make_grid(x, nrow=int(np.sqrt(config.eval.batch_size)), padding=2, normalize=True)
        save_image(x_grid, f'{date_root}/{name}last_image.png')

        train_set = CT_Dataset(config.data.root, transform=transforms_train, train=False, ww=1800, wl=400)
        dl_train = DataLoader(dataset=train_set, batch_size=config.eval.batch_size, drop_last=True, shuffle=True)
        high, low = next(iter(dl_train))
        high = high.to(config.device)
        low = low.to(config.device)
        sinogram_full = radon_all.A(low)
        sinogram = radon.A(low)
        with torch.no_grad():
            diff = unet(sinogram_full)
        mask = F.normalize(diff, dim=0)
        name = 'bone'
        pc_MCG = controllable_generation.get_pc_radon_Artifact(sde, predictor, corrector,
                                                        inverse_scaler, snr=snr,
                                                        continuous=config.training.continuous,
                                                        radon=radon, radon_all=radon_all,
                                                        save_root=date_root,  name=name,
                                                        lamb_schedule=lamb_schedule, mask=mask)
        x, psnrb = pc_MCG(score_model, scaler(img), scaler(high), measurement=sinogram)
        x_grid = make_grid(x, nrow=int(np.sqrt(config.eval.batch_size)), padding=2, normalize=True)
        save_image(x_grid, f'{date_root}/{name}last_image.png')

        plt.plot(psnrl[:1000], color='red', label='lung')
        plt.plot(psnrb[:1000], color='green', label='bone')
        plt.legend()
        plt.savefig(f'{date_root}/PSNR_setting.png')

if __name__ == '__main__':
    config = get_config()

    test(config)