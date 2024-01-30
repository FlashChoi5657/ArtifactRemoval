from models import utils as mutils
import torch
import numpy as np
from sampling import NoneCorrector, NonePredictor, shared_corrector_update_fn, shared_predictor_update_fn
import functools
from physics.ct import CT
from utils import show_samples, show_samples_gray, clear, clear_color
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from PIL import Image

def get_pc_radon_MCG(sde, predictor, corrector, inverse_scaler, snr,
                     n_steps=1, probability_flow=False, continuous=False, weight=1.0,
                     denoise=True, eps=1e-5, radon=None, radon_all=None, save_progress=False, save_root=None,
                     lamb_schedule=None, mask=None, measurement_noise=False):
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def _A(x): 
        return radon.A(x)

    def _AT(sinogram):  # bluring
        return radon.AT(sinogram)

    def _AINV(sinogram): # sparse view
        return radon.A_filter(sinogram)

    def _A_all(x):
        return radon_all.A(x)

    def _AINV_all(sinogram):
        return radon_all.A_filter(sinogram)

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, _, _ = update_fn(x, vec_t, model=model)
                return x

        return radon_update_fn

    def get_corrector_update_fn(update_fn):
        def radon_update_fn(model, data, x, t, measurement=None, i=None, norm_const=None):
            vec_t = torch.ones(data.shape[0], device=data.device) * t

            # mn True
            if measurement_noise:
                measurement_mean, std = sde.marginal_prob(measurement, vec_t)
                measurement = measurement_mean + torch.randn_like(measurement) * std[:, None, None, None]
                
            # input to the score function
            x = x.requires_grad_()
            x_next, x_next_mean, score = update_fn(x, vec_t, model=model)

            lamb = lamb_schedule.get_current_lambda(i) # 1.0 -> 0.5 까지 0.005씩 감소

            # x0 hat estimation
            _, bt = sde.marginal_prob(x, vec_t)
            bt = torch.unsqueeze(bt, 1)
            bt = torch.unsqueeze(bt, 2)
            bt = torch.unsqueeze(bt, 3)
            hatx0 = x + (bt ** 2) * score

            # MCG method
            # norm = torch.linalg.norm(_AINV(measurement - _A(hatx0)))
            norm = torch.norm(_AINV(measurement - _A(hatx0))) # 모두 더해 제곱근, 스칼라 값, 거리.
            # print(norm, measurement.shape, hatx0.shape) 계속 감소, [4,1,182,30], [4,1,128,128]
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
            # print(norm_grad.shape) [4,1,128,128]
            norm_grad *= weight
            norm_grad = _AINV_all(_A_all(norm_grad) * (1. - mask))

            x_next = x_next + lamb * _AT(measurement - _A(x_next)) / norm_const - norm_grad
            x_next = x_next.detach()
            return x_next

        return radon_update_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_radon_update_fn = get_corrector_update_fn(corrector_update_fn)

    def pc_radon(model, data, label, measurement=None):
        PSNR = PeakSignalNoiseRatio().to(device=data.device)
        SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device=data.device)
        psnr = np.zeros(2000)
        x = sde.prior_sampling(data.shape).to(data.device)
        ones = torch.ones_like(x).to(data.device)
        norm_const = _AT(_A(ones)) # ones -> fbp -> image , degradation?
        timesteps = torch.linspace(sde.T, eps, sde.N)
        for i in tqdm(range(sde.N)):
            t = timesteps[i]
            x = predictor_denoise_update_fn(model, data, x, t)
            x = corrector_radon_update_fn(model, data, x, t, measurement=measurement, i=i, norm_const=norm_const)
            # psnr[i] = psnr_value
            if save_progress:
                if (i % 200) == 0:
                    x_grid = make_grid(x, nrow=1, padding=2, normalize=True)
                    # img = Image.fromarray(x_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    # imgs.append(img)
                    save_image(x_grid, f'{save_root}/progress{i}.png')
        ssim_value = (SSIM(x, label)).detach().cpu().numpy()
        psnr_value = (PSNR(x, label)).detach().cpu().numpy()
        print(psnr_value, ssim_value)
        # plt.plot(psnr[:i], color='red', label='PSNR')
        # plt.legend()
        # plt.savefig(f'{save_root}/PSNR_MCG.png')
        # imgs[0].save(f'{save_root}/MCG/denoising_movie.gif', save_all=True, append_images=imgs[1:], duration=1, loop=0)
        return inverse_scaler(x if denoise else x)

    return pc_radon


def get_pc_radon_song(sde, predictor, corrector, inverse_scaler, snr,
                      n_steps=1, probability_flow=False, continuous=False,
                      denoise=True, eps=1e-5, radon=None, save_progress=False, save_root=None, lamb=1.0,
                      freq=10):
    """ Sparse application of measurement consistency """
    # Define predictor & corrector
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def _A(x):
        return radon.A(x)

    def _A_dagger(sinogram):
        return radon.A_filter(sinogram)

    def data_fidelity(mask, x, x_mean, vec_t=None, measurement=None, lamb=lamb, i=None):
        y_mean, std = sde.marginal_prob(measurement, vec_t)
        hat_y = (y_mean + torch.rand_like(y_mean) * std[:, None, None, None]) * mask
        weighted_hat_y = hat_y * lamb

        sino = _A(x)
        sino_meas = sino * mask
        weighted_sino_meas = sino_meas * (1 - lamb)
        sino_unmeas = sino * (1. - mask)

        weighted_sino = weighted_sino_meas + sino_unmeas

        updated_y = weighted_sino + weighted_hat_y
        x = _A_dagger(updated_y)

        sino_mean = _A(x_mean)
        updated_y_mean = sino_mean * mask * (1. - lamb) + sino * (1. - mask) + y_mean * lamb
        x_mean = _A_dagger(updated_y_mean)
        return x, x_mean

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean, _ = update_fn(x, vec_t, model=model)
                return x, x_mean

        return radon_update_fn

    def get_corrector_update_fn(update_fn):
        def radon_update_fn(model, data, mask, x, t, measurement=None, i=None):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean, _ = update_fn(x, vec_t, model=model)
                x, x_mean = data_fidelity(mask, x, x_mean, vec_t=vec_t, measurement=measurement, lamb=lamb, i=i)
                return x, x_mean

        return radon_update_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_denoise_update_fn = get_update_fn(corrector_update_fn)
    corrector_radon_update_fn = get_corrector_update_fn(corrector_update_fn)

    def pc_radon(model, data, label, mask, measurement=None):
        PSNR = PeakSignalNoiseRatio().to(device=data.device)
        SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device=data.device)
        with torch.no_grad():
            x = sde.prior_sampling(data.shape).to(data.device)
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                x, x_mean = predictor_denoise_update_fn(model, data, x, t)
                if (i % freq) == 0:
                    x, x_mean = corrector_radon_update_fn(model, data, mask, x, t, measurement=measurement, i=i)
                else:
                    x, x_mean = corrector_denoise_update_fn(model, data, x, t)
                if (i % 200) == 0:
                    x_grid = make_grid(x, nrow=1, padding=2, normalize=True)
                    save_image(x_grid, f'{save_root}/progress{i}.png')
            ssim_value = (SSIM(x, label)).detach().cpu().numpy()
            psnr_value = (PSNR(x, label)).detach().cpu().numpy()
            print(psnr_value, ssim_value)
            return inverse_scaler(x_mean if denoise else x)

    return pc_radon


def get_pc_radon_Artifact(sde, predictor, corrector, inverse_scaler, snr,
                     n_steps=1, probability_flow=False, continuous=False, weight=0.1, name=None,
                     denoise=True, eps=1e-5, radon=None, radon_all=None, save_progress=False, save_root=None,
                     lamb_schedule=None, mask=None, measurement_noise=True):
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def _A(x): 
        return radon.A(x)

    def _AT(sinogram):  # bluring
        return radon.AT(sinogram)

    def _AINV(sinogram): # sparse view
        return radon.A_filter(sinogram)

    def R_T(x):
        return radon_all.A(x)

    def FBP(sinogram):
        return radon_all.A_filter(sinogram)

    def data_fidelity(mask, x, x_mean, vec_t=None, measurement=None, lamb=None):
        # y hat timestep s        
        y_mean, std = sde.marginal_prob(measurement, vec_t)
        hat_y = (y_mean + torch.rand_like(y_mean) * std[:, None, None, None]) * mask
        weighted_hat_y = hat_y * lamb
        
        # degradation sinogram, mask는 degradation 
        sino = R_T(x)
        weighted_sino_masked = sino * (1 - lamb) * mask
        sino_unmasked = sino * (1. - mask)

        weighted_sino = weighted_sino_masked + sino_unmasked

        updated_y = weighted_sino + weighted_hat_y
        x = FBP(updated_y)

        sino_mean = R_T(x_mean)
        updated_y_mean = sino_mean * mask * (1. - lamb) + sino * (1. - mask) + y_mean * lamb
        x_mean = FBP(updated_y_mean)
        return x, x_mean

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, _, _ = update_fn(x, vec_t, model=model)
                return x

        return radon_update_fn

    def get_corrector_update_fn(update_fn):
        def radon_update_fn(model, data, x, t, measurement=None, i=None, norm_const=None):
            vec_t = torch.ones(data.shape[0], device=data.device) * t
            if measurement_noise:
                measurement_mean, std = sde.marginal_prob(measurement, vec_t)
                measurement = measurement_mean + torch.randn_like(measurement) * std[:, None, None, None]
            # input to the score function
            x = x.requires_grad_()
            x_next, x_next_mean, score = update_fn(x, vec_t, model=model)

            lamb = lamb_schedule.get_current_lambda(i) # 1.0 -> 0.5 까지 0.005씩 감소

            # x0 hat estimation
            _, bt = sde.marginal_prob(x, vec_t)
            bt = torch.unsqueeze(bt, 1)
            bt = torch.unsqueeze(bt, 2)
            bt = torch.unsqueeze(bt, 3)
            hatx0 = x + (bt ** 2) * score

            # MCG method
            # norm = torch.linalg.norm(_AINV(measurement - _A(hatx0)))
            norm = torch.norm(data-hatx0) # 모두 더해 제곱근, 스칼라 값, 거리.
            # print(norm, measurement.shape, hatx0.shape) 계속 감소, [4,1,182,30], [4,1,128,128]
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
            # print(norm_grad.shape) [4,1,128,128]
            norm_grad *= weight
            norm_grad = FBP(R_T(norm_grad) * (1. - mask))


            # merged = measurement * mask * 0.5 + 0.5 * mask * R_T(x_next) + (1. - mask) * R_T(x_next)
            pocs = _AT(measurement - _A(x_next)) / norm_const
            x_next = x_next + pocs - norm_grad
            # x_next = FBP(merged) / norm_const - norm_grad
            x_next = x_next.detach()
            return x_next

        return radon_update_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_radon_update_fn = get_corrector_update_fn(corrector_update_fn)

    def pc_radon(model, data, label, measurement=None, jump=None):
        PSNR = PeakSignalNoiseRatio().to(device=data.device)
        SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device=data.device)
        psnr = np.zeros(2000)
        ssim = np.zeros(2000)
        x = sde.prior_sampling(data.shape).to(data.device)
        ones = torch.ones_like(x).to(data.device)
        norm_const = _AT(_A(ones)) # ones -> fbp -> image , degradation?
        # norm_const = FBP(R_T(ones))
        timesteps = torch.linspace(sde.T, eps, sde.N)
        jump = jump
        for i in tqdm(range(sde.N//jump)):
            # if jump*(i+2) > 2000:
            #     t = timesteps[-1]
            # else:
            t = timesteps[jump*i]
            x = predictor_denoise_update_fn(model, data, x, t)
            x = corrector_radon_update_fn(model, data, x, t, measurement=measurement, i=i, norm_const=norm_const)

            # psnr[i] = psnr_value
            # ssim[i] = ssim_value
            if (i % 200) == 0:
                x_grid = make_grid(x, nrow=2, padding=2, normalize=True)
                # img = Image.fromarray(x_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                # imgs.append(img)
                save_image(x_grid, f'{save_root}/{jump}_progress{i}.png')
        # plt.subplot(1,2,1)
        # plt.plot(psnr[:i], color='red', label='PSNR')
        # plt.legend()
        # plt.subplot(1,2,2)
        # plt.plot(ssim[:i], color='green', label='SSIM')
        # plt.legend()
        psnr_value = (PSNR(x, label)).detach().cpu().numpy()
        ssim_value = (SSIM(x, label)).detach().cpu().numpy()        
        # plt.savefig(f'{save_root}/{jump}_performace.png')
        # imgs[0].save(f'{save_root}/MCG/denoising_movie.gif', save_all=True, append_images=imgs[1:], duration=1, loop=0)
        print(jump, ':', psnr_value, ssim_value)
        return inverse_scaler(x if denoise else x), psnr, ssim

    return pc_radon



def get_pc_radon_POCS(sde, predictor, corrector, inverse_scaler, snr,
                      n_steps=1, probability_flow=False, continuous=False,
                      denoise=True, eps=1e-5, radon=None, save_progress=False, save_root=None,
                      lamb_schedule=None, measurement_noise=False, final_consistency=False):
    """ Sparse application of measurement consistency """
    # Define predictor & corrector
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def _A(x):
        return radon.A(x)

    def _AT(sinogram):
        return radon.A_filter(sinogram)

    def kaczmarz(x, x_mean, measurement=None, lamb=1.0, i=None,
                 norm_const=None):
        x = x + lamb * _AT(measurement - _A(x)) / norm_const
        x_mean = x
        return x, x_mean

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean, _ = update_fn(x, vec_t, model=model)
                return x, x_mean

        return radon_update_fn

    def get_corrector_update_fn(update_fn):
        def radon_update_fn(model, data, x, t, measurement=None, i=None, norm_const=None):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean, _ = update_fn(x, vec_t, model=model)
                lamb = lamb_schedule.get_current_lambda(i)

                if measurement_noise:
                    measurement_mean, std = sde.marginal_prob(measurement, vec_t)
                    measurement = measurement_mean + torch.randn_like(measurement) * std[:, None, None, None]

                x, x_mean = kaczmarz(x, x_mean, measurement=measurement, lamb=lamb, i=i,
                                     norm_const=norm_const)
                return x, x_mean

        return radon_update_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_radon_update_fn = get_corrector_update_fn(corrector_update_fn)

    def pc_radon(model, data, measurement=None):
        with torch.no_grad():
            x = sde.prior_sampling(data.shape).to(data.device)

            ones = torch.ones_like(x).to(data.device)
            norm_const = _AT(_A(ones))
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                x, x_mean = predictor_denoise_update_fn(model, data, x, t)
                x, x_mean = corrector_radon_update_fn(model, data, x, t, measurement=measurement, i=i,
                                                      norm_const=norm_const)
                if save_progress:
                    if (i % 20) == 0:
                        print(f'iter: {i}/{sde.N}')
                        plt.imsave(save_root / 'recon' / f'progress{i}.png', clear(x_mean), cmap='gray')
            # Final step which coerces the data fidelity error term to be zero,
            # and thereby satisfying Ax = y
            if final_consistency:
                x, x_mean = kaczmarz(x, x_mean, measurement, lamb=1.0, norm_const=norm_const)

            return inverse_scaler(x_mean if denoise else x)

    return pc_radon






def get_pc_radon_UNet_plus(sde, predictor, corrector, inverse_scaler, snr,
                     n_steps=1, probability_flow=False, continuous=False, weight=0.1, name=None,
                     denoise=True, eps=1e-5, radon=None, radon_all=None, save_progress=False, save_root=None,
                     lamb_schedule=None, mask=None, measurement_noise=True):
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def _A(x): 
        return radon.A(x)

    def _AT(sinogram):  # bluring
        return radon.AT(sinogram)


    def R_T(x):
        return radon_all.A(x)

    def FBP(sinogram):
        return radon_all.A_filter(sinogram)

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, _, _ = update_fn(x, vec_t, model=model)
                return x

        return radon_update_fn

    def get_corrector_update_fn(update_fn):
        def radon_update_fn(model, data, x, t, measurement=None, i=None, norm_const=None):
            vec_t = torch.ones(data.shape[0], device=data.device) * t
            if measurement_noise:
                measurement_mean, std = sde.marginal_prob(measurement, vec_t)
                measurement = measurement_mean + torch.randn_like(measurement) * std[:, None, None, None]
            # input to the score function
            x = x.requires_grad_()
            x_next, x_next_mean, score = update_fn(x, vec_t, model=model)

            lamb = lamb_schedule.get_current_lambda(i) # 1.0 -> 0.5 까지 0.005씩 감소

            # x0 hat estimation
            _, bt = sde.marginal_prob(x, vec_t)
            bt = torch.unsqueeze(bt, 1)
            bt = torch.unsqueeze(bt, 2)
            bt = torch.unsqueeze(bt, 3)
            hatx0 = x + (bt ** 2) * score

            # MCG method
            norm = torch.norm(data-hatx0) # 모두 더해 제곱근, 스칼라 값, 거리.
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
            norm_grad *= weight
            norm_grad_w = FBP(R_T(norm_grad) * (1. - mask))

            pocs = _AT(measurement - _A(x_next)) / norm_const
            x_next = x_next + pocs - norm_grad_w
            x_next = x_next.detach()
            return x_next

        return radon_update_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_radon_update_fn = get_corrector_update_fn(corrector_update_fn)

    def pc_radon(model, data, label, measurement=None, jump=None):
        PSNR = PeakSignalNoiseRatio().to(device=data.device)
        SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device=data.device)
        psnr = np.zeros(2000)
        ssim = np.zeros(2000)
        x = sde.prior_sampling(data.shape).to(data.device)
        ones = torch.ones_like(x).to(data.device)
        norm_const = _AT(_A(ones)) # ones -> fbp -> image , degradation?
        timesteps = torch.linspace(sde.T, eps, sde.N)
        jump = jump
        for i in tqdm(range(sde.N//jump)):
            # if jump*(i+2) > 2000:
                # t = timesteps[-1]
            # else:
            t = timesteps[jump*i]
            x = predictor_denoise_update_fn(model, data, x, t)
            x = corrector_radon_update_fn(model, data, x, t, measurement=measurement, i=i, norm_const=norm_const)

            # if (i%500) ==  0:
            #     x_grid = make_grid(x, nrow=2, padding=2, normalize=True)
            #     save_image(x_grid, f'{save_root}/{jump}_progress{i}.png')
            #     x_grid1 = make_grid(mcg, nrow=2, padding=2, normalize=True)
            #     save_image(x_grid1, f'{save_root}/{jump}_mcg{i}.png')
            #     x_grid2 = make_grid(projection, nrow=2, padding=2, normalize=True)
            #     save_image(x_grid2, f'{save_root}/{jump}_projection{i}.png')
            #     x_grid3 = make_grid(merge_mcg, nrow=2, padding=2, normalize=True)
            #     save_image(x_grid3, f'{save_root}/{jump}_mcg_merge{i}.png')
        psnr_value = (PSNR(x, label)).detach().cpu().numpy()
        ssim_value = (SSIM(x, label)).detach().cpu().numpy()        
        print(jump, ':', psnr_value, ssim_value)

        return inverse_scaler(x if denoise else x)

    return pc_radon