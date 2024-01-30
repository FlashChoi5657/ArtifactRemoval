

def get_ct_subsampling_mask(size, n_angles, expansion):
    diameter = math.ceil(np.sqrt(2.) * size)
    expanded_diameter = expand_diameter(diameter, expansion)
    sampled_row_ids = np.round(np.linspace(0, size - 1, n_angles)).astype(np.int32)
    return jnp.zeros((size, expanded_diameter)).at[sampled_row_ids, :].set(1.)

def get_masks(config):
    mask = get_ct_subsampling_mask(config.data.image_size, n_angles=config.sampling.n_projections,
                                    expansion=config.sampling.expansion)[None, ..., None]
    return mask

    # elif config.sampling.task in ('sparse_mar', 'mar'):
    #     if config.sampling.task == 'mar':
    #         n_projections = config.data.image_size
    #     else:
    #         n_projections = config.sampling.n_projections

    #         mask1 = (~get_metal_trace(img[..., 0], projection=config.data.image_size,
    #                                 expansion=config.sampling.expansion)[..., None]).astype(jnp.float32)
    #         mask2 = get_ct_subsampling_mask(config.data.image_size, n_angles=n_projections,
    #                                         expansion=config.sampling.expansion)[None, ..., None]
    #         return mask1 * mask2


def get_ct_mask(size, n_angles, expansion):
    diameter = math.ceil(np.sqrt(2.) * size)
    expanded_diameter = expand_diameter(diameter, expansion)
    x, y = get_kspace_radial(diameter, expanded_diameter, n_angles)
    return jnp.zeros((expanded_diameter, expanded_diameter)).at[y, x].set(1.)


def merge_known_with_mask(config, x_space, known, mask, coeff=1.):
    # if config.sampling.task == 'mri':
    #     return known * mask * coeff + x_space * (1. - mask * coeff)
    # if config.sampling.task in ('ct', 'mar', 'sparse_mar'):
    size = config.data.image_size
    expansion = config.sampling.expansion
    x_sino = fft_kspace_to_sino(x_space[..., 0], size, size, expansion)[..., None]
    known_sino = fft_kspace_to_sino(known[..., 0], size, size, expansion)[..., None]
    merged_sino = x_sino * (1. - mask * coeff) + known_sino * mask * coeff
    merged_kspace = fft_sino_to_kspace(merged_sino[..., 0], size, size, expansion)[..., None]
    ct_mask = get_ct_mask(size, size, expansion)[None, ..., None]
    merged_kspace = merged_kspace * ct_mask + x_space * (1. - ct_mask)
    return merged_kspace

def langevin_projection_sampler(config, sde, corrector, model, shape, radon, lamb
                                inverse_scaler, n_steps=1, continuous=True, eps=1e-5):
    # to_space = lambda x: fft_radon_to_kspace(x[..., 0], config.sampling.expansion)[..., None]
    # from_space = lambda x: fft_radon_to_image(x[..., 0], config.data.image_size)[..., None]
    def _A(x):
        return radon.A(x)

    def _A_dagger(sinogram):
        return radon.A_filter(sinogram)
    
    def get_inpaint_update_fn(update_fn):
        def inpaint_update_fn(x, t, mask, known, coeff):

            y_mean, std = sde.marginal_prob(known, t)
            hat_y = y_mean + torch.randn_like(y_mean) * std[:, None, None, None]
            w_hat_y = hat_y * coeff
            
            x_sino = _A(x)
            merged_sino = x_sino * (1. - mask * coeff) + w_hat_y * mask * coeff + x_sino * (1. - mask)
            x = _A_dagger(merged_sino)
            
            mean_sino = _A(x_mean)
            updated_y_mean = mean_sino * mask * (1. - coeff) + x_sino * (1. - mask) + y_mean * coeff
            
            x_space = merge_known_with_mask(config, x_space, noisy_known, mask, coeff)
            x = from_space(x_space)

            rng, step_rng = jax.random.split(rng)
            x, x_mean = update_fn(step_rng, state, x, t)

            return x

        return inpaint_update_fn
    
    def langevin_projection_sampler(rng, state, img, coeff, snr):
        # Initial sample
        rng, step_rng = random.split(rng)
        x = sde.prior_sampling(step_rng, shape)

        mask = get_masks(config, img)
        known = get_known(config, img)

        corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                                sde=sde,
                                                model=model,
                                                corrector=corrector,
                                                continuous=continuous,
                                                snr=snr,
                                                n_steps=n_steps)

        cs_corrector_update_fn = get_inpaint_update_fn(corrector_update_fn)

        timesteps = jnp.linspace(sde.T, eps, sde.N)

        def loop_body(carry, i):
            rng, x = carry
            t = timesteps[i]
            vec_t = jnp.ones(shape[0]) * t
            rng, step_rng = random.split(rng)
            x = cs_corrector_update_fn(step_rng, state, x, vec_t, mask, known, coeff)
            output = x
            return (rng, x), output

        _, all_samples = jax.lax.scan(loop_body, (rng, x), jnp.arange(0, sde.N), length=sde.N)

        output = all_samples[-1]
        if denoise:
            t_eps = jnp.full((output.shape[0],), eps)
            k, std = sde.marginal_prob(jnp.ones_like(output), t_eps)
            score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state,
                                            train=False, continuous=continuous, return_state=False)
            score = score_fn(output, t_eps)
            output = output / k + batch_mul(std ** 2, score / k)
            output_space = to_space(output)
            output_space = merge_known_with_mask(config, output_space, known, mask, coeff)
            output = from_space(output_space)

        return inverse_scaler(output)

    return jax.pmap(langevin_projection_sampler, axis_name='batch', in_axes=(0, 0, 0, None, None))






