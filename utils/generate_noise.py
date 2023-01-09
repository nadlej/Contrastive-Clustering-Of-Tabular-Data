import numpy as np
import torch

def generate_noisy_xbar(x, masking_type, masking_ratio):
    no, dim = x.shape
    x = np.array(x)
    x_bar_noisy = np.zeros([no, dim])

    if masking_type == 'swap_noise':
        process_swap_noise(x, x_bar_noisy)
    elif masking_type == 'gaussian':
        x_bar_noisy = process_gaussian_noise(x, x_bar_noisy)
    elif masking_type == 'mixed':
        x_bar_noisy = process_mixed_noise(x, x_bar_noisy, masking_ratio)

    x_bar = get_masked_data(x, x_bar_noisy, masking_ratio)
    x_bar = torch.Tensor(x_bar)
    return x_bar

def process_swap_noise(x, x_bar_noisy):
    no, dim = x.shape
    for i in range(dim):
        idx = np.random.permutation(no)
        x_bar_noisy[:, i] = x[idx, i]

def process_gaussian_noise(x, x_bar_noisy):
    x_bar_noisy = x + np.random.normal(0, 0.1, x.shape)
    return x_bar_noisy

def process_mixed_noise(x, x_bar_noisy, masking_ratio):
    x_bar_part = np.copy(x_bar_noisy)
    x_bar_part = get_masked_data(x, x_bar_part, masking_ratio)
    process_swap_noise(x_bar_part, x_bar_part)
    x_bar_noisy = process_gaussian_noise(x_bar_part, x_bar_noisy)
    return x_bar_noisy

def get_masked_data(x_ori, x_per, masking_ratio):
    mask = np.random.binomial(1, masking_ratio, x_ori.shape)
    x_bar = x_ori * (1 - mask) + x_per * mask
    return x_bar