#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss_mask(network_output, gt, mask = None):
    if mask is None:
        return l1_loss(network_output, gt)
    else:
        return torch.abs((network_output - gt) * mask).sum() / mask.sum()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, mask=None, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if mask is not None:
        img1 = img1 * mask + (1 - mask)
        img2 = img2 * mask + (1 - mask)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def loss_photometric(image, gt_image, opt, valid=None,ws=11):
    Ll1 =  l1_loss_mask(image, gt_image, mask=valid)
    loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image, mask=valid, window_size=ws)))
    return loss

#########################3

def _weighted_ssim_scale(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map

def weighted_ssim_scale(img1, img2, weight_map, window_size=11):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    ssim_map = _weighted_ssim_scale(img1, img2, window, window_size, channel)
    #return ((1.0 - ssim_map) * weight_map).mean()
    return (ssim_map * weight_map).mean()

def weighted_l1_loss_scale(pred, target, weight_map):
    """
    pred, target: [3, H, W], CUDA
    weight_map: [1, H, W], CUDA
    """
    abs_diff = torch.abs(pred - target)
    weighted = abs_diff * weight_map
    return weighted.mean()

def weighted_l1_loss_pixelwise(pred, target, weight_map):
    """
    pred, target: [C, H, W], CUDA
    weight_map: [1, H, W], CUDA
    """
    abs_diff = torch.abs(pred - target)  # [C, H, W]

    # Flatten spatial dims
    abs_diff_flat = abs_diff.view(abs_diff.size(0), -1)      # [C, H*W]
    weight_flat = weight_map.view(weight_map.size(0), -1)    # [1, H*W]

    # Normalize weights over spatial dimension for each channel
    norm_weights = weight_flat / (weight_flat.sum(dim=1, keepdim=True) + 1e-8)  # [1, H*W]

    # Weighted sum of absolute differences
    weighted_loss = (abs_diff_flat * norm_weights).sum(dim=1)  # [C]

    # Average over channels
    return weighted_loss.mean()


def _weighted_ssim_pixelwise(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map  # [1, C, H, W]

def weighted_ssim_pixelwise(img1, img2, weight_map, window_size=11):
    """
    img1, img2: [C, H, W] (no batch dim)
    weight_map: [1, H, W]
    Returns scalar weighted SSIM loss
    """
    C, H, W = img1.shape

    # Add batch dimension
    img1 = img1.unsqueeze(0)  # [1, C, H, W]
    img2 = img2.unsqueeze(0)
    weight_map = weight_map.unsqueeze(0)  # [1, 1, H, W]

    # Create Gaussian window
    window = create_window(window_size, C)
    window = window.to(img1.device).type_as(img1)

    # Compute SSIM map
    ssim_map = _weighted_ssim_pixelwise(img1, img2, window, window_size, C)  # [1, C, H, W]
    ssim_map = torch.clamp(ssim_map, 0.0, 1.0) ## clamp SSIM
    #loss_map = 1.0 - ssim_map  # [1, C, H, W]
    loss_map = ssim_map
    # Flatten spatial dims
    loss_flat = loss_map.view(C, -1)          # [C, H*W]
    weight_flat = weight_map.view(1, -1)      # [1, H*W]

    # Normalize weights to sum to 1 over spatial dims
    norm_weights = weight_flat / (weight_flat.sum(dim=1, keepdim=True) + 1e-8)  # [1, H*W]

    # Compute weighted sum per channel
    weighted_loss = (loss_flat * norm_weights).sum(dim=1)  # [C]

    return weighted_loss.mean()


def loss_photometric_texture_scale(image, gt_image, opt, valid=None,ws=11,texture_map=None):
    Ll1 =  weighted_l1_loss_scale(image, gt_image, texture_map)
    loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - weighted_ssim_scale(image, gt_image,texture_map, window_size=ws)))
    return loss

def loss_photometric_texture_pixelwise(image, gt_image, opt, valid=None,ws=11,texture_map=None):
    Ll1 =  weighted_l1_loss_pixelwise(image, gt_image, texture_map)
    loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - weighted_ssim_pixelwise(image, gt_image,texture_map, window_size=ws)))
    return loss