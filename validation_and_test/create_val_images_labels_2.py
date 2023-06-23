# ------------------------------------------------------------------------------
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# see CC-BY-NC-SA-4.0.md for details
# Written by Taihong Xiao
# ------------------------------------------------------------------------------

from data_new import dataset, download

import os
import cv2
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))



def net_forward(net, x, uselayer=5):
    x = net.conv1(x)
    x = net.bn1(x)
    x = net.relu(x)
    x = net.maxpool(x)

    x = net.layer1(x)
    x = net.layer2(x)
    x = net.layer3(x)
    if uselayer == 3:
        return x
    x = net.layer4(x)
    if uselayer == 4:
        return x

    x = net.avgpool(x)
    if uselayer == 'avgpool':
        return x
    x = torch.flatten(x, 1)
    x = net.fc(x)
    if uselayer == 5:
        return x

def batch_random_rotate(images, meshgrids, max_degree):
    """Randomly rotates a batch of images.
    Args:
        images: [B, C, H, W]
        meshgrids: [B, 2, H, W]
        max_degree: the max degree of random rotate, a float number

    Returns:
        rotate_images: [B, C, H, W]
        rotate_grids: [B, 2, H, W]
    """
    import math

    B, C, H, W = images.size()
    max_angle = max_degree * math.pi / 180
    angle = (torch.rand(B, 1).to(images.device) * 2 - 1) * max_angle # range [-max_angle, max_angle]
    zero = torch.zeros_like(angle)

    theta = torch.cat(
        [torch.cos(angle), -torch.sin(angle), zero,
         torch.sin(angle),  torch.cos(angle), zero], 1
    ).view(-1, 2, 3)

    grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)
    rotate_images = F.grid_sample(images, grid, align_corners=False)
    rotate_grids = F.grid_sample(meshgrids, grid, align_corners=False)

    # return rotate_images.contiguous(), rotate_grids.contiguous()
    return rotate_images, rotate_grids

def batch_random_crop(images, meshgrids, kps):
    """Randomly crops a batch of images.
    Args:
        images: [B, C, H, W]
        meshgrids: [B, 2, H, W]
        crop_size: [H0, W0]
        kps: [B, 2, N], width and height

    Returns:
        crop_images: [B, C, H0, W0]
        crop_grids: [B, 2, H0, W0]
    """
    B, C, H, W = images.size()
    # H0, W0 = crop_size
    kps = kps.to(torch.int64)
    top_left = kps.min(-1)[0]
    bottom_right = kps.max(-1)[0]

    x1 = torch.randint(top_left[0,0], (B, 1)).to(torch.float32).to(images.device)
    y1 = torch.randint(top_left[0,1], (B, 1)).to(torch.float32).to(images.device)

    x2 = torch.randint(bottom_right[0,0], W, (B, 1)).to(torch.float32).to(images.device)
    y2 = torch.randint(bottom_right[0,1], H, (B, 1)).to(torch.float32).to(images.device)

    # x1 = torch.randint(W - W0, (B, 1)).to(torch.float32).to(images.device)
    # y1 = torch.randint(H - H0, (B, 1)).to(torch.float32).to(images.device)
    # x2 = x1 + W0
    # y2 = y1 + H0
    zero = torch.zeros_like(x1)
    H0 = int((y2 - y1).item())
    W0 = int((x2 - x1).item())

    theta = torch.cat(
        [(x2 - x1) / (W - 1), zero, (x1 + x2 - W + 1) / (W - 1),
         zero, (y2 - y1) / (H - 1), (y1 + y2 - H + 1) / (H - 1)], 1
    ).view(-1, 2, 3)

    grid = F.affine_grid(theta, (B, C, H0, W0), align_corners=False)
    crop_images = F.grid_sample(images, grid, align_corners=False)
    crop_grids = F.grid_sample(meshgrids, grid, align_corners=False)

    # cv2.imwrite('0.png', cv2.cvtColor(np.array(crop_images[0].detach().numpy().transpose(1,2,0) * 128 + 128, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    # return crop_images.contiguous(), crop_grids.contiguous()
    return crop_images, crop_grids

def batch_flip(first_frame, meshgrids):
    """Randomly flip the image and meshgrids and attention map.

    Args:
        first_frame: [B, C, H, W], torch.Tensor, can be on gpu
        meshgrids: [B, 2, H, W], torch.Tensor, can be on gpu

    returns:
        flip_frame: [B, C, H, W]
        flip_meshgrids: [B, 2, H, W]
    """
    B, C, H, W = first_frame.size()
    is_flip = (torch.rand(B, 1, 1, 1) > 0.5).to(first_frame.device)
    flip_frame     = torch.where(is_flip.repeat([1, C, H, W]), first_frame.flip(-1), first_frame)
    flip_meshgrids = torch.where(is_flip.repeat([1, 2, H, W]), meshgrids.flip(-1), meshgrids)
    return flip_frame, flip_meshgrids


def rotate_crop_with_gt(first_frame, kps, max_degree):
    """Rotate and crop the first frame and obtain the ground truth using attention.

    Args:
        first_frame: [B, C, H, W], torch.Tensor, can be on a gpu
        kps: [B, 2, N], torch.Tensor, can eb on a gpu
        max_degree: the max degree of random rotate, a float number
        crop_size: [H0, W0]

    returns:
        crop: [B, C, args.patch_size, args.patch_size]
        crop_gt: [B, args.patch_size // 8, args.patch_size // 8, 2]
    """
    B, C, H, W = first_frame.size()

    meshgrid = torch.stack(torch.meshgrid(
        torch.arange(H, dtype=torch.float32).to(first_frame.device) / H,
        torch.arange(W, dtype=torch.float32).to(first_frame.device) / W), 0)

    # create meshgrids, shape: [B, 2, H, W]
    meshgrids = torch.stack([meshgrid] * B, 0)
    first_frame, meshgrids = batch_flip(first_frame, meshgrids)
    crop, crop_gt = batch_random_rotate(first_frame, meshgrids, max_degree)
    crop, crop_gt = batch_random_crop(crop, crop_gt, kps)
    # crop_gt = F.interpolate(crop_gt, (crop_gt.size(2) // downscale, crop_gt.size(3) // downscale))
    scale = torch.tensor([H, W], dtype=torch.float32).view(1,-1,1,1).to(first_frame.device)

    crop_gt = (crop_gt * scale).permute(0,2,3,1)
    crop_gt = crop_gt[..., range(1,-1,-1)] # swap x and y
    crop = crop.detach()
    crop_gt = crop_gt.detach()

    # return crop.contiguous(), crop_gt.contiguous()
    return crop, crop_gt

def draw_dots(img, crd, color=(255,0,0)):
    """
    INPUTS:
    - img: BRG image,
    - crd: coordinates, np.ndarray float
    """
    image = img.copy()
    crd = np.rint(crd) # downsampled three times
    for p in crd:
        image = cv2.circle(image, (p[0], p[1]), 5, color, -1)
    return image

def save_vis(frame1, frame2, coord, savedir):
    b = frame1.size(0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[None, :, None, None]
    frame1 = (frame1.cpu() * std + mean) * 255
    frame2 = (frame2.cpu() * std + mean) * 255

    for cnt in range(b):
        im1 = frame1[cnt].cpu().detach().numpy().transpose(1,2,0)
        im_frame1 = cv2.cvtColor(np.array(im1, dtype = np.uint8), cv2.COLOR_RGB2BGR)

        im2 = frame2[cnt].cpu().detach().numpy().transpose(1,2,0)
        im_frame2 = cv2.cvtColor(np.array(im2, dtype = np.uint8), cv2.COLOR_RGB2BGR)
        im_frame2 = cv2.resize(im_frame2, (im_frame1.shape[1],im_frame1.shape[0]))

        crd = coord.data.cpu().numpy()[cnt]
        im_frame3 = draw_dots(im_frame1, crd)
        im_frame3 = cv2.resize(im_frame3, (im_frame1.shape[1],im_frame1.shape[0]))
        im = np.concatenate((im_frame1, im_frame2, im_frame3), axis=1)

        cv2.imwrite(os.path.join(savedir, "vis.png"), im)



if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='SCOT in pytorch')
    parser.add_argument('--gpu', type=str, default='', help='GPU id')
    parser.add_argument('--datapath', type=str, default='./Datasets_SCOT')
    parser.add_argument('--dataset', type=str, default='pfpascal')
    parser.add_argument('--split', type=str, default='val', help='trn,val.test')
    parser.add_argument('--thres', type=str, default='bbox', choices=['auto', 'img', 'bbox'])

    parser.add_argument('--arch', metavar='ARCH', default='resnet50')
    parser.add_argument('--modelpath', type=str, default='./ckpt/checkpoint_0062.pth.tar')

    parser.add_argument("--downscale", type=int, default=16, help='downsample scale.')
    parser.add_argument("--patch_size", type=int, default=128, help="crop size for localization.")
    parser.add_argument("--rotate",type=int,default=10,help='degree to rotate training images')

    # moco argument
    parser.add_argument('--moco_dim', default=128, type=int, help='feature dimension (default: 128)')
    parser.add_argument('--temp', default=0.0007, type=float, help='temperature for affinity matrix')

    parser.add_argument('--save_dir', type=str, default='./Datasets_SCOT/PF-PASCAL/val_images/', help='dir to save crop and gt')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = MoCo_Cycle(models.__dict__[args.arch], args.modelpath, args.moco_dim, args.temp, args.downscale, args.rotate, args.patch_size).to(device)
    # model.eval()

    dset = download.load_dataset(args.dataset, args.datapath, args.thres, device, args.split)
    dataloader = DataLoader(dset, batch_size=1, num_workers=0)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    num_kps = 10

    for idx, data in enumerate(dataloader):
        print('{}/{}'.format(1 + idx, len(dataloader)))
        crop, crop_gt = rotate_crop_with_gt(data['src_img'], data['src_kps'], 10)
        # coord = crop_gt.view(crop_gt.size(0), -1, crop_gt.size(-1))
        # save_vis(data['src_img'], crop, coord, '.')

        crop_h = crop.size(2)
        crop_w = crop.size(3)
        w_ind = torch.randint(0, crop_w, size=(1, num_kps)).to(crop.device)
        h_ind = torch.randint(0, crop_h, size=(1, num_kps)).to(crop.device)
        ind = w_ind + h_ind * crop_w
        # if idx == 3:
        #     from IPython import embed; embed(); exit()
        src_val_kps = torch.index_select(crop_gt[0].view(-1, 2), 0, ind[0]).t()
        crop_kps = torch.cat([w_ind, h_ind], 0).to(torch.float32)

        crop_name = os.path.splitext(data['src_imname'][0])[0] + '_' + os.path.splitext(data['trg_imname'][0])[0] +  '_crop.pt'
        crop_kps_name = os.path.splitext(data['src_imname'][0])[0] + '_' + os.path.splitext(data['trg_imname'][0])[0] + '_crop_kps.pt'
        src_val_kps_name = os.path.splitext(data['src_imname'][0])[0] + '_' + os.path.splitext(data['trg_imname'][0])[0] + '_kps.pt'

        torch.save(crop.squeeze(0).cpu(), os.path.join(args.save_dir, crop_name))
        torch.save(crop_kps.cpu(), os.path.join(args.save_dir, crop_kps_name))
        torch.save(src_val_kps.cpu(), os.path.join(args.save_dir, src_val_kps_name))


