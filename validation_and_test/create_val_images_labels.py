# ------------------------------------------------------------------------------
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# see CC-BY-NC-SA-4.0.md for details
# Written by Taihong Xiao
# ------------------------------------------------------------------------------

from data import dataset, download

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

def batch_random_rotate_att(images, meshgrids, att_map, max_degree):
    """Randomly rotates a batch of images.
    Args:
        images: [B, C, H, W]
        meshgrids: [B, 2, H, W]
        att_map: [B, 1, H, W], torch.Tensor, can be on gpu
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
    rotate_att_map = F.grid_sample(att_map, grid, align_corners=False)

    # return rotate_images.contiguous(), rotate_grids.contiguous()
    return rotate_images, rotate_grids, rotate_att_map


def batch_random_crop_att(images, meshgrids, att_map, crop_size):
    """Randomly crops a batch of images.
    Args:
        images: [B, C, H, W]
        meshgrids: [B, 2, H, W]
        att_map: [B, 1, H, W], torch.Tensor, can be on gpu
        crop_size: [H0, W0]

    Returns:
        crop_images: [B, C, H0, W0]
        crop_grids: [B, 2, H0, W0]
    """
    B, C, H, W = images.size()
    H0, W0 = crop_size

    max_h, _ = torch.max(att_map[:,0], -1)
    max_val, ind_h = torch.max(max_h, -1)
    max_w, _ = torch.max(att_map[:,0], -2)
    max_val, ind_w = torch.max(max_w, -1)

    ind_h = ind_h.view(B, 1)
    ind_w = ind_w.view(B, 1)

    # in case the crop image is out of boundary
    ind_h = torch.clamp(ind_h, min=H0//2, max=H - (H0 - H0//2))
    ind_w = torch.clamp(ind_w, min=W0//2, max=W - (W0 - W0//2))

    # max_val = max_val[:, None, None, None].repeat(1, 1, H, W)
    # ind_list = torch.where(att_map == max_val)

    x1 = (ind_w - W0//2).to(torch.float32)
    y1 = (ind_h - H0//2).to(torch.float32)
    x2 = x1 + W0
    y2 = y1 + H0

    # from IPython import embed; embed(); exit()
    # x1 = torch.randint(W - W0, (B, 1)).to(torch.float32).to(images.device)
    # y1 = torch.randint(H - H0, (B, 1)).to(torch.float32).to(images.device)
    # x2 = x1 + W0
    # y2 = y1 + H0
    zero = torch.zeros_like(x1)

    theta = torch.cat(
        [(x2 - x1) / (W - 1), zero, (x1 + x2 - W + 1) / (W - 1),
         zero, (y2 - y1) / (H - 1), (y1 + y2 - H + 1) / (H - 1)], 1
    ).view(-1, 2, 3)

    grid = F.affine_grid(theta, (B, C, H0, W0), align_corners=False)
    crop_images = F.grid_sample(images, grid, align_corners=False)
    crop_grids = F.grid_sample(meshgrids, grid, align_corners=False)

    # cv2.imwrite('0.png', cv2.cvtColor(np.array(crop_images[0].detach().cpu().numpy().transpose(1,2,0) * 128 + 128, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    # return crop_images.contiguous(), crop_grids.contiguous()
    return crop_images, crop_grids


def batch_flip_att(first_frame, meshgrids, att_map):
    """Randomly flip the image and meshgrids and attention map.

    Args:
        first_frame: [B, C, H, W], torch.Tensor, can be on gpu
        meshgrids: [B, 2, H, W], torch.Tensor, can be on gpu
        att_map: [B, 1, H, W], torch.Tensor, can be on gpu

    returns:
        flip_frame: [B, C, H, W]
        flip_meshgrids: [B, 2, H, W]
        flip_att_map: [B, 1, H, W]
    """
    B, C, H, W = first_frame.size()
    is_flip = (torch.rand(B, 1, 1, 1) > 0.5).to(first_frame.device)
    flip_frame     = torch.where(is_flip.repeat([1, C, H, W]), first_frame.flip(-1), first_frame)
    flip_meshgrids = torch.where(is_flip.repeat([1, 2, H, W]), meshgrids.flip(-1), meshgrids)
    flip_att_map   = torch.where(is_flip.repeat([1, 1, H, W]), att_map.flip(-1), att_map)
    return flip_frame, flip_meshgrids, flip_att_map


def att_rotate_crop_with_gt(first_frame, att_map, max_degree, crop_size, downscale=16):
    """Rotate and crop the first frame and obtain the ground truth using attention.

    Args:
        first_frame: [B, C, H, W], torch.Tensor, can be on gpu
        att_map: [B, 1, H, W], torch.Tensor, can be on gpu
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
    first_frame, meshgrids, att_map = batch_flip_att(first_frame, meshgrids, att_map)
    crop, crop_gt, crop_att = batch_random_rotate_att(first_frame, meshgrids, att_map, max_degree)
    # from IPython import embed; embed(); exit()
    crop, crop_gt = batch_random_crop_att(crop, crop_gt, crop_att, crop_size)
    # crop_gt = F.interpolate(crop_gt, (crop_gt.size(2) // downscale, crop_gt.size(3) // downscale))
    scale = torch.tensor([H, W], dtype=torch.float32).view(1,-1,1,1).to(first_frame.device)

    crop_gt = (crop_gt * scale).permute(0,2,3,1)
    crop_gt = crop_gt[..., range(1,-1,-1)] # swap x and y
    crop = crop.detach()
    crop_gt = crop_gt.detach()

    # return crop.contiguous(), crop_gt.contiguous()
    return crop, crop_gt


class MoCo_Cycle(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, modelpath='', dim=128, temp=0.0007, downscale=16, rotate=10, patch_size=128):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        uselayer: use which layer to obtain feature for correspondence
        downscale: downsample scale
        temp: the temperature for affinity matrix in cycle
        norm: normalize the feature or not
        """
        super(MoCo_Cycle, self).__init__()

        self.temp = temp
        self.downscale = downscale
        self.rotate = rotate
        self.patch_size = patch_size

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)

        # restore model
        if modelpath:
            print('loading pretrained model from {}'.format(modelpath))
            ckpt = {key[len('module.encoder_q.'):]: val
                    for key, val in torch.load(modelpath)['state_dict'].items()
                    if key.startswith('module.encoder_q.')}
            self.encoder_q.load_state_dict(ckpt)
            print('model loaded!')

    def self_attention(self, im_q, is_normalize=False):
        """Obtain the self attention map.

        Compute the similarity between the feature vector from the last
        layer with the last second feature map."""
        B, C, H, W = im_q.size()

        if is_normalize:
            normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            im_q_norm = normalize((im_q + 1 ) / 2)
        else:
            im_q_norm = im_q

        # embeddings
        feat_q1 = net_forward(self.encoder_q, im_q_norm, uselayer='avgpool').detach() # (B, C1)
        feat_q2 = net_forward(self.encoder_q, im_q_norm, uselayer=4).detach() # (B, C2, H_f, W_f)

        feat_q1_norm = F.normalize(feat_q1, p=2, dim=1).squeeze(-1).squeeze(-1)
        feat_q2_norm = F.normalize(feat_q2, p=2, dim=1)

        # similarity range [-1, 1]
        raw_sim = torch.einsum('bc,bchw->bhw', feat_q1_norm, feat_q2_norm)
        sim = raw_sim - raw_sim.min()
        sim = sim / sim.max() # range [0, 1]

        att_map = F.interpolate(sim[:,None], (H, W)) # (B, 1, H, W)
        return att_map

    def forward(self, im_q):
        att_map = self.self_attention(im_q)

        # preprocessing to obtain the cropped frame and GT
        crop, crop_gt = att_rotate_crop_with_gt(im_q, att_map, self.rotate, (self.patch_size, self.patch_size), self.downscale)
        return crop, crop_gt



if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='SCOT in pytorch')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
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

    parser.add_argument('--save_dir', type=str, default='./Datasets_SCOT/PF-PASCAL/crop/', help='dir to save crop and gt')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MoCo_Cycle(models.__dict__[args.arch], args.modelpath, args.moco_dim, args.temp, args.downscale, args.rotate, args.patch_size).to(device)
    model.eval()

    dset = download.load_dataset(args.dataset, args.datapath, args.thres, device, args.split)
    dataloader = DataLoader(dset, batch_size=1, num_workers=0)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for idx, data in enumerate(dataloader):
        crop, crop_gt = model(data['src_img'].cuda())

        if args.dataset == 'pfpascal':
            crop_name = os.path.splitext(data['src_imname'][0])[0] + '_crop.pt'
            crop_gt_name = os.path.splitext(data['src_imname'][0])[0] + '_crop_gt.pt'
            torch.save(crop.squeeze(0).cpu(), os.path.join(args.save_dir, crop_name))
            torch.save(crop_gt.squeeze(0).cpu(), os.path.join(args.save_dir, crop_gt_name))

        elif args.dataset == 'pfwillow':
            pre_name = os.path.splitext(data['src_imname'][0])[0]
            crop_name = pre_name + '_crop.pt'
            crop_gt_name = pre_name + '_crop_gt.pt'
            if not os.path.exists(os.path.dirname(os.path.join(args.save_dir, crop_name))):
                os.makedirs(os.path.dirname(os.path.join(args.save_dir, crop_name)))
            torch.save(crop.squeeze(0).cpu(), os.path.join(args.save_dir, crop_name))
            torch.save(crop_gt.squeeze(0).cpu(), os.path.join(args.save_dir, crop_gt_name))

        elif args.dataset == 'spair':
            src_pre_name =  os.path.splitext(data['src_imname'][0])[0]
            trg_pre_name =  os.path.splitext(data['trg_imname'][0])[0]
            crop_name = os.path.join(args.save_dir,
                '{:06d}-{}-{}_crop.gt'.format(idx+1, src_pre_name, trg_pre_name)
            )
            crop_gt_name = os.path.join(args.save_dir,
                '{:06d}-{}-{}_crop_gt.gt'.format(idx+1, src_pre_name, trg_pre_name)
            )

            torch.save(crop.squeeze(0).cpu(), crop_name)
            torch.save(crop_gt.squeeze(0).cpu(), crop_gt_name)

        print('{}/{}'.format(idx, len(dataloader)))

