# -------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Modified by Taihong Xiao
# -------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from functools import reduce
from operator import add


class Normalize(nn.Module):
	"""Given mean: (R, G, B) and std: (R, G, B),
	will normalize each channel of the torch.*Tensor, i.e.
	channel = (channel - mean) / std
	"""

	def __init__(self, mean, std = (1.0,1.0,1.0)):
		super(Normalize, self).__init__()
		self.mean = nn.Parameter(torch.FloatTensor(mean).cuda(), requires_grad=False)
		self.std = nn.Parameter(torch.FloatTensor(std).cuda(), requires_grad=False)

	def forward(self, frames):
		b,c,h,w = frames.size()
		frames = (frames - self.mean.view(1,3,1,1).repeat(b,1,h,w))/self.std.view(1,3,1,1).repeat(b,1,h,w)
		return frames


def create_flat_grid(F_size, GPU=True):
	"""
	INPUTS:
	 - F_size: feature size
	OUTPUT:
	 - return a standard grid coordinate
	"""
	b, c, h, w = F_size
	theta = torch.tensor([[1,0,0],[0,1,0]])
	theta = theta.unsqueeze(0).repeat(b,1,1)
	theta = theta.float()

	# grid is a uniform grid with left top (-1,1) and right bottom (1,1)
	# b * (h*w) * 2
	grid = torch.nn.functional.affine_grid(theta, F_size)
	grid[:,:,:,0] = (grid[:,:,:,0]+1)/2 * w
	grid[:,:,:,1] = (grid[:,:,:,1]+1)/2 * h
	grid_flat = grid.view(b,-1,2)
	if(GPU):
		grid_flat = grid_flat.cuda()
	return grid_flat


def coords2bbox(coords, patch_size, h_tar, w_tar):
	"""
	INPUTS:
	 - coords: coordinates of pixels in the next frame
	 - patch_size: patch size
	 - h_tar: target image height
	 - w_tar: target image widthg
	"""
	b = coords.size(0)
	center = torch.mean(coords, dim=1) # b * 2
	center_repeat = center.unsqueeze(1).repeat(1,coords.size(1),1)
	dis_x = torch.sqrt(torch.pow(coords[:,:,0] - center_repeat[:,:,0], 2))
	dis_x = torch.mean(dis_x, dim=1).detach()
	dis_y = torch.sqrt(torch.pow(coords[:,:,1] - center_repeat[:,:,1], 2))
	dis_y = torch.mean(dis_y, dim=1).detach()
	left = (center[:,0] - dis_x*2).view(b,1)
	left[left < 0] = 0
	right = (center[:,0] + dis_x*2).view(b,1)
	right[right > w_tar] = w_tar
	top = (center[:,1] - dis_y*2).view(b,1)
	top[top < 0] = 0
	bottom = (center[:,1] + dis_y*2).view(b,1)
	bottom[bottom > h_tar] = h_tar
	new_center = torch.cat((left,right,top,bottom),dim=1)
	return new_center


class NetExtension(nn.Module):
    def __init__(self, net):
        super(NetExtension, self).__init__()
        # self.net = net
        self.modules_list = list(net.children())
        self.subnet = nn.Sequential(*self.modules_list[:-3])
        self.layer4 = nn.Sequential(*self.modules_list[-3:-2])
        self.layer5_1 = nn.Sequential(*self.modules_list[-2:-1])
        self.layer5_2 = nn.Sequential(*self.modules_list[-1:])

    def forward(self, x, uselayer=5):
        x = self.subnet(x)
        if uselayer == 3:
            return x

        x = self.layer4(x)
        if uselayer == 4:
            return x

        x = self.layer5_1(x)
        x = torch.flatten(x, 1)
        x = self.layer5_2(x)
        if uselayer == 5:
            return x


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

def extract_feat(net, x, uselayer=13):
    nbottlenecks = [3, 4, 6, 3]
    bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
    layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])

    b_id = bottleneck_ids[uselayer-1]
    l_id = layer_ids[uselayer-1]

    x = net.conv1(x)
    x = net.bn1(x)
    x = net.relu(x)
    x = net.maxpool(x)

    for i in range(1, l_id):
        x = net.__getattr__('layer%d' % i)(x)

    x = net.__getattr__('layer%d' % l_id)[:b_id+1](x)

    return x


class MoCo_Cycle(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, uselayer=13, downscale=16, temp=0.0007, norm=True):
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

        self.K = K
        self.m = m
        self.T = T
        self.uselayer = uselayer
        self.downscale = downscale
        self.temp = temp
        self.norm = norm

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward_moco(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

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

        if self.norm:
            feat_q1_norm = F.normalize(feat_q1, p=2, dim=1).squeeze()
            feat_q2_norm = F.normalize(feat_q2, p=2, dim=1)

        # similarity range [-1, 1]
        raw_sim = torch.einsum('bc,bchw->bhw', feat_q1_norm, feat_q2_norm)
        sim = raw_sim - raw_sim.min()
        sim = sim / sim.max() # range [0, 1]

        att_map = F.interpolate(sim[:,None], (H, W)) # (B, 1, H, W)
        return att_map


    def forward_cycle(self, im_q, im_k, rotate, patch_size, dropout_rate=0, is_normalize=False):
        att_map = self.self_attention(im_q)

        # preprocessing to obtain the cropped frame and GT
        crop, crop_gt = att_rotate_crop_with_gt(im_q, att_map, rotate, (patch_size, patch_size), self.downscale)

        B, C, H0, W0 = crop.size()
        B, C, H1, W1 = im_q.size()
        B, C, H2, W2 = im_k.size()

        if is_normalize:
            normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            crop_norm = normalize((crop + 1 ) / 2)
            im_q_norm = normalize((im_q + 1 ) / 2)
            im_k_norm = normalize((im_k + 1 ) / 2)

        else:
            crop_norm = crop
            im_q_norm = im_q
            im_k_norm = im_k

        # Embeddings
        # feats0 = net_forward(self.encoder_q, crop_norm, uselayer=3)
        # feats1 = net_forward(self.encoder_q, im_q_norm, uselayer=3)
        # feats2 = net_forward(self.encoder_q, im_k_norm, uselayer=3)

        feats0 = extract_feat(self.encoder_q, crop_norm, uselayer=self.uselayer)
        feats1 = extract_feat(self.encoder_q, im_q_norm, uselayer=self.uselayer)
        feats2 = extract_feat(self.encoder_q, im_k_norm, uselayer=self.uselayer)
        # from IPython import embed; embed(); exit()

        B, C_f, H_f0, W_f0 = feats0.size()
        B, C_f, H_f1, W_f1 = feats1.size()
        B, C_f, H_f2, W_f2 = feats2.size()

        if self.norm:
            # normalize and reshape to (B, C_f, H_f * W_f)
            feats0 = F.normalize(feats0, p=2, dim=1).view(B, C_f, -1)
            feats1 = F.normalize(feats1, p=2, dim=1).view(B, C_f, -1)
            feats2 = F.normalize(feats2, p=2, dim=1).view(B, C_f, -1)
        else:
            feats0 = feats0.view(B, C_f, -1)
            feats1 = feats1.view(B, C_f, -1)
            feats2 = feats2.view(B, C_f, -1)

        # Transitions from cropped frame to frame 2 and backward from frame 2 to frame 1
        A02 = torch.einsum('bcp,bcq->bpq', feats0, feats2) / self.temp # affinity between crop frame1 and frame 2
        A21 = torch.einsum('bcp,bcq->bpq', feats2, feats1) / self.temp # affinity between consecutive frames
        A01 = torch.einsum('bcp,bcq->bpq', feats0, feats1) / self.temp # affinity between consecutive frames

        # walk and coordinate propagation
        At = torch.einsum('bpq,bqr->bpr', F.dropout(A02, p=dropout_rate).softmax(-1),
                          F.dropout(A21, p=dropout_rate).softmax(-1))

        grid_flat1 = create_flat_grid([B, C_f, H_f1, W_f1])
        grid_flat2 = create_flat_grid([B, C_f, H_f2, W_f2])
        coords10_cycle = self.downscale * torch.einsum('bpq,bqr->bpr', At, grid_flat1)
        coords10       = self.downscale * torch.einsum('bpq,bqr->bpr', A01.softmax(-1), grid_flat1)
        coords20       = self.downscale * torch.einsum('bpq,bqr->bpr', A02.softmax(-1), grid_flat2)

        aff_list = [
            F.dropout(A01, p=dropout_rate).softmax(-1),
            At,
        ]

        coords_list = [
            coords10_cycle.view(B, 2, H_f0, W_f0),
            coords10.view(B, 2, H_f0, W_f0),
            coords20.view(B, 2, H_f0, W_f0),
        ]

        return crop, crop_gt, att_map, At, coords10_cycle, coords10, coords20, aff_list, coords_list


    def forward(self, im_q, im_k, src_img, trg_img, rotate, patch_size, dropout_rate=0):
        """
        Input:
            im_q: a batch of query images [B x 3 x args.image_size x args.image_size]
            im_k: a batch of key images [B x 3 x args.image_size x args.image_size]
            topk: retrieving topk images
            rotate: max degree for rotation
            patch_size: crop size for image cycle
        """
        moco_output = self.forward_moco(im_q, im_k)
        image_cycle = self.forward_cycle(src_img, trg_img, rotate, patch_size, dropout_rate, False)
        return moco_output, image_cycle


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def mask_from_5p_lmk(lmk, img_size, feat_size, sigma=1):
    """Creates one-hot mask from 5p landmarks.

    Args:
        lmk: landmark of query image of shape [B, 5, 2]
        img_size: [H, W]
        feat_size:[H_f, W_f]

    Return:
        mask: one-hot mask label, [B, 5, H_f, W_f]
    """
    B, C, _ = lmk.size()
    H, W = img_size
    H_f, W_f = feat_size
    mask = torch.zeros(B, C, H, W).to(lmk.device)
    # grid_h, grid_w = torch.meshgrid(torch.arange(H), torch.arange(W))
    grid_hw = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=-1).to(lmk.device) # shape [H, W, 2]
    grids_hw = torch.stack([grid_hw] * C, dim=0)
    mask_list = []
    for i in range(B):
        centers = lmk[i].view(C, 1, 1, -1)
        mask_i = torch.exp(-torch.norm(grids_hw - centers, dim=-1) ** 2 / (2 * sigma ** 2))
        # mask_i = 1 - torch.sign(torch.norm(grids_hw - centers, dim=-1))
        mask_list.append(mask_i)
    mask = torch.stack(mask_list, 0)
    mask_resize = F.interpolate(mask, (H_f, W_f))
    # rescale to [0, 1]
    scale = mask_resize.view(B, C, -1).max(-1)[0].view(B, C, 1, 1)
    return mask_resize / scale

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
    crop_gt = F.interpolate(crop_gt, (crop_gt.size(2) // downscale, crop_gt.size(3) // downscale))
    scale = torch.tensor([H, W], dtype=torch.float32).view(1,-1,1,1).to(first_frame.device)

    crop_gt = (crop_gt * scale).permute(0,2,3,1)
    crop_gt = crop_gt[..., range(1,-1,-1)] # swap x and y
    crop = crop.detach()
    crop_gt = crop_gt.detach()

    # return crop.contiguous(), crop_gt.contiguous()
    return crop, crop_gt

