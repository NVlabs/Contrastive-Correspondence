# Copyright (c) 2023, Yanbin Liu
# All rights reserved.
#
# BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. All advertising materials mentioning features or use of this software
#    must display the following acknowledgement:
#    This product includes software developed by the <organization>.
# 4. Neither the name of the <organization> nor the
#    names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE 
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Source: https://github.com/csyanbin/SCOT
# Modified by Taihong Xiao

import logging
import re

import torch.nn.functional as F
import torch


def init_logger(logfile):
    r"""Initialize logging settings"""
    logging.basicConfig(filemode='w',
                        filename=logfile,
                        level=logging.INFO,
                        format='%(message)s',
                        datefmt='%m-%d %H:%M:%S')

    # Configuration on console logs
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_args(args):
    r"""Log program arguments"""
    logging.info('\n+========== SCOT Arguments ===========+')
    for arg_key in args.__dict__:
        logging.info('| %20s: %-24s |' % (arg_key, str(args.__dict__[arg_key])))
    logging.info('+================================================+\n')


def where(predicate):
    r"""Returns indices which match given predicate"""
    matching_idx = predicate.nonzero()
    n_match = len(matching_idx)
    if n_match != 0:
        matching_idx = matching_idx.t().squeeze(0)
    return matching_idx


def intersect1d(tensor1, tensor2):
    r"""Takes two 1D tensor and returns tensor of common values"""
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]


def parse_hyperpixel(hyperpixel_ids):
    r"""Parse given hyperpixel list (string -> int)"""
    return list(map(int, re.findall(r'\d+', hyperpixel_ids)))


def get_bbox_mask(mask, thres=0.0):
    """mask:HxW"""
    pos = (mask>=thres).nonzero()
    pmin = pos.min(0)[0] # top-left
    pmax = pos.max(0)[0] # bottom-right
    bbox = torch.tensor([pmin[1],pmin[0],pmax[1],pmax[0]])

    return bbox


def resize(img, kps, side_thres=300):
    r"""Resize given image with imsize: (1, 3, H, W)"""
    imsize = torch.tensor(img.size()).float()
    kps = kps.float()
    side_max = torch.max(imsize)
    inter_ratio = 1.0
    #if side_max > side_thres:
    if True:
        inter_ratio = side_thres / side_max
        img = F.interpolate(img,
                            size=(int(imsize[2] * inter_ratio), int(imsize[3] * inter_ratio)),
                            mode='bilinear',
                            align_corners=False)
        kps *= inter_ratio
    return img.squeeze(0), kps, inter_ratio


def resize_mask(mask, imsize):
    mask = mask.unsqueeze(0).float()
    mask = F.interpolate(mask, size=(imsize[1],imsize[2]),
                        mode='bilinear', align_corners=False)
    return mask[0][0]


def resize_TSS(img, side_thres=300):
    r"""Resize given image with imsize: (1, 3, H, W)"""
    imsize = torch.tensor(img.size()).float()
    side_max = torch.max(imsize)
    inter_ratio = 1.0
    #if side_max > side_thres:
    if True:
        inter_ratio = side_thres / side_max
        img = F.interpolate(img,
                            size=(int(imsize[2] * inter_ratio), int(imsize[3] * inter_ratio)),
                            mode='bilinear',
                            align_corners=False)
        return img.squeeze(0), imsize[1:], img.size()[1:], inter_ratio


