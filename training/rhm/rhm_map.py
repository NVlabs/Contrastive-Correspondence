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

import math
import torch
import torch.nn.functional as F

from . import geometry


def appearance_similarity(src_feats, trg_feats, exp1=3):
    r"""Semantic appearance similarity (exponentiated cosine)

    Args:
        src_feats: shape [B, P, C]
        trg_feats: shape [B, Q, C]

    Return:
        sim: shape [B, P, Q]
    """
    src_feats_norm = F.normalize(src_feats, p=2, dim=2)
    trg_feats_norm = F.normalize(trg_feats, p=2, dim=2)
    sim = torch.einsum('bpc,bqc->bpq', src_feats_norm, trg_feats_norm)
    sim = torch.pow(torch.clamp(sim, min=0), exp1)
    return sim


def perform_sinkhorn(C, epsilon, mu, nu, warm=False, niter=1, tol=10e-9):
    """Main Sinkhorn Algorithm"""
    if not warm:
        a = torch.ones((C.shape[0],C.shape[1])) / C.shape[1]
        a = a.to(C.device) # (b, m)

    K = torch.exp(-C/epsilon) # (b, m, n)

    Err = torch.zeros((niter,2)).cuda()
    for i in range(niter):
        b = nu/torch.einsum('bmn,bm->bn', K, a) # (b, n)
        if i%2==0:
            Err[i,0] = torch.norm(a*(torch.einsum('bmn,bn->bm', K, b)) - mu, p=1) # (b, m)
            if i>0 and (Err[i,0]) < tol:
                break

        a = mu / torch.einsum('bmn,bn->bm', K, b) # (b, m)

        if i%2==0:
            Err[i,1] = torch.norm(b*(torch.einsum('bmn,bm->bn', K, a)) - nu, p=1)
            if i>0 and (Err[i,1]) < tol:
                break

        PI = torch.bmm(torch.bmm(torch.diag_embed(a), K), torch.diag_embed(b))
        # PI = torch.mm(torch.mm(torch.diag(a[:,-1]),K), torch.diag(b[:,-1]))

    del a; del b; del K
    return PI,mu,nu,Err


def appearance_similarityOT(src_feats, trg_feats, exp1=1.0, exp2=1.0, eps=0.05, src_weights=None, trg_weights=None):
    r"""Semantic appearance similarity (exponentiated cosine)

    Args:
        src_feats: shape [B, P, C]
        trg_feats: shape [B, Q, C]
        src_weights: shape [B, N]
        trg_weights: shape [B, N]

    Return:
        PI: shape [B, P, Q]
    """

    sim = appearance_similarity(src_feats, trg_feats, exp1)
    cost = 1 - sim

    b, m, n = cost.shape
    mu = (torch.ones((b,m))/m).to(cost.device)
    mu = src_weights / src_weights.sum(1, keepdim=True)

    nu = (torch.ones((b,n))/n).to(cost.device)
    nu = trg_weights / trg_weights.sum(1, keepdim=True)
    
    ## ---- <Run Optimal Transport Algorithm> ----
    #mu = mu.unsqueeze(1)
    #nu = nu.unsqueeze(1)
    # with torch.no_grad():

    # if torch.isnan(sim).any():
    #     from IPython import embed; embed(); exit()

    epsilon = eps
    cnt = 0
    while True:
        PI,a,b,err = perform_sinkhorn(cost, epsilon, mu, nu)
        #PI = sinkhorn_stabilized(mu, nu, cost, reg=epsilon, numItermax=50, method='sinkhorn_stabilized', cuda=True)
        if not torch.isnan(PI).any():
            # if cnt>0:
            #     print(cnt)
            break
        else: # Nan encountered caused by overflow issue is sinkhorn
            # print(cnt, PI)
            # from IPython import embed; embed(); exit()
            epsilon *= 2.0
            cnt += 1

    PI = m*PI # re-scale PI
    #exp2 = 1.0 for spair-71k, TSS
    #exp2 = 0.5 # for pf-pascal and pfwillow
    PI = torch.pow(torch.clamp(PI, min=0), exp2)
    # print('sinkhorn run successfully')
    return PI


def hspace_bin_ids(src_imsize, src_box, trg_box, hs_cellsize, nbins_x):
    r"""Compute Hough space bin id for the subsequent voting procedure"""
    src_ptref = torch.tensor(src_imsize, dtype=torch.float).to(src_box.device)
    src_trans = geometry.center(src_box)
    trg_trans = geometry.center(trg_box)
    xy_vote = (src_ptref.unsqueeze(0).expand_as(src_trans) - src_trans).unsqueeze(2).\
        repeat(1, 1, len(trg_box)) + \
        trg_trans.t().unsqueeze(0).repeat(len(src_box), 1, 1)

    bin_ids = (xy_vote / hs_cellsize).long()

    return bin_ids[:, 0, :] + bin_ids[:, 1, :] * nbins_x


def build_hspace(src_imsize, trg_imsize, ncells):
    r"""Build Hough space where voting is done"""
    hs_width = src_imsize[0] + trg_imsize[0]
    hs_height = src_imsize[1] + trg_imsize[1]
    hs_cellsize = math.sqrt((hs_width * hs_height) / ncells)
    nbins_x = int(hs_width / hs_cellsize) + 1
    nbins_y = int(hs_height / hs_cellsize) + 1

    return nbins_x, nbins_y, hs_cellsize


def rhm(src_hyperpixels, trg_hyperpixels, hsfilter, sim, exp1, exp2, eps, ncells=8192):
    r"""Regularized Hough matching"""
    # Unpack hyperpixels
    src_hpgeomt, src_hpfeats, src_imsize, src_feats_size, src_weights = src_hyperpixels
    trg_hpgeomt, trg_hpfeats, trg_imsize, trg_feats_size, trg_weights = trg_hyperpixels

    # Prepare for the voting procedure
    if sim in ['cos', 'cosGeo', 'Geo']:
        votes = appearance_similarity(src_hpfeats, trg_hpfeats, exp1)
    if sim in ['OT', 'OTGeo']:
        votes = appearance_similarityOT(src_hpfeats, trg_hpfeats, exp1, exp2, eps, src_weights, trg_weights)
    if sim in ['OT', 'cos', 'cos2']:
        return votes

    bs = votes.size(0)
    nbins_x, nbins_y, hs_cellsize = build_hspace(src_imsize, trg_imsize, ncells)

    bin_ids = hspace_bin_ids(src_imsize, src_hpgeomt, trg_hpgeomt, hs_cellsize, nbins_x)
    hspace = src_hpgeomt.new_zeros((bs, votes.size(1), nbins_y * nbins_x))

    # Proceed voting
    hbin_ids = bin_ids.add(torch.arange(0, votes.size(1)).to(src_hpgeomt.device).
                           mul(hspace.size(1)).unsqueeze(1).expand_as(bin_ids)) # shape [P, Q]
    hspace = hspace.view(bs, -1).index_add(1, hbin_ids.view(-1), votes.view(bs, -1)).view_as(hspace)
    hspace = torch.sum(hspace, dim=1)

    # Aggregate the voting results
    hspace = F.conv2d(hspace.view(bs, 1, nbins_y, nbins_x),
                      hsfilter.unsqueeze(0).unsqueeze(0), padding=3).view_as(hspace)

    # print('rhm run successfully!')
    return votes * torch.index_select(hspace, dim=1, index=bin_ids.view(-1)).view_as(votes)

