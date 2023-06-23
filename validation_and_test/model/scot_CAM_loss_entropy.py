# ------------------------------------------------------------------------------
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# see CC-BY-NC-SA-4.0.md for details
# Written by Taihong Xiao
# ------------------------------------------------------------------------------

from functools import reduce
from operator import add

import torch.nn.functional as F
import torch
import gluoncvth as gcv
import math

from . import geometry
from . import util
from . import rhm_map
#from torchvision.models import resnet
from . import resnet


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

def mutual_nn_filter(correlation_matrix):
    r"""Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18)"""
    corr_src_max = torch.max(correlation_matrix, dim=2, keepdim=True)[0]
    corr_trg_max = torch.max(correlation_matrix, dim=1, keepdim=True)[0]
    corr_src_max[corr_src_max == 0] += 1e-30
    corr_trg_max[corr_trg_max == 0] += 1e-30

    corr_src = correlation_matrix / corr_src_max
    corr_trg = correlation_matrix / corr_trg_max

    return correlation_matrix * (corr_src * corr_trg)

def l1normalize(x):
    r"""L1-normalization"""
    vector_sum = torch.sum(x, dim=2, keepdim=True)
    vector_sum[vector_sum == 0] = 1.0
    return x / vector_sum

def information_entropy(correlation_matrix, rescale_factor=1):
    r"""Computes information entropy of all candidate matches"""
    bsz = correlation_matrix.size(0)

    correlation_matrix = mutual_nn_filter(correlation_matrix)

    side = int(math.sqrt(correlation_matrix.size(1)))
    new_side = side // rescale_factor

    trg2src_dist = correlation_matrix.view(bsz, -1, side, side)
    src2trg_dist = correlation_matrix.view(bsz, side, side, -1).permute(0, 3, 1, 2)

    # Squeeze distributions for reliable entropy computation
    trg2src_dist = F.interpolate(trg2src_dist, [new_side, new_side], mode='bilinear', align_corners=True)
    src2trg_dist = F.interpolate(src2trg_dist, [new_side, new_side], mode='bilinear', align_corners=True)

    src_pdf = l1normalize(trg2src_dist.view(bsz, -1, (new_side * new_side)))
    trg_pdf = l1normalize(src2trg_dist.view(bsz, -1, (new_side * new_side)))

    src_pdf[src_pdf == 0.0] = 1e-30
    trg_pdf[trg_pdf == 0.0] = 1e-30

    src_ent = (-(src_pdf * torch.log2(src_pdf)).sum(dim=2)).view(bsz, -1)
    trg_ent = (-(trg_pdf * torch.log2(trg_pdf)).sum(dim=2)).view(bsz, -1)
    score_net = (src_ent + trg_ent).mean(dim=1) / 2

    return score_net.mean()


class SCOT_CAM:
    r"""SCOT framework"""
    def __init__(self, backbone, hyperpixel_ids, benchmark, device, cam,
                 num_classes, modelpath, temp=0.0007, icycle_lw=0.0001, entropy_lw=0.0001):
        r"""Constructor for SCOT framework"""

        # Feature extraction network initialization.
        if backbone == 'resnet50':
            if len(modelpath):
                self.backbone = resnet.resnet50(pretrained=False, num_classes=128).to(device)
                ckpt = {key[len('module.encoder_q.'):]: val
                        for key, val in torch.load(modelpath)['state_dict'].items()
                        if key.startswith('module.encoder_q.')}
                self.backbone.load_state_dict(ckpt)
            else:
                self.backbone = resnet.resnet50(pretrained=True).to(device)
            nbottlenecks = [3, 4, 6, 3]

        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True).to(device)
            nbottlenecks = [3, 4, 23, 3]

        elif backbone == 'fcn101':
            self.backbone = gcv.models.get_fcn_resnet101_voc(pretrained=True).to(device).pretrained
            if len(cam)==0:
                self.backbone1 = gcv.models.get_fcn_resnet101_voc(pretrained=True).to(device)
                self.backbone1.eval()
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.backbone.eval()

        # Hyperpixel id and pre-computed jump and receptive field size initialization
        # Reference: https://fomoro.com/research/article/receptive-field-calculator
        # (the jump and receptive field sizes for 'fcn101' are heuristic values)
        self.hyperpixel_ids = util.parse_hyperpixel(hyperpixel_ids)
        if backbone in ['resnet50', 'resnet101']:
            self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 32, 32]).to(device)
            self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139, 171, 203, 235, 267, 299, 363, 427]).to(device)
        elif backbone in ['resnet50_ft', 'resnet101_ft']:
            self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 8, 8]).to(device)
            self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139]).to(device)
        else:
            self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 8, 8]).to(device)
            self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139]).to(device)

        # Miscellaneous
        self.hsfilter = geometry.gaussian2d(7).to(device)
        self.device = device
        self.benchmark = benchmark

        self.temp = temp
        self.icycle_lw = icycle_lw
        self.entropy_lw = entropy_lw


    def __call__(self, *args, **kwargs):
        r"""Forward pass"""
        maptype = args[6]
        src_bbox = args[7]
        trg_bbox = args[8]
        src_mask = args[9]
        trg_mask = args[10]
        backbone = args[11]
        crop = args[12]
        crop_gt = args[13]
        src_hyperpixels = self.extract_hyperpixel(args[0], maptype, src_bbox, src_mask, backbone)
        trg_hyperpixels = self.extract_hyperpixel(args[1], maptype, trg_bbox, trg_mask, backbone)
        crop_hyperpixels = self.extract_hyperpixel(crop[0], maptype, trg_bbox, trg_mask, backbone)

        N0, C0 = crop_hyperpixels[1].size()
        N1, C1 = src_hyperpixels[1].size()
        H0 = int(math.sqrt(N0))
        H1 = int(math.sqrt(N1))
        downscale = args[0].size(-1) / H1

        A02 = torch.einsum('pc,qc->pq', crop_hyperpixels[1], trg_hyperpixels[1]) / self.temp
        A21 = torch.einsum('qc,rc->qr', trg_hyperpixels[1], src_hyperpixels[1]) / self.temp
        At = torch.einsum('pq,qr->pr', A02.softmax(-1), A21.softmax(-1))

        grid_flat = create_flat_grid([1, C1, H1, H1])
        coords = downscale * torch.einsum('pq,bqr->bpr', At, grid_flat)

        # resize the crop_gt to fit the current feature size
        crop_gt_resize = F.interpolate(crop_gt.permute(0,3,1,2) / args[0].size(-1), (H0, H0)) * args[0].size(-1)
        crop_gt_resize = crop_gt_resize.permute(0,2,3,1)
        crop_label = crop_gt_resize.view(crop_gt_resize.size(0), -1, crop_gt_resize.size(-1))
        cycle_loss = F.mse_loss(coords, crop_label) * self.icycle_lw

        Aff = (A21 * self.temp).unsqueeze(0)
        entropy_loss = information_entropy(F.relu(Aff).pow(2)) * self.entropy_lw
        loss = cycle_loss + entropy_loss

        # a = F.mse_loss(coords, crop_label) # average of each pixel
        # b = torch.norm(coords- crop_label, p=2)
        # c = torch.sum((coords- crop_label) ** 2 ) / coords.view(-1).size(0)
        # d = torch.sqrt(torch.sum((coords- crop_label) ** 2, 2)).mean()

        return -loss
        # confidence_ts = rhm_map.rhm(src_hyperpixels, trg_hyperpixels, self.hsfilter, args[2], args[3], args[4], args[5])
        # return confidence_ts, src_hyperpixels[0], trg_hyperpixels[0]

    def compute_aff(feat1, feat2):
        C, N = feat1.size()
        feat_norm1 = torch.norm(feat1, p=2, dim=0)
        feat_norm2 = torch.norm(feat2, p=2, dim=0)
        aff = torch.einsum('cp,cq->pq', feat_norm1, feat_norm2) / self.temp




    def extract_hyperpixel(self, img, maptype, bbox, mask, backbone="resnet101"):
        r"""Given image, extract desired list of hyperpixels"""
        hyperfeats, rfsz, jsz, feat_map, fc = self.extract_intermediate_feat(img.unsqueeze(0), return_hp=True, backbone=backbone)
        hpgeometry = geometry.receptive_fields(rfsz, jsz, hyperfeats.size()).to(self.device)
        hyperfeats = hyperfeats.view(hyperfeats.size()[0], -1).t()

        # Prune boxes on margins (Otherwise may cause error)
        if self.benchmark in ['TSS']:
            hpgeometry, valid_ids = geometry.prune_margin(hpgeometry, img.size()[1:], 10)
            hyperfeats = hyperfeats[valid_ids, :]

        weights = torch.ones(len(hyperfeats),1).to(hyperfeats.device)
        if maptype in [1]: # weight points
            if mask is None:
                # get CAM mask
                if backbone=='fcn101':
                    mask = self.get_FCN_map(img.unsqueeze(0), feat_map, fc, sz=(img.size(1),img.size(2)))
                else:
                    mask = self.get_CAM_multi(img.unsqueeze(0), feat_map, fc, sz=(img.size(1),img.size(2)), top_k=2)
                scale = 1.0
            else:
                scale = 255.0

            hpos = geometry.center(hpgeometry)
            hselect = mask[hpos[:,1].long(),hpos[:,0].long()].to(hpos.device)
            weights = 0.5*torch.ones(len(hyperfeats),1).to(hpos.device)

            weights[hselect>0.4*scale,:] = 0.8
            weights[hselect>0.5*scale,:] = 0.9
            weights[hselect>0.6*scale,:] = 1.0
        
        return hpgeometry, hyperfeats, img.size()[1:][::-1], weights


    def extract_intermediate_feat(self, img, return_hp=True, backbone='resnet101'):
        r"""Extract desired a list of intermediate features"""

        feats = []
        rfsz = self.rfsz[self.hyperpixel_ids[0]]
        jsz = self.jsz[self.hyperpixel_ids[0]]

        # Layer 0
        feat = self.backbone.conv1.forward(img)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)
        if 0 in self.hyperpixel_ids:
            feats.append(feat.clone())

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(self.bottleneck_ids, self.layer_ids)):
            res = feat
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

            if bid == 0:
                res = self.backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

            feat += res

            if hid + 1 in self.hyperpixel_ids:
                feats.append(feat.clone())
                #if hid + 1 == max(self.hyperpixel_ids):
                #    break
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        # GAP feature map
        feat_map = feat
        if backbone!='fcn101':
            x = self.backbone.avgpool(feat)
            x = torch.flatten(x, 1)
            fc = self.backbone.fc(x)
        else:
            fc = None

        if not return_hp: # only return feat_map and fc
            return feat_map,fc

        # Up-sample & concatenate features to construct a hyperimage
        for idx, feat in enumerate(feats):
            if idx == 0:
                continue
            feats[idx] = F.interpolate(feat, tuple(feats[0].size()[2:]), None, 'bilinear', True)
        feats = torch.cat(feats, dim=1)

        return feats[0], rfsz, jsz, feat_map, fc
    

    def get_CAM(self, feat_map, fc, sz, top_k=2):
        logits = F.softmax(fc, dim=1)
        scores, pred_labels = torch.topk(logits, k=top_k, dim=1)
        pred_labels = pred_labels[0]
        bz, nc, h, w = feat_map.size()

        output_cam = []
        for label in pred_labels:
            cam = self.backbone.fc.weight[label,:].unsqueeze(0).mm(feat_map.view(nc,h*w))
            cam = cam.view(1,1,h,w)
            cam = F.interpolate(cam, (sz[0],sz[1]), None, 'bilinear', True)[0,0] # HxW
            cam = (cam-cam.min()) / cam.max()
            output_cam.append(cam)
        output_cam = torch.stack(output_cam,dim=0) # kxHxW
        output_cam = output_cam.max(dim=0)[0] # HxW

        return output_cam


    def get_CAM_multi(self, img, feat_map, fc, sz, top_k=2):
        scales = [1.0,1.5,2.0]
        map_list = []
        for scale in scales:
            if scale>1.0:
                if scale*scale*sz[0]*sz[1] > 800*800:
                    scale = min(800/img_h,800/img_w)
                    scale = min(1.5,scale)
                img = F.interpolate(img, (int(scale*sz[0]),int(scale*sz[1])), None, 'bilinear', True) # 1x3xHxW
                feat_map, fc = self.extract_intermediate_feat(img,return_hp=False)

            logits = F.softmax(fc, dim=1)
            scores, pred_labels = torch.topk(logits, k=top_k, dim=1)
            pred_labels = pred_labels[0]
            bz, nc, h, w = feat_map.size()

            output_cam = []
            for label in pred_labels:
                cam = self.backbone.fc.weight[label,:].unsqueeze(0).mm(feat_map.view(nc,h*w))
                cam = cam.view(1,1,h,w)
                cam = F.interpolate(cam, (sz[0],sz[1]), None, 'bilinear', True)[0,0] # HxW
                cam = (cam-cam.min()) / cam.max()
                output_cam.append(cam)
            output_cam = torch.stack(output_cam,dim=0) # kxHxW
            output_cam = output_cam.max(dim=0)[0] # HxW
            
            map_list.append(output_cam)
        map_list = torch.stack(map_list,dim=0)
        sum_cam = map_list.sum(0)
        norm_cam = sum_cam / (sum_cam.max()+1e-5)

        return norm_cam


    def get_FCN_map(self, img, feat_map, fc, sz):
        #scales = [1.0,1.5,2.0]
        scales = [1.0]
        map_list = []
        for scale in scales:
            if scale*scale*sz[0]*sz[1] > 1200*800:
                scale = 1.5
            img = F.interpolate(img, (int(scale*sz[0]),int(scale*sz[1])), None, 'bilinear', True) # 1x3xHxW
            #feat_map, fc = self.extract_intermediate_feat(img,return_hp=False,backbone='fcn101')
            feat_map = self.backbone1.evaluate(img)
            
            predict = torch.max(feat_map, 1)[1]
            mask = predict-torch.min(predict)
            mask_map = mask / torch.max(mask)
            mask_map = F.interpolate(mask_map.unsqueeze(0).double(), (sz[0],sz[1]), None, 'bilinear', True)[0,0] # HxW
    
        return mask_map
