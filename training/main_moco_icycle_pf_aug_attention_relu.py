# ------------------------------------------------------------------------------
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# see CC-BY-NC-SA-4.0.md for details
# Written by Taihong Xiao
# ------------------------------------------------------------------------------

import os
import cv2
import sys
import time
import math
import shutil
import builtins
import torch
import logging
import argparse
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.models as models


from libs.loader import ZipDataset
from libs.concentration_loss import CenterLoss

import torch.multiprocessing as mp
import moco.loader
import moco.builder_moco_icycle_pf_aug_attention_relu

torch.multiprocessing.set_sharing_strategy('file_system')
# torch.autograd.set_detect_anomaly(True)


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='')
# file/folder pathes
parser.add_argument("--videoRoot", type=str, default="/Kinetics/compress/train_256/", help='train video path')
parser.add_argument("--videoList", type=str, default="/Kinetics/compress/train.txt", help='train video list (after "train_256")')
parser.add_argument('--resume', type=str, default='', metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--restore', type=str, default='', metavar='PATH', help='path to restore checkpoint from the last stage (default: none)')
parser.add_argument("-c","--savedir",type=str,default="match_track_comb/",help='checkpoints path')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument("--uselayer", type=int, default=13, help="choose layer blocks.")
parser.add_argument("--downscale", type=int, help='downsample scale.')

# PF dataset arguments
parser.add_argument('--datapath', type=str, default='./Datasets_SCOT')
parser.add_argument('--dataset', type=str, default='pfpascal', choices=['pfpascal', 'pfwillow', 'spair'])
parser.add_argument('--split', type=str, default='test', choices=['trn', 'val', 'test'])
parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
parser.add_argument('--cam', type=str, default='', help='activation map folder, empty for end2end computation')


# main parameters
parser.add_argument("--pretrainRes",action="store_true")
parser.add_argument("--gn", action="store_true", help='use group normalization')
parser.add_argument("--batch_size",type=int, default=1, help="batch size")
parser.add_argument('--workers', type=int, default=16)
parser.add_argument("--image_size", type=int, default=224, help="image size for image branch (moco).")
parser.add_argument("--patch_size", type=int, default=256, help="crop size for localization.")
parser.add_argument("--patch_size1", type=int, default=128, help="crop size for localization for the image cycle branch.")
parser.add_argument("--full_size", type=int, default=640, help="full size for one frame.")
parser.add_argument("--rotate",type=int,default=10,help='degree to rotate training images')
parser.add_argument("--scale",type=float,default=1.2,help='random scale')
parser.add_argument("--lr",type=float,default=0.0001,help='learning rate')
parser.add_argument('--lr_mode', type=str, default='step', choices=['poly', 'step', 'cos'])
parser.add_argument('--lr_schedule', type=int, default=[120, 160], nargs='*', help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument("--window_len",type=int,default=2,help='number of images (2 for pair and 3 for triple)')
parser.add_argument("--log_interval",type=int,default=1000,help='')
parser.add_argument("--save_interval",type=int,default=1000,help='save every x iteration')
parser.add_argument("--momentum", type=float, default=0.9, help='momentum for SGD solver')
parser.add_argument("--weight_decay",type=float,default=0.0001,help='weight decay')
parser.add_argument("--device", type=int, default=4, help="0~device_count-1 for single GPU, device_count for dataparallel.")
parser.add_argument("--nepoch", type=int, default=200, help='max epoch')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

# moco argument
parser.add_argument("--aug_plus", action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--moco_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco_k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco_t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--moco_lw', default=1, type=float,
                    help='moco loss weight')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

# cycle argument
parser.add_argument("--norm", default=True, type=lambda x: (str(x).lower() == 'true'),
                    help="normalize feature or not")
parser.add_argument("--frame_num", type=int, default=4,
                    help="number of frames for dataloader.")
parser.add_argument("--dropout_rate", type=float, default=0.1,
                    help="dropout rate.")
parser.add_argument("--temp", type=float, default=0.0007,
                    help="temprature for cycle.")
parser.add_argument('--vcycle_lw', default=0.0001, type=float,
                    help='video cycle loss weight')
parser.add_argument('--icycle_lw', default=0.0001, type=float,
                    help='image cycle loss weight')
parser.add_argument('--topk', default=3, type=int,
                    help='retrieving topk images in the image cycle')
parser.add_argument('--exclude_self', action='store_true',
                    help='excluding the top1 retrieved image in the image cycle')

# concenration regularization
parser.add_argument("--lc",type=float,default=1e4, help='weight of concentration loss')
parser.add_argument("--lc_win",type=int,default=8, help='win_len for concentration loss')

# others
parser.add_argument("--mode", type=str, default='train', choices=['train', 'test'], help="mode.")
parser.add_argument("--image_dir", type=str, default=None, help="Image dir.")
parser.add_argument("--style_dir", type=str, default=None, help="Style transfered image dir.")

# distributed training
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                    'N processes per node, which has N GPUs. This is the '
                    'fastest way to use PyTorch for either single node or '
                    'multi node data parallel training')



def main():
    print("Begin parser arguments.")
    args = parser.parse_args()
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    args.ckpt_dir = os.path.join(args.savedir, 'ckpt')
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    if args.mode == 'train':
        args.savepatch = os.path.join(args.savedir, 'savepatch')

    if not os.path.exists(args.savepatch):
        os.makedirs(args.savepatch)

    # print args and write them into log file.
    print(' '.join(sys.argv))
    print('\n')
    logfile = open(os.path.join(args.savedir, "logargs.txt"), "w")
    logfile.write(' '.join(sys.argv))
    logfile.write('\n')

    for k, v in args.__dict__.items():
        print(k, ':', v)
        logfile.write('{}:{}\n'.format(k,v))
    logfile.close()

    args.multiGPU = (args.device == torch.cuda.device_count())
    args.gpu = None

    if not args.multiGPU:
        torch.cuda.set_device(args.device)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    else:
        args.gpu = 0 # for debugging on a single gpu

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        # main_worker(args.gpu, ngpus_per_node, args) # for debug use
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder_moco_icycle_pf_aug_attention_relu.MoCo_Cycle(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, args.uselayer, args.downscale, args.temp, args.norm)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    closs = CenterLoss(win_len=args.lc_win, stride=args.lc_win)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # optionally restore from a checkpoint (from previous training stage)
    if args.restore:
        if os.path.isfile(args.restore):
            print("=> loading checkpoint '{}'".format(args.restore))
            if args.gpu is None:
                checkpoint = torch.load(args.restore)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                ckpt = torch.load(args.restore, map_location=loc)['state_dict']
                # ckpt_q = torch.load(args.restore, map_location=loc)['state_dict']
                # from IPython import embed; embed(); exit()
                # ckpt_k = {'module.encoder_k.' + key[len('module.encoder_q'):]: val for key, val in ckpt_q.items() if key.startswith('module.encoder_q')}
                # ckpt = {**ckpt_q, **ckpt_k}

            # args.start_epoch = checkpoint['epoch']
            # best_loss = checkpoint['best_loss']
            model.load_state_dict(ckpt, strict=False)
            print("=> loaded checkpoint '{}'".format(args.restore))
        else:
            print("=> no checkpoint found at '{}'".format(args.restore))

    best_loss = 1e10

    # optionally resume from a checkpoint (in the current stage)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (loss {})"
                  .format(args.resume, best_loss))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # create dataloader
    train_loader, train_sampler = create_pfloader(args)

    print('start training')
    for epoch in range(args.start_epoch, args.nepoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        best_loss = train(args, train_loader, model, optimizer, closs, epoch, best_loss, ngpus_per_node)

        # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        #         and args.rank % ngpus_per_node == 0):
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': args.arch,
        #         'state_dict': model.state_dict(),
        #         'optimizer' : optimizer.state_dict(),
        #     }, is_best=False, filename=os.path.join(args.savedir, 'checkpoint_{:04d}.pth.tar'.format(epoch)), savedir=args.savedir)


def train(args, train_loader, model, optimizer, closs, epoch, best_loss, ngpus_per_node):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_i_cycle = AverageMeter('Loss_i_cycle', ':.4e')
    losses_center = AverageMeter('Loss_center', ':.4e')
    losses_moco = AverageMeter('Loss_moco', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, losses_i_cycle, losses_center, losses_moco, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (((im_q, im_k), _), data) in enumerate(train_loader):
        """
        im_q: a batch of query images [B x 3 x args.image_size x args.image_size]
        im_k: a batch of key images [B x 3 x args.image_size x args.image_size]
        crop.shape: [B x C x args.patch_size x args.patch_size]
        crop_gt.shape: [B x 2 x args.patch_size x args.patch_size]
        frames.shape: [B x T x C x args.full_size x args.full_size]
        """
        
        src_img = data['src_img']
        trg_img = data['trg_img']

        if args.gpu is not None:
            im_q = im_q.cuda(args.gpu, non_blocking=True)
            im_k = im_k.cuda(args.gpu, non_blocking=True)
            src_img = src_img.cuda(args.gpu, non_blocking=True)
            trg_img = trg_img.cuda(args.gpu, non_blocking=True)
            # v_frames = v_frames.cuda(args.gpu, non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        moco_output, image_cycle = model(im_q, im_k, src_img, trg_img, args.rotate, args.patch_size, args.dropout_rate)
        moco_logits, moco_labels = moco_output
        i_crop, i_crop_gt, i_att_map, i_At, i_coords10_cycle, i_coords10, i_coords20, i_aff_list, i_coords_list = image_cycle

        # computes losses
        i_cycle_label = i_crop_gt.view(i_crop_gt.size(0), -1, i_crop_gt.size(-1)) # shape (B*P)
        loss_i_cycle = F.mse_loss(i_coords10_cycle, i_cycle_label) * args.icycle_lw
        loss_moco = F.cross_entropy(moco_logits, moco_labels) * args.moco_lw
        loss_center = sum(closs(coord) for coord in i_coords_list) * args.lc
        loss = loss_i_cycle + loss_center + loss_moco

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(moco_logits, moco_labels, topk=(1, 5))
        top1.update(acc1[0], im_q.size(0))
        top5.update(acc5[0], im_q.size(0))

        losses.update(loss.item(), im_q.size(0))
        losses_i_cycle.update(loss_i_cycle.item(), im_q.size(0))
        losses_center.update(loss_center.item(), im_q.size(0))
        losses_moco.update(loss_moco.item(), im_q.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # display the training progress
        progress.display(i)

        # visualization
        if(i != 0 and i % args.log_interval == 0):
            # correspondence visualization
            save_vis_image(args.savepatch, epoch, i, 0, i_crop, src_img, i_coords10_cycle, i_coords10, i_cycle_label, i_att_map)
            save_vis_image(args.savepatch, epoch, i, 1, i_crop, trg_img, i_coords20)

        # save model
        if((i != 0 and i % args.save_interval == 0) or i + 1 == len(train_loader)):
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                is_best = losses.avg < best_loss
                best_loss = min(losses.avg, best_loss)
                checkpoint_path = os.path.join(
                    args.ckpt_dir,
                    'checkpoint_{:04d}_{:06d}.pth.tar'.format(epoch, i)
                )
                save_checkpoint({
                    'epoch': epoch,
                    'iteration': i,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_loss': best_loss,
                }, is_best, filename=checkpoint_path, savedir=args.ckpt_dir)

                log_current(epoch, i, best_loss, losses.avg,
                            losses_i_cycle.avg, losses_center.avg, top1, top5,
                            filename=os.path.join(args.savedir, "log_current.txt"))

    return best_loss

def create_pfloader(args):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    train_dataset_moco = torchvision.datasets.ImageFolder(
        os.path.join(args.image_dir),
        moco.loader.TwoCropsTransform(transforms.Compose(augmentation))
    )

    from data import dataset, download
    device = torch.device('cpu')
    train_dataset_pf = download.load_dataset(args.dataset, args.datapath, args.thres, device, args.split, args.cam)
    # dataloader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)

    train_dataset = ZipDataset([train_dataset_moco, train_dataset_pf])

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    return train_loader, train_sampler

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


def batch_random_crop(images, meshgrids, crop_size):
    """Randomly crops a batch of images.
    Args:
        images: [B, C, H, W]
        meshgrids: [B, 2, H, W]
        crop_size: [H0, W0]

    Returns:
        crop_images: [B, C, H0, W0]
        crop_grids: [B, 2, H0, W0]
    """
    B, C, H, W = images.size()
    H0, W0 = crop_size

    x1 = torch.randint(W - W0, (B, 1)).to(torch.float32).to(images.device)
    y1 = torch.randint(H - H0, (B, 1)).to(torch.float32).to(images.device)
    x2 = x1 + W0
    y2 = y1 + H0
    zero = torch.zeros_like(x1)

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


def random_rotate_crop_with_gt(first_frame, max_degree, crop_size):
    """Randomly rotate and crop the first frame and obtain the ground truth.

    Args:
        first_frame: [B, C, H, W], torch.Tensor, can be on gpu
        max_degree: the max degree of random rotate, a float number
        crop_size: [H0, W0]

    returns:
        crop: [B, C, args.patch_size, args.patch_size]
        crop_gt: [B, args.patch_size // 8, args.patch_size // 8, 2]
    """
    B, C, H, W = first_frame.size()
    downscale = 16

    meshgrid = torch.stack(torch.meshgrid(
        torch.arange(H, dtype=torch.float32).to(first_frame.device) / H,
        torch.arange(W, dtype=torch.float32).to(first_frame.device) / W), 0)

    # create meshgrids, shape: [B, 2, W, H]
    meshgrids = torch.stack([meshgrid] * B, 0)
    crop, crop_gt = batch_random_rotate(first_frame, meshgrids, max_degree)
    crop, crop_gt = batch_random_crop(crop, crop_gt, crop_size)
    crop_gt = F.interpolate(crop_gt, (crop_gt.size(2) // downscale, crop_gt.size(3) // downscale))
    scale = torch.tensor([H, W], dtype=torch.float32).view(1,-1,1,1).to(first_frame.device)

    crop_gt = (crop_gt * scale).permute(0,2,3,1)
    crop_gt = crop_gt[..., range(1,-1,-1)] # swap x and y
    crop = crop.detach()
    crop_gt = crop_gt.detach()

    # return crop.contiguous(), crop_gt.contiguous()
    return crop, crop_gt


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', savedir='models'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(savedir, 'model_best.pth.tar'))


def save_vis_image(savedir, epoch, iteration, idx, frame1, frame2, coord, coord1=None, coord_gt=None, att_map=None):
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

        crd = coord.data.cpu().numpy()[cnt]
        im_frame2_1 = draw_dots(im_frame2, crd)
        im_frame2_1 = cv2.resize(im_frame2_1, (im_frame1.shape[1],im_frame1.shape[0]))
        im = np.concatenate((im_frame1, im_frame2_1), axis=1)

        if coord1 is not None:
            crd1 = coord1.data.cpu().numpy()[cnt]
            im_frame2_2 = draw_dots(im_frame2, crd1, color=(0,255,0))
            im_frame2_2 = cv2.resize(im_frame2_2, (im_frame1.shape[1],im_frame1.shape[0]))
            im = np.concatenate((im, im_frame2_2), axis=1)

        if coord_gt is not None:
            crd_gt = coord_gt.data.cpu().numpy()[cnt]
            im_frame2_3 = draw_dots(im_frame2, crd_gt, color=(0,0,255))
            im_frame2_3 = cv2.resize(im_frame2_3, (im_frame1.shape[1],im_frame1.shape[0]))
            im = np.concatenate((im, im_frame2_3), axis=1)
        
        if att_map is not None:
            heatmap = np.uint8(att_map[cnt].data.cpu().numpy().transpose(1,2,0) * 255)
            heatmap = cv2.resize(heatmap, (im_frame2.shape[1], im_frame2.shape[0]))
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            im_frame_2_heat = np.array(0.5*im_frame2 + 0.3*heatmap, dtype=np.uint8)
            im_frame_2_heat = cv2.resize(im_frame_2_heat, (im_frame1.shape[1],im_frame1.shape[0]))
            im = np.concatenate((im, im_frame_2_heat), axis=1)

        if idx == 0:
            cv2.imwrite(os.path.join(savedir, "epoch_{:04d}_iter_{:06d}_{:02d}_i_loc_frame_gt.png".format(epoch, iteration, cnt)), im)
        else:
            cv2.imwrite(os.path.join(savedir, "epoch_{:04d}_iter_{:06d}_{:02d}_i_loc_frame_{:02d}.png".format(epoch, iteration, cnt, idx)), im)

def save_vis_video(savedir, epoch, iteration, idx, frame1, frame2, coord, coord1=None, coord_gt=None):
    b = frame1.size(0)
    frame1 = frame1 * 128 + 128
    frame2 = frame2 * 128 + 128

    for cnt in range(b):
        im1 = frame1[cnt].cpu().detach().numpy().transpose(1,2,0)
        im_frame1 = cv2.cvtColor(np.array(im1, dtype = np.uint8), cv2.COLOR_RGB2BGR)

        im2 = frame2[cnt].cpu().detach().numpy().transpose(1,2,0)
        im_frame2 = cv2.cvtColor(np.array(im2, dtype = np.uint8), cv2.COLOR_RGB2BGR)

        crd = coord.data.cpu().numpy()[cnt]
        im_frame2_1 = draw_dots(im_frame2, crd)
        im_frame2_1 = cv2.resize(im_frame2_1, (im_frame1.shape[1],im_frame1.shape[0]))
        im = np.concatenate((im_frame1, im_frame2_1), axis=1)

        if coord1 is not None:
            crd1 = coord1.data.cpu().numpy()[cnt]
            im_frame2_2 = draw_dots(im_frame2, crd1, color=(0,255,0))
            im_frame2_2 = cv2.resize(im_frame2_2, (im_frame1.shape[1],im_frame1.shape[0]))
            im = np.concatenate((im, im_frame2_2), axis=1)

        if coord_gt is not None:
            crd_gt = coord_gt.data.cpu().numpy()[cnt]
            im_frame2_3 = draw_dots(im_frame2, crd_gt, color=(0,0,255))
            im_frame2_3 = cv2.resize(im_frame2_3, (im_frame1.shape[1],im_frame1.shape[0]))
            im = np.concatenate((im, im_frame2_3), axis=1)

        if idx == 0:
            cv2.imwrite(os.path.join(savedir, "epoch_{:04d}_iter_{:06d}_{:02d}_v_loc_frame_gt.png".format(epoch, iteration, cnt)), im)
        else:
            cv2.imwrite(os.path.join(savedir, "epoch_{:04d}_iter_{:06d}_{:02d}_v_loc_frame_{:02d}.png".format(epoch, iteration, cnt, idx)), im)

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

def draw_bbox(img, bbox, c=(51,255,255)):
    """
    INPUTS:
    - segmentation, h * w * 3 numpy array
    - bbox: left, right, top, bottom
    OUTPUT:
    - image with a drawn bbox
    """
    # print("bbox: ", bbox)
    image = img.copy()
    pt1 = (int(bbox[0]),int(bbox[2]))
    pt2 = (int(bbox[1]),int(bbox[3]))
    # color = np.array([51,255,255], dtype=np.uint8)
    # c = tuple(map(int, color))
    image = cv2.rectangle(image, pt1, pt2, c, 5)
    return image

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.lr_mode == 'cos':  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.nepoch))
    elif args.lr_mode == 'step':  # stepwise lr schedule
        for milestone in args.lr_schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    elif args.lr_mode == 'poly':
        lr *= (1 - epoch / args.nepoch) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def log_current(epoch, iteration, best_loss, loss_ave, loss_i_cycle,
                loss_center, top1, top5, filename="log_current.txt"):
    with open(filename, "a") as text_file:
        print("time: {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")), file=text_file)
        print("epoch: {}".format(epoch), file=text_file)
        print("iteration: {}".format(iteration), file=text_file)
        print("best_loss: {}".format(best_loss), file=text_file)
        print("current_loss: {}".format(loss_ave), file=text_file)
        print("current_loss_i_cycle: {}".format(loss_i_cycle), file=text_file)
        print("current_loss_center: {}".format(loss_center), file=text_file)
        print("top1: {} ({})".format(top1.val, top1.avg), file=text_file)
        print("top5: {} ({})".format(top5.val, top5.avg), file=text_file)

if __name__ == '__main__':
    main()
