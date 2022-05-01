"""
Project decription:
    In this project, we split the model (conv layers) into two parts: part 1 is executed in trusted platform (SGX);
    part 2 is in untrusted platform, like GPU or normal CPU. After a conv layer, results from both platforms need to be
    collected in the SGX, and then apply non-linear operation. After non-linear operation, output activations need to be
    re-split into two parts and fed into trusted and untrusted platforms.

Implementation highlight:
    During implementation, the following operations need to be customized:
    - conv layer: conv layer is the central op which needs to be changed. The customized conv layer is able to distribute
                  computation into trusted/untrusted platforms. In addition, the backward need to be rewritten;
    - non-linear layer: non-linear op needs to rewritten and executed in the trusted platform. Similar to conv layer, the
                  backward also need to be rewritten;
    - pooling layer: similar to non-linear layer, both forward and backward need to be re-written.

    The following new operations are added into non-linear or pooling layer:
    - SVD: SVD op is included in non-liear/pooling layer to re-split activations.

Author:


Note:
    Test FTPSync***
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, './')
from python.utils import build_network, infer_memory_size, init_model, est_MI
from python.dnn_transform import transform_model_sgx, transform_model_lowrank, get_internal_rank
from python.dnn_sgx import sgxDNN
from python.datasetutils import *
from python.dataproc import data_lowrank, data_withnoise
from python.mi import get_activations

parser = argparse.ArgumentParser()
parser.add_argument( '--model', default='resnet18', type=str, help='model name' )
parser.add_argument( '--device', default='gpu', type=str, choices=['gpu', 'cpu'], help='untrusted platform' )
parser.add_argument( '--sgx', action='store_true', help='whether or not use sgx' )
parser.add_argument( '--dataset', default='cifar10', type=str, choices=['cifar10', 'imagenet'] )
parser.add_argument( '--root-dir', default='./', type=str )

# Train parameters
parser.add_argument( '--batch-size', default=128, type=int )
parser.add_argument( '--lr', default=0.1, type=float )
parser.add_argument( '--scheduler', default='cos', choices=[ 'step', 'cos' ], help='lr scheduler' )
parser.add_argument( '--lr-decay', default=0.1, type=float, help='lr decay factor' )
parser.add_argument( '--decay-period', default=75, type=int, help='lr decay period' )
parser.add_argument( '--momentum', default=0.9, type=float )
parser.add_argument( '--wd', default=0.0005, type=float, help='weight decay' )
parser.add_argument( '--epochs', default=200, type=int )
parser.add_argument( '--workers', default=8, type=int, help='number of data loading workers' )
parser.add_argument( '--check-freq', default=100, type=int, help='checkpoint frequency' )
parser.add_argument( '--save-dir', default='./checkpoints/', type=str, help='checkpoint dir' )
parser.add_argument( '--log-dir', default='./log/cifar10/tmp', type=str, help='train log dir' )

# Adding noise
parser.add_argument( '--noisyinput', action='store_true', help='whether add noise to inputs' )
parser.add_argument( '--nsr', default=0.1, type=float, help='noise-to-signal ratio' )
parser.add_argument( '--estmi', action='store_true', help='whether to estimate mutual information' )

# Low-rank data
parser.add_argument( '--lowrank', action='store_true', help='whether use low-rank inputs' )
parser.add_argument( '--rank', default=1, type=int, help='rank of inputs for first layer' )
parser.add_argument( '--rankdist', action='store_true', help='record distributed of rank' )

# Attacker training
parser.add_argument( '--MSattack', action='store_true', help='membership attack training' )
parser.add_argument( '--MIattack', action='store_true', help='model inversion attach training' )

parser.add_argument( '--profiling', action='store_true', help='whether or not profile performance' )

args = parser.parse_args()


def main():
    # - build model
    if args.dataset == 'cifar10':
        sz_img, len_feature = 32, 512  # default for VGG16
        model = build_network(
            args.model, args.dataset, len_feature=len_feature, num_classes=10
        )
        train_set, test_set = dataset_CIFAR10_train, dataset_CIFAR10_test
    elif args.dataset == 'imagenet':
        sz_img, len_feature = 224, 4608  # default for VGG16
        if 'resnet' in args.model:
            len_feature = 512
        model = build_network(
            args.model, args.dataset, len_feature=len_feature, num_classes=1000
        )
        train_set, test_set = dataset_IMAGENET_train, dataset_IMAGENET_test

    # - construct dataset
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True )
    val_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.workers, pin_memory=True )

    # =========================================================================
    # format model to one of the following format
    # - Model with SGX
    # - Model with low-rank data (run on GPU only)
    sgxdnn = None
    if args.sgx:
        # - initialize SGX execution Obj
        sgxdnn = sgxDNN( use_sgx=True, n_enclaves=1, batchsize=args.batch_size )
        # - transform model to using customized modules in SGX
        model, need_sgx = transform_model_sgx( model, sgxdnn )
        # - init model parameters
        init_model(model)
        # - construct SGX execution Context
        infer_memory_size( sgxdnn, model, need_sgx, args.batch_size, [3, sz_img, sz_img] )
        sgxdnn.sgx_context( model, need_sgx )
    elif args.lowrank:
        model = transform_model_lowrank( model, args.model )

    if args.device == 'gpu':
        model.cuda()
        model = torch.nn.DataParallel( model )
    else:
        model.cpu()
    print( "====================================Model architecture====================================")
    print( model )
    print("===========================================================================================")

    # - define loss, optimizer, and lr scheduler
    criterion = torch.nn.CrossEntropyLoss()
    if args.device == 'gpu':
        criterion.cuda()
    else:
        criterion.cpu()
    optimizer = torch.optim.SGD( model.parameters(), args.lr,
                                 momentum=args.momentum, weight_decay=args.wd )
    if args.scheduler == 'step':
        lr_lambda = lambda epoch: args.lr_decay ** ( epoch // args.decay_period )
        scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=lr_lambda )
    elif args.scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, args.epochs, eta_min=0.001 )

    # - start training
    writer = SummaryWriter( log_dir=args.log_dir )
    best_prec1_val = 0.0
    for epoch in range( 0, args.epochs ):
        ranks = {}
        get_internal_rank(model, ranks, True)

        prec1_train, loss_train = train( model, train_loader, criterion, optimizer, sgxdnn, epoch, ranks )

        if args.sgx and args.profiling:
            print_perf( sgxdnn.time_fwd, sgxdnn.time_bwd, sgxdnn.time_fwd_sgx, sgxdnn.time_bwd_sgx )
            break

        prec1_val, loss_val = validate(model, val_loader, criterion, sgxdnn, epoch)
        if best_prec1_val < prec1_val.avg:
            best_prec1_val = prec1_val.avg

        # - save checkpoint
        # add here

        # - save training and val metric
        writer.add_scalar( 'Loss/train', loss_train.avg, epoch )
        writer.add_scalar( 'Accuracy/train', prec1_train.avg, epoch )
        writer.add_scalar( 'Loss/val', loss_val.avg, epoch )
        writer.add_scalar( 'Accuracy/val', prec1_val.avg, epoch )
        if args.rankdist:
            for k in ranks.keys():
                writer.add_histogram( 'Rank/'+k, np.array( ranks[ k ] ), epoch )

        scheduler.step()

    # - save models
    print( 'The best validation accuracy: ', best_prec1_val )
    torch.save( 
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        args.save_dir+'model.pt'
    )

    # - get intermediate activations and estimate mutual information
    if args.estmi:
        outputs_inside = []
        get_activations( model, outputs_inside )

        # compute mutual information (MI) between inputs and intermediate features
        for i, ( input, target ) in enumerate( train_loader ):
            outputs_inside = []
            if args.device == 'gpu':
                input = input.cuda()
                target = target.cuda()
            noise = torch.zeros_like( input )
            if args.noisyinput:
                noise_mean = 0.0
                noise_std = args.nsr * torch.ones_like( input )
                noise = torch.normal( noise_mean, noise_std )
                input += noise
            optimizer.zero_grad()
            output = model( input )
            loss = criterion(output, target)
            loss.backward()
            mi = est_MI( input - noise, noise, outputs_inside )
            print( 'mutual info: {}'.format( mi ) )
            break

    writer.close()


def train( model, train_loader, criterion, optimizer, sgxdnn, epoch, ranks ):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()

    model.train()

    end = time.time()
    ranks_actual = { 'input': [] }
    for i, ( input, target ) in enumerate( train_loader ):
        # - record data loading time
        data_time.update( time.time() - end )

        # =====================================================================
        # Data pre-processing
        # - if move to gpus
        # - if add noise to input data
        # - if use low-rank parts of input data
        # - if use non low-rank parts for membership attack training

        if args.device == 'gpu':
            input = input.cuda()
            target = target.cuda()
        if args.noisyinput:
            input = data_withnoise( input, args.nsr )
        if args.lowrank:
            input, rank_actual = data_lowrank( input, args.rank )
            if args.rankdist:
                ranks_actual[ 'input' ].append( rank_actual )
        if args.MSattack:
            input_lowrank, rank_actual = data_lowrank( input, args.rank )
            input -= input_lowrank
            # input = data_withnoise( input, args.nsr )

        # - show input in tensorboard
        # if epoch == 0:
        #     grid = torchvision.utils.make_grid( input )
        #     writer.add_image( 'input', grid, 0 )

        optimizer.zero_grad()
        if args.sgx:
            sgxdnn.reset()

        # - forward and backward
        output = model( input )
        loss = criterion( output, target )
        loss.backward()

        # - update parameter
        optimizer.step()

        if args.rankdist:
            get_internal_rank( model, ranks, False )

        # - report acc and loss
        prec1 = cal_acc( output, target )[ 0 ]

        avg_acc.update( prec1.item(), input.size( 0 ) )
        avg_loss.update( loss.item(), input.size( 0 ) )

        # - record elapsed time
        batch_time.update( time.time() - end )
        end = time.time()

        if i % args.check_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {avg_loss.avg:.4f}\t'
                  'Prec@1 {avg_acc.avg:.3f}'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, avg_loss=avg_loss, avg_acc=avg_acc ) )
        if i == 100 and args.sgx:
            print_perf( sgxdnn.time_fwd, sgxdnn.time_bwd, sgxdnn.time_fwd_sgx, sgxdnn.time_bwd_sgx )
            print_power( sgxdnn.power_fwd, sgxdnn.power_fwd_sgx )
            quit()

    return avg_acc, avg_loss


def validate(model, val_loader, criterion, sgxdnn, epoch):
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()

    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.sgx:
            sgxdnn.reset()

        if args.device == 'gpu':
            input = input.cuda()
            target = target.cuda()

        # - forward
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        prec1 = cal_acc(output, target)[0]
        avg_loss.update(loss.item(), input.size(0))
        avg_acc.update(prec1.item(), input.size(0))

    print('Epoch: [{}]\tLoss {:.4f}\tPrec@1 {:.3f}'.format(epoch, avg_loss.avg, avg_acc.avg))

    return avg_acc, avg_loss


def cal_acc(output, target, topk=(1,)):
    """
    Calculate model accuracy
    :param output:
    :param target:
    :param topk:
    :return: topk accuracy
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    acc = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        acc.append(correct_k.mul_(100.0 / batch_size))
    return acc


def print_perf( time_fwd, time_bwd, time_fwd_sgx, time_bwd_sgx ):
    """Print run time of fwd/bwd in Python and SGX
    :param time_fwd/time_bwd: total run time of fwd/bwd on each layer
    :param time_fwd/sgx/time_bwd_sgd: run time of fwd/bwd in sgx
    """
    for lyr in time_fwd:
        print('Layer {}\t: forward time: {:.4f} (sgx: {:.4f}),\t backward time: {:.4f} (sgx: {:.4f})'.format(
            lyr, time_fwd[lyr].avg, time_fwd_sgx[lyr].avg, time_bwd[lyr].avg, time_bwd_sgx[lyr].avg))


def print_power(power_fwd, power_fwd_sgx):
    power_tot = 0.0
    power_gpu = 0.0
    for lyr in power_fwd:
        power_gpu += power_fwd[ lyr ].avg
        power_tot += power_fwd_sgx[ lyr ].avg
    print( 'Total power vs gpu power: {},\t{}'.format( power_tot, power_gpu ) )
    print( 'Power ratio: {}'.format( power_gpu/power_tot ) )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__( self ):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
