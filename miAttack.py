"""
Model Inversion (mi) attack on AsymML.
The attack model can access 1) well-trained model parameters; 2) data in GPU memory,
and try to reconstruct images that are similar to training dataset.

Ref:
    attack model: http://arxiv.org/abs/1911.07135

Author:

Note:
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from skimage import metrics

sys.path.insert(0, './')
from python.models.attacker import miGenerator, miDiscriminator
from python.utils import init_netGD, build_network
import python.datasetutils as datautils
from python.dataproc import data_lowrank, data_withnoise

parser = argparse.ArgumentParser()
parser.add_argument( '--dataset', default='cifar10', type=str, choices=['cifar10', 'imagenet'] )

# target model under attack
parser.add_argument( '--nz', default=100, type=int, help='size of latent vector' )
parser.add_argument( '--model', default='resnet18', type=str, help='target model' )
parser.add_argument( '--modeldir', default='./checkpoints/resnet18model.pt', type=str, help='model path' )

# train parameters
parser.add_argument( '--batch-size', default=64, type=int )
parser.add_argument( '--lr', default=0.0005, type=float )
parser.add_argument( '--lr-decay', default=0.1, type=float, help='lr decay factor' )
parser.add_argument( '--decay-period', default=100, type=int, help='lr decay period' )
parser.add_argument( '--momentum', default=0.9, type=float )
parser.add_argument( '--wd', default=0.0005, type=float, help='weight decay' )
parser.add_argument( '--epochs', default=1, type=int )
parser.add_argument( '--workers', default=8, type=int, help='number of data loading workers' )
parser.add_argument( '--check-freq', default=20, type=int, help='checkpoint frequency' )
parser.add_argument( '--ckpdir', default='./checkpoints', type=str, help='checkpoint dir' )
parser.add_argument( '--log-dir', default='./log/attack/resnet18_nsr_0.1', type=str, help='train log dir' )

# DP noise parameters
parser.add_argument( '--nsr', default=0.1, type=float, help='noise to signal ratio' )
args = parser.parse_args()


def main():
    device = torch.device( "cuda:0" )
    # =========================================================================
    # create attack model with 1) generator; 2) discriminator
    # model architecture is based on The Secret Revealer:
    # (http://arxiv.org/abs/1911.07135)

    # - generator
    netG = miGenerator( 32, args.nz, 128 ).to( device )
    nn.DataParallel( netG )
    netG.apply( init_netGD )
    print('=========================Generator=========================')
    print( netG )

    # - discriminator
    netD = miDiscriminator( 64 ).to( device )
    nn.DataParallel( netD )
    netD.apply( init_netGD )
    print( '=========================Discriminator=========================' )
    print( netD )

    # - define loss and optimization
    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam( netG.parameters(), lr=args.lr/4, betas=(0.5, 0.999) )
    optimizerD = torch.optim.Adam( netD.parameters(), lr=args.lr, betas=(0.5, 0.999) )
    lr_lambda = lambda epoch: args.lr_decay ** (epoch // args.decay_period)
    schedulerG = torch.optim.lr_scheduler.LambdaLR( optimizerG, lr_lambda=lr_lambda)
    schedulerD = torch.optim.lr_scheduler.LambdaLR( optimizerD, lr_lambda=lr_lambda )

    # - create dataset
    if args.dataset == 'cifar10':
        trainset = datautils.dataset_CIFAR10_train
        trainloader = torch.utils.data.DataLoader( trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers )
        valset = datautils.dataset_CIFAR10_test
        valloader = torch.utils.data.DataLoader( valset,
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True )

    # =========================================================================
    # start training...
    # update netD and netG altenatively
    writer = SummaryWriter( args.log_dir )
    fix_noise = torch.randn( args.batch_size, args.nz, 1, 1, device=device )
    for epoch in range( args.epochs ):
        train( netD, netG, trainloader, criterion, optimizerD, optimizerG, device, epoch, writer )

        # - check generator's capability
        if epoch % 2 == 0:
            with torch.no_grad():
                for i, data in enumerate( valloader, 0 ):
                    images = data[ 0 ].to( device )
                    images_lowrank, _ = data_lowrank( images )
                    images_residual = images - images_lowrank
                    images_noise = data_withnoise( images_residual, args.nsr )
                    fake = netG( images_noise, fix_noise ).detach().cpu()
                    break
            writer.add_images( 'train/real', images, epoch )
            writer.add_images( 'train/recon', fake, epoch )

    # =========================================================================
    # perform model inversion ( with data in untrusted GPUs as prior information )
    # 1) load the target model
    if args.dataset == 'cifar10':
        sz_img, len_feature = 32, 512
        model = build_network( args.model, args.dataset, len_feature=len_feature, num_classes=10  ).cuda()
        chkpoint = torch.load( args.modeldir )
        model.load_state_dict( chkpoint[ 'model_state_dict' ] )
        model.eval()

    # 2) load data and start model inversion optimization
    # detailed optimization procedure can be found in
    # 
    n_tries = 10
    criterion = torch.nn.CrossEntropyLoss()
    gloss = AverageMeter()
    gacc = AverageMeter()
    g_psnr, g_ssim = AverageMeter(), AverageMeter()
    for i, data in enumerate( valloader, 0 ):
        images = data[ 0 ].to( device )
        labels = data[ 1 ].to( device )
        images_lowrank, _ = data_lowrank( images )
        images_residual = images - images_lowrank
        images_noisy = data_withnoise( images_residual, args.nsr )
        loss_min = float( 'inf' )
        acc_best = 0.0
        for j in range( n_tries ):
            noise_j = torch.randn( args.batch_size, args.nz, 1, 1, device=device )
            fake = netG( images_noisy, noise_j )
            fake_normalieze = torchvision.transforms.functional.normalize(
                fake,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            output = model( fake_normalieze )
            loss = criterion( output, labels ).item()
            prec1 = cal_acc( output, labels )[ 0 ].item()
            if loss < loss_min:
                loss_min = loss
                acc_best = prec1
                images_recon = fake

        # - PSNR and SSIM
        psnr_i, ssim_i = cal_metrics( images, images_recon )
        g_psnr.update( psnr_i, args.batch_size )
        g_ssim.update( ssim_i, args.batch_size )

        gloss.update( loss_min, args.batch_size )
        gacc.update( acc_best, args.batch_size )
        writer.add_images( 'miAttack/real', F.interpolate( images, size=(64, 64) ), i )
        writer.add_images( 'miAttack/recon', F.interpolate( images_recon, size=(64, 64) ), i )
        writer.add_images( 'miAttack/residual', F.interpolate( images_noisy, size=(64, 64) ), i )
    print( 'generator loss: {:.4f}\t generator acc: {:.4f}'.format( gloss.avg, gacc.avg ) )
    print( 'generator PSNR: {:.4f}\t generator SSIM: {:.4f}'.format( g_psnr.avg, g_ssim.avg ) )

    writer.close()


def train( netD, netG,  trainloader, criterion, optimizerD, optimizerG, device, epoch, writer ):
    """
    train wrapper for generator and discriminator
    :param netD discriminator model
    :param netG generator model
    :param trainloader train data loader
    :param criterion loss function
    :param optimizerD optimizer for discriminator
    :param optimizerG optimizer for generator
    :param device runtime device (GPUs)
    :param epoch current epoch
    :param writer summary writer for tensorboard
    """
    avglossD = AverageMeter()
    avglossG = AverageMeter()
    for i, data in enumerate( trainloader, 0 ):
        # =========================================================================
        # train discriminator: netD

        # - first feed real data
        netD.zero_grad()
        images = data[ 0 ].to( device )
        b = images.size( 0 )
        labels = torch.full( (b,), 1.0, dtype=torch.float, device=device )
        output = netD( images ).view( -1 )
        lossD_real = criterion( output, labels )
        lossD_real.backward()

        # - then feed fake data and prior information
        noise = torch.randn( b, args.nz, 1, 1, device=device )
        images_lowrank, _ = data_lowrank( images )
        images_residual = images - images_lowrank
        images_noisy = data_withnoise( images_residual, args.nsr )
        images_fake = netG( images_noisy, noise )
        labels.fill_( 0.0 )
        output = netD( images_fake.detach() ).view( -1 )
        lossD_fake = criterion( output, labels )
        lossD_fake.backward()

        lossD = lossD_real + lossD_fake
        optimizerD.step()
        avglossD.update( lossD.item(), b )

        # =========================================================================
        # train generator: netG
        netG.zero_grad()
        labels.fill_( 1.0 )
        output = netD( images_fake ).view( -1 )
        lossG = criterion( output, labels )
        lossG.backward()

        optimizerG.step()
        avglossG.update( lossG.item(), b )

        # =========================================================================
        # print train progress
        if i % 100 == 0:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Loss D {avglossD.avg:.4f}\t Loss G {avglossG.avg:.4f}'.format(
                    epoch, args.epochs, i, len(trainloader), avglossD=avglossD, avglossG=avglossG ) )
            writer.add_scalar( 'loss/netD', avglossD.avg, global_step=epoch*len(trainloader)+i )
            writer.add_scalar( 'loss/netG', avglossG.avg, global_step=epoch*len(trainloader)+i )


def cal_acc(output, target, topk=(1,)):
    """
    Calculate model accuracy
    :param output:
    :param target:
    :param topk:
    :return: topk accuracy
    """
    maxk = max(topk)
    batch_size = target.size( 0 )

    _, pred = output.topk( maxk, 1, True, True )
    pred = pred.t()
    correct = pred.eq( target.view( 1, -1 ).expand_as( pred ) )

    acc = []
    for k in topk:
        correct_k = correct[:k].view( -1 ).float().sum( 0 )
        acc.append( correct_k.mul_( 100.0 / batch_size ) )
    return acc


def cal_metrics( orig, recon ):
    """Calculate metrics such as PSNR and SSIM
    :param orig original data
    :param recon reconstructed data from the attack model
    """
    n_batch = orig.size( 0 )
    orig = orig.cpu().detach().numpy()
    orig = np.transpose( orig, ( 0,2,3,1 ) )
    recon = recon.cpu().detach().numpy()
    recon = np.transpose( recon, ( 0,2,3,1 ) )

    val_psnr, val_ssim = 0.0, 0.0
    for b in range( n_batch ):
        orig_i = orig[ b ]
        recon_i = recon[ b ]
        val_psnr += metrics.peak_signal_noise_ratio( orig_i, recon_i )
        val_ssim += metrics.structural_similarity( orig_i, recon_i, multichannel=True )

    return val_psnr/n_batch, val_ssim/n_batch


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
