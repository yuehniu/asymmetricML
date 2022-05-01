"""
Deep Gradient (dg) attack on AsymML.
The attack model can access gradients during training, and find a reconstructed image that
generated similar gradients as the original data.

Ref:
    attack model: https://arxiv.org/abs/1906.08935

Author:

Note:
"""
# -*- coding: utf-8 -*-
import argparse

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, datasets, transforms
from python.utils import label_to_onehot, cross_entropy_for_onehot
from python.utils import build_network
import python.datasetutils as datautils
from python.dataproc import data_lowrank, data_withnoise

print( torch.__version__, torchvision.__version__ )

parser = argparse.ArgumentParser( description='Deep Leakage from Gradients.' )
parser.add_argument(
    '--model', type=str, default='resnet18',
    help='target model to be attacked.'
)
parser.add_argument(
    '--dataset', type=str, default='cifar10',
    help='target dataset to be attacked.'
)
parser.add_argument(
    '--bs', default=64, type=int,
    help='batch size'
)
parser.add_argument(
    '--lr', default=0.1, type=float,
    help='initial learning rate'
)
parser.add_argument(
    '--index', type=int, default=25,
    help='the index for leaking images on CIFAR.'
)
parser.add_argument(
    '--image', type=str, default="",
    help='the path to customized image.'
)
parser.add_argument(
    '--pretrain', type=str, default='./checkpoints/resnet18model.pt',
    help='pretrained weights'
)
parser.add_argument(
    '--logdir', type=str, default='./log/dgattack/resnet18_pretrain_nsr_0.1',
    help='lod dir for storing attack results'
)
parser.add_argument( '--nsr', default=0.1, type=float, help='noise to signal ratio' )
args = parser.parse_args()

device = torch.device( "cuda:0" )
net = build_network(
    args.model, args.dataset, len_feature=512, num_classes=10
).to( device )
if args.pretrain:
    chkpoint = torch.load( args.pretrain )
    net.load_state_dict( chkpoint[ 'model_state_dict' ] )

if args.dataset == 'cifar10':
    dst = datautils.dataset_CIFAR10_train
    dstloader = torch.utils.data.DataLoader(
        dst, batch_size=args.bs, shuffle=True, num_workers=4
    )
elif args.dataset == 'cifar100':
    dst = datautils.dataset_CIFAR100_train
    dstloader = torch.utils.data.DataLoader(
        dst, batch_size=args.bs, shuffle=True, num_workers=4
    )
else:
    raise ValueError( 'Unsupported datasets!!!' )
torch.manual_seed( 5 )
# net.apply(weights_init)

criterion = torch.nn.CrossEntropyLoss()

writer = SummaryWriter( args.logdir )
for i, data in enumerate( dstloader, 0 ):
    images = data[ 0 ].to( device )
    labels = data[ 1 ].to( device )

    # compute original gradient
    preds = net( images )
    y = criterion( preds, labels )
    dy_dx = torch.autograd.grad( y, net.parameters() )

    original_dy_dx = list( ( _.detach().clone() for _ in dy_dx ) )

    # generate dummy data and label
    images_lowrank, _ = data_lowrank( images )
    images_residual = images - images_lowrank
    images_noisy = data_withnoise( images_residual, args.nsr )
    dummy_data = images_noisy.clone().requires_grad_( True )
    # dummy_data = torch.randn( gt_data.size() ).to( device ).requires_grad_( True )
    # dummy_label = torch.randn( gt_onehot_label.size() ).to( device ).requires_grad_( True )
    # plt.imshow( tt( dummy_data[ 0 ].cpu() ) )

    optimizer = torch.optim.LBFGS( [ dummy_data ], lr=args.lr )

    history = []
    for i in range( 1 ):
        def closure():
            optimizer.zero_grad()

            dummy_pred = net( dummy_data )
            # dummy_onehot_label = F.softmax( dummy_label, dim=-1 )
            dummy_loss = criterion( dummy_pred, labels )
            dummy_dy_dx = torch.autograd.grad( dummy_loss, net.parameters(), create_graph=True )

            grad_diff = 0
            for gx, gy in zip( dummy_dy_dx, original_dy_dx ):
                grad_diff += ( ( gx - gy ) ** 2 ).sum()
            grad_diff.backward()

            writer.add_scalar( 'Loss/loss',  grad_diff, i )

            return grad_diff


        optimizer.step( closure )

        writer.add_images( 'dgAttack/real', F.interpolate( images, size=(64, 64) ), i )
        writer.add_images( 'dgAttack/recon', F.interpolate( dummy_data, size=(64, 64) ), i )
        writer.add_images( 'dgAttack/residual', F.interpolate( images_noisy, size=(64, 64) ), i )

    writer.close()

    break
