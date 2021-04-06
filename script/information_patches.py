"""
Description:
    This script is used to profile information of patches generated from kxk kernels on input images.
    See Section 5.2 in paper for more information

Author:
    Yue (Julien) Niu

Note:
"""
import argparse
import os
import sys
import time
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import pickle

sys.path.insert(0, './')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='imagenet', type=str, choices=['imagenet'])
parser.add_argument('--root-dir', default='../', type=str)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--save-dir', default='./results', type=str, help='statistic results dir')

def main():
    global args

    args = parser.parse_args()

    # construct dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_set = datasets.ImageFolder('/home/julien/dataset/cv/imagenet/train', transforms.Compose([
            transforms.RandomResizedCrop(112),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
                normalize,
    ]))

    """
    test_set = datasets.ImageFolder('/home/julien/dataset/cv/imagenet/val', transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ]))
    """

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True, drop_last = True,
        num_workers=4, pin_memory=True)
    """
    val_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size, shuffle=False, drop_last = True,
        num_workers=args.workers, pin_memory=True)
    """
    kern_sz = [ 3, 5, 7, 9, 11 ]
    data_profling( train_loader, kern_sz )

def data_profling( train_loader, kern_sz ):
    info = dict()
    for sz in kern_sz:
        info[ sz ] = []

    for i, (input, target) in enumerate(train_loader):

        input = input
        target = target

        for sz in kern_sz:
            info_batch_i = calc_information( input, sz )
            info[ sz ].append( info_batch_i.item() )

        print( "[INFO] processing batch {}(/{})".format( i, len( train_loader) ) )

        if i == 1000:
            break

    # save statistics
    file = open( args.save_dir+"/info_stat.pkl", "wb" )
    pickle.dump( info, file )
    file.close()

def calc_information( input, sz ):
    """calculate information of a batch given kernel sz
    @arg input: input batch ( batch_sz * channels * height * width )
    @arg sz: kernel size
    @return information of the batch
    """
    dim_in = input.shape
    n_tot = dim_in[ 0 ] * dim_in[ 1 ]
    sz_img = dim_in[ 2 ]

    assert sz_img > sz # kernel size must less than image size

    # generate patches
    pad = ( sz - 1 ) // 2
    sz_img_pad = sz_img + pad + pad
    kernel = sz_img_pad - sz + 1
    p2d = ( pad, pad, pad, pad )
    input_pad = torch.nn.functional.pad( input, p2d )
    input_3dim = input_pad.view( -1, sz_img_pad, sz_img_pad )
    input_4dim = torch.unsqueeze( input_3dim, 1 )
    unfold = torch.nn.Unfold( kernel_size = (sz, sz) )
    input_patches = unfold( input_4dim )

    # svd
    s = np.linalg.svd( input_patches, compute_uv = False )
    s = torch.from_numpy( s )
    assert len( s.shape ) == 2

    # calculate information
    s_norm = torch.nn.functional.normalize( s, p = 1, dim = 1 )
    info_batch = torch.sum( torch.mul( s_norm, s_norm ), dim = 1 )
    info_batch = -torch.log2( info_batch )

    return torch.mean( info_batch )

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

if __name__ == '__main__':
    main()
