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

sys.path.insert(0, './')
from python.utils import build_network, infer_memory_size, init_model
from python.dnn_transform import transform_model
from python.dnn_sgx import sgxDNN

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='vgg16', type=str, help='model name')
parser.add_argument('--device', default='gpu', type=str, choices=['gpu', 'cpu'], help='untrusted platform')
parser.add_argument('--sgx', action='store_true', help='whether or not use sgx')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--root-dir', default='./', type=str)

parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--wd', default=0.0005, type=float, help='weight decay')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--check-freq', default=20, type=int, help='checkpoint frequency')
parser.add_argument('--save-dir', default='./checkpoints', type=str, help='checkpoint dir')

parser.add_argument('--profile', action='store_true', help='whether or not profile performance')

def main():
    global args

    args = parser.parse_args()

    # build model
    model = build_network(args.model, num_classes=10)
    if args.device == 'gpu':
        model.cuda()
    else:
        model.cpu()

    # construct dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=args.root_dir+'/data/cifar10', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True, drop_last = True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=args.root_dir+'/data/cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False, drop_last = True,
        num_workers=args.workers, pin_memory=True)

    if args.sgx:
        # initilize SGX execution Obj
        sgxdnn = sgxDNN(use_sgx=False, n_enclaves=1, batchsize=args.batch_size)

        # transform model
        model, need_sgx = transform_model(model, sgxdnn)
        if args.device == 'gpu':
            model.cuda()
        else:
            model.cpu()

        # Construct SGX execution Context
        infer_memory_size(sgxdnn, model, need_sgx, args.batch_size, [3, 32, 32])
        sgxdnn.sgx_context(model, need_sgx)

        init_model(model)
    else:
        sgxdnn = None

    # define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    if args.device == 'gpu':
        criterion.cuda()
    else:
        criterion.cpu()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.wd)

    # start training
    for epoch in range(0, args.epochs):
        prec1_train, loss_train = train(model, train_loader, criterion, optimizer, sgxdnn, epoch)

        prec1_val, loss_val = validate(model, val_loader, criterion, sgxdnn, epoch)

        # save checkpoint
        #if epoch % args.check_freq == 0:
        #    state = {
        #        'epoch': epoch + 1,
        #        'state_dict': model.state_dict(),
        #        'prec1': prec1_val,
        #    }
        #    file_name = os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch))
        #    torch.save(state, file_name)

def train(model, train_loader, criterion, optimizer, sgxdnn, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # record data loading time
        data_time.update(time.time() - end)

        if args.device == 'gpu':
            input = input.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        if args.sgx:
            sgxdnn.reset()

        # forward and backward
        output = model(input)
        loss = criterion(output, target)
        loss.backward()

        # update parameter
        optimizer.step()

        # report acc and loss
        prec1 = cal_acc(output, target)[0]

        avg_acc.update(prec1.item(), input.size(0))
        avg_loss.update(loss.item(), input.size(0))

        # record elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.check_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {avg_loss.avg:.4f}\t'
                  'Prec@1 {avg_acc.avg:.3f}'.format(
                      epoch, i, len(train_loader), batch_time = batch_time,
                      data_time = data_time, avg_loss = avg_loss, avg_acc = avg_acc))

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

        # forward
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
