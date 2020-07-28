# coding: utf-8
import os
import time
import random
import logging
import argparse
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import *
from dataset import *
from cvtransforms import *


SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


parser = argparse.ArgumentParser(description='Pytorch-GLMIM-LRW')
parser.add_argument('--nClasses', default=500, type=int, help='the number of classes')
#path : path to model, when you start Baseline, it's empty, otherwise it would load the model.
parser.add_argument('--path', default=r'', type=str, help='path to Baseline, empty for training')
parser.add_argument('--dataset', default=r'', type=str, help='path/to/lrw/roi_80_116_175_211_npy_gray/')
parser.add_argument('--mode', default='Baseline', type=str)
parser.add_argument('--every-frame', default=True, action='store_false', help='predicition based on every frame')
parser.add_argument('--lr', default=1e-5, type=float, help='initial learning rate')
parser.add_argument('--batch-size', default=70, type=int, help='mini-batch size')
parser.add_argument('--workers', default=5, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=500, type=int, help='number of total epochs')
parser.add_argument('--s_epochs', default=1, type=int, help='number of start epochs')
parser.add_argument('--interval', default=40, type=int, help='display interval')
parser.add_argument('--test', default=False, action='store_false', help='perform on the test phase')
args = parser.parse_args()
print(args)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def data_loader(args):
    dsets = {x: MyDataset(x, args.dataset) for x in ['train','val', 'test']}
    dset_loaders = {x: DataLoader(dsets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.workers) for x in ['train', 'val', 'test']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
    print('\nStatistics: train: {}, val: {}, test: {}'.format(dset_sizes['train'], dset_sizes['val'], dset_sizes['test']))
    return dset_loaders, dset_sizes

def reload_model(model, logger, path=""):
    if not bool(path):
        logger.info('train from scratch')
        return
    own_state = model.state_dict()
    state_dict = torch.load(path)
    for name, param in state_dict.items():
        if name not in own_state:
            print('layer {} skip, not exist'.format(name))
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        if own_state[name].shape != param.shape:
            print('layer {} skip, shape not same'.format(name))
            continue
        own_state[name].copy_(param)

def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

def train(model, dset_loaders, criterion, epoch, phase, optimizer, args, logger, use_gpu):

    model.train()
    logger.info('-' * 10)
    logger.info('Epoch {}/{}'.format(epoch, args.epochs - 1))
    logger.info('Current Learning rate: {}'.format(showLR(optimizer)))

    running_loss, running_corrects, running_all = 0., 0., 0.
    since = time.time()
    last_time_batch_idx = -1
    for batch_idx, (inputs, targets) in enumerate(dset_loaders[phase]):
        batch_img = RandomCrop(inputs.numpy(), (88, 88))
        batch_img = ColorNormalize(batch_img)
        batch_img = HorizontalFlip(batch_img)

        batch_img = np.reshape(batch_img, (batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], batch_img.shape[3], 1))
        inputs = torch.from_numpy(batch_img)
        inputs = inputs.float().permute(0, 4, 1, 2, 3).contiguous()
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()
        outputs = model(inputs)

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # stastics
        running_loss += loss.item() * inputs.size(0)
        batch_correct = (preds == targets.data).sum().item()
        running_corrects += batch_correct
        running_all += len(inputs)

        if batch_idx % args.interval == 0 or (batch_idx == len(dset_loaders[phase])-1):
            print('Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss batch: {:.4f}\tLoss total: {:.4f}\tAcc batch:{:.4f}\tAcc total:{:.4f}\tEstimated time:{:5.0f}s\r'.format(
                running_all,
                len(dset_loaders[phase].dataset),
                100. * batch_idx / (len(dset_loaders[phase])-1),
                float(loss),
                float(running_loss) / running_all,
                float(batch_correct) / len(inputs),
                float(running_corrects) / running_all,
                (time.time() - since) / (batch_idx - last_time_batch_idx) * (len(dset_loaders[phase]) - batch_idx - 1))),
            last_time_batch_idx = batch_idx
            since = time.time()

    loss_epoch =  float(running_loss) / len(dset_loaders[phase].dataset)
    acc_epoch = float(running_corrects) / len(dset_loaders[phase].dataset)

    logger.info('{} Epoch:\t{:2}\tLoss: {:.4f}\tAcc:{:.4f}\n'.format(
        phase,
        epoch,
        loss_epoch,
        acc_epoch))

def test(model, dset_loaders, criterion, epoch, phase, args, logger, use_gpu, save=True):

    model.eval()
    with torch.no_grad():

        running_loss, running_corrects, running_all = 0., 0., 0.
        since = time.time()
        last_time_batch_idx = -1
        for batch_idx, (inputs, targets) in enumerate(dset_loaders[phase]):

            batch_img = CenterCrop(inputs.numpy(), (88, 88))
            batch_img = ColorNormalize(batch_img)

            batch_img = np.reshape(batch_img, (batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], batch_img.shape[3], 1))
            inputs = torch.from_numpy(batch_img)
            inputs = inputs.float().permute(0, 4, 1, 2, 3).contiguous()
            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, targets)
            # stastics
            running_loss += loss.data * inputs.size(0)
            running_corrects += (preds == targets.data).sum().item()
            running_all += len(inputs)

            if batch_idx % args.interval == 0 or (batch_idx == len(dset_loaders[phase])-1):
                print('Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss: {:.4f}\tAcc:{:.4f}\tEstimated time:{:5.0f}s\r'.format(
                    running_all,
                    len(dset_loaders[phase].dataset),
                    100. * batch_idx / (len(dset_loaders[phase])-1),
                    float(running_loss) / running_all,
                    float(running_corrects) / running_all,
                    (time.time() - since)/(batch_idx-last_time_batch_idx) * (len(dset_loaders[phase]) - batch_idx - 1))),
                last_time_batch_idx = batch_idx
                since = time.time()

        loss_epoch = float(running_loss) / len(dset_loaders[phase].dataset)
        acc_epoch = float(running_corrects) / len(dset_loaders[phase].dataset)
        logger.info('{} Epoch:\t{:2}\tLoss: {:.4f}\tAcc:{:.4f}'.format(
            phase,
            epoch,
            loss_epoch,
            acc_epoch)+'\n')
    if save:
        return acc_epoch

def test_adam(args, use_gpu):
    save_path = './' + args.mode
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    # logging info
    filename = save_path+'/'+args.mode+'_'+str(args.lr)+'.txt'
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename, mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    net = Lipreading(mode=args.mode, inputDim=512, hiddenDim=1024, nClasses=args.nClasses, frameLen=29, every_frame=args.every_frame)
    print(net)
    # reload model
    reload_model(net, logger, args.path)
    # define loss function and optimizer
    model = torch.nn.DataParallel(net)
    if use_gpu:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-5, amsgrad=True)

    dset_loaders, dset_sizes = data_loader(args)
    if args.test:
        test(model, dset_loaders, criterion, 0, 'test',  args, logger, use_gpu,  False)
        return

    best_acc = -1
    for epoch in range(args.s_epochs, args.epochs):
        train(model, dset_loaders, criterion, epoch, 'train', optimizer, args, logger, use_gpu)
        test_acc = test(model, dset_loaders, criterion, 0, 'val', args, logger, use_gpu, True)
        if test_acc > best_acc:
            best_acc = max(test_acc, best_acc)
            state_dict = net.state_dict()
            torch.save(state_dict, '{}/epoch{}_acc{}.pt'.format(save_path,str(epoch),str(best_acc)))

if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    test_adam(args, use_gpu)
