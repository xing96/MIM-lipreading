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


parser = argparse.ArgumentParser(description='Pytorch-GLMIM-LRW1000')
parser.add_argument('--nClasses', default=1000, type=int, help='the number of classes')
#path : path to model, when you start Baseline + local + Global, the path is the Baseline of Baseline + Local
parser.add_argument('--path', default=r'', type=str, help='path to model')
#path2 : path to model Global, when you start Baseline + local + Global, it is empty. Here we fix the Fron-tend because the GMIM has only limited effect on LRW-1000 (refer to paper).
parser.add_argument('--path2', default=r'', type=str, help='path to Global')
parser.add_argument('--dataset', default=r'', type=str, help='path to LRW1000/images/images')
parser.add_argument('--trn', default=r'', type=str, help='path/to/trn_1000.txt')
parser.add_argument('--val', default=r'', type=str, help='path/to/val_1000.txt')
parser.add_argument('--tst', default=r'', type=str, help='path/to/tst_1000.txt')
parser.add_argument('--pad', default= 30, type=int, help='pad')
parser.add_argument('--mode', default='Baseline_Local_Global', type=str)
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--batch-size', default= 130, type=int, help='mini-batch size')
parser.add_argument('--workers', default=1, type=int, help='number of data loading workers, set 0 if windows')
parser.add_argument('--epochs', default=500, type=int, help='number of total epochs')
parser.add_argument('--s_epochs', default=1, type=int, help='number of start epochs')
parser.add_argument('--interval', default=30, type=int, help='display interval')
parser.add_argument('--test', default=True, action='store_false', help='perform on the test phase')
args = parser.parse_args()
print(args)
best_acc = -1
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def make_one_hot_global(label,length):
    batch = label.size(0)
    a = torch.zeros(batch, length).scatter_(1, label.view(-1, 1), 1)
    return a
def data_loader(args):
    dsets = {'train': LipreadingDataset(args.dataset, args.trn, args.pad), 'val': LipreadingDataset(args.dataset, args.val, args.pad, False), 'test': LipreadingDataset(args.dataset, args.tst, args.pad, False)}
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

def train(model, model2, dset_loaders, criterion, BCEcriterion, epoch, phase, optimizer ,optimizer_Global,args, logger, use_gpu):

    model.train()
    logger.info('-' * 10)
    logger.info('Epoch {}/{}'.format(epoch, args.epochs - 1))
    logger.info('Current Learning rate: {}'.format(showLR(optimizer)))

    running_loss, running_corrects, global_loss, running_all = 0., 0., 0.,0.
    since = time.time()
    last_time_batch_idx = -1
    for batch_idx, (inputs,targets) in enumerate(dset_loaders[phase]):

        label_real = torch.full((inputs.size(0), ), 1)
        label_fake = torch.full((inputs.size(0) ,), 0)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

            target_mi = make_one_hot_global(targets, 1000)#They would be concatenated with final representations(Global)
            label_fake = label_fake.cuda()
            label_real = label_real.cuda()

        outputs = model(inputs)

        _, preds = torch.max(outputs.data, 1)

        optimizer.zero_grad()
        optimizer_Global.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()

        # Paired samples(Global)
        info_real_output_global = model2(target_mi, outputs)
        loss_real_global = BCEcriterion(info_real_output_global.squeeze(), label_real)
        loss_real_global.backward(retain_graph=True)

        # Unpaired samples(Global)
        info_fake_output_global = model2(target_mi, torch.cat(
            (outputs[2:, ...], outputs[0:2, ...]), dim=0))
        loss_fake_global = BCEcriterion(info_fake_output_global.squeeze(), label_fake)
        loss_fake_global.backward()

        optimizer.step()
        optimizer_Global.step()

        # stastics
        running_loss += loss.item() * inputs.size(0)
        batch_correct = (preds == targets.data).sum().item()
        running_corrects += batch_correct
        running_all += len(inputs)

        error_info_global = loss_real_global.item() + loss_fake_global.item()
        global_loss += error_info_global * inputs.size(0)

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
    global_loss_epoch = float(global_loss) / len(dset_loaders[phase].dataset)

    logger.info('{} Epoch:\t{:2}\tLoss: {:.4f}\tAcc:{:.4f}\tglobal:{:.4f}\n'.format(
        phase,
        epoch,
        loss_epoch,
        acc_epoch,global_loss_epoch))

def test(model, save_path, dset_loaders, criterion, epoch, phase, args, logger, use_gpu, save=True):

    model.eval()
    global best_acc
    with torch.no_grad():

        running_loss, running_corrects, running_all = 0., 0., 0.
        since = time.time()
        last_time_batch_idx = -1
        for batch_idx, (inputs, targets) in enumerate(dset_loaders[phase]):

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
            if acc_epoch > best_acc:
                best_acc = max(acc_epoch, best_acc)
                torch.save(model.module.state_dict(), '{}/epoch{}_acc{}.pt'.format(save_path,str(epoch),str(best_acc)))

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

    net = Lipreading(mode=args.mode, inputDim=512, hiddenDim=1024, nClasses=args.nClasses)
    info = GlobalInfo()
    reload_model(net, logger, args.path)
    reload_model(info, logger, args.path2)
    model = torch.nn.DataParallel(net)
    model2 = torch.nn.DataParallel(info)
    if use_gpu:
        model = model.cuda()
        model2 = model2.cuda()
    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.BCELoss()
    params = list(net.gru.parameters()) +list(net.lstm_weight.parameters())+list(net.fc_weight_h.parameters())+list(net.fc.parameters())
    optimizer = optim.Adam(params, lr=args.lr,weight_decay=5e-5)
    optimizer_Global = optim.Adam(info.parameters(), lr=args.lr,weight_decay=5e-5)

    dset_loaders, dset_sizes = data_loader(args)
    if args.test:
        test(model, save_path, dset_loaders, criterion, 0, 'test',  args, logger, use_gpu,  False)
        return

    for epoch in range(args.s_epochs, args.epochs):
        train(model, model2, dset_loaders, criterion,criterion2, epoch, 'train', optimizer,optimizer_Global,args, logger, use_gpu)
        test(model, save_path,dset_loaders, criterion, epoch, 'val', args, logger, use_gpu, True)

if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    test_adam(args, use_gpu)


