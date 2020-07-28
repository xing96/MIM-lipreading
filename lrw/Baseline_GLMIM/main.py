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

#path : path to model, when you start Baseline + local + Global, the path is the Baseline of Baseline + Local
parser.add_argument('--path', default=r'', type=str, help='path to model Baseline')
#path2 : path to model Global, when you start Baseline + local + Global, it is empty
parser.add_argument('--path2', default=r'', type=str, help='path to model Global')
# path3 : path to model Local, when you start Baseline + local + Global, the path is the Local model of Baseline + Local. You could also ignore this model and fix the Fro-tend.
parser.add_argument('--path3', default=r'', type=str, help='path to model Local')
parser.add_argument('--dataset', default='', type=str, help='path/to/lrw/roi_80_116_175_211_npy_gray/')
parser.add_argument('--mode', default='Baseline_Local_Global', type=str)
parser.add_argument('--every-frame', default=True, action='store_false', help='predicition based on every frame')
parser.add_argument('--lr', default=1e-5, type=float, help='initial learning rate')
parser.add_argument('--batch-size', default= 100, type=int, help='mini-batch size')
parser.add_argument('--workers', default=10, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=500, type=int, help='number of total epochs')
parser.add_argument('--s_epochs', default=1, type=int, help='number of start epochs')
parser.add_argument('--interval', default=30, type=int, help='display interval')
parser.add_argument('--test', default=False, action='store_false', help='perform on the test phase')
args = parser.parse_args()
print(args)
best_acc = -1

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
def make_one_hot_by_time_local(label,length,time):
    batch = label.size(0)
    a = torch.zeros(batch, length).scatter_(1, label.view(-1, 1), 1)
    a = a.view(batch,length,1).expand(batch,length,time).transpose(2,1).contiguous().view(batch*time,length)
    return a
def make_one_hot_global(label,length):
    batch = label.size(0)
    a = torch.zeros(batch, length).scatter_(1, label.view(-1, 1), 1)
    return a
def data_loader(args):
    dsets = {x: MyDataset(x, args.dataset) for x in ['train', 'val', 'test']}
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

def train(model, model2, model3,dset_loaders, criterion, BCEcriterion, epoch, phase, optimizer ,optimizer_info,args, logger, use_gpu):

    model.train()
    logger.info('-' * 10)
    logger.info('Epoch {}/{}'.format(epoch, args.epochs - 1))
    logger.info('Current Learning rate: {}'.format(showLR(optimizer)))

    running_loss, running_corrects, global_loss, running_all = 0., 0., 0.,0.
    since = time.time()
    last_time_batch_idx = -1
    for batch_idx, (inputs, targets) in enumerate(dset_loaders[phase]):
        batch_img = RandomCrop(inputs.numpy(), (88, 88))
        batch_img = ColorNormalize(batch_img)
        batch_img = HorizontalFlip(batch_img)

        batch_img = np.reshape(batch_img, (batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], batch_img.shape[3], 1))
        inputs = torch.from_numpy(batch_img)
        inputs = inputs.float().permute(0, 4, 1, 2, 3).contiguous()

        label_real = torch.full((inputs.size(0), ), 1)
        label_fake = torch.full((inputs.size(0) ,), 0)

        label_real_local = torch.full((inputs.size(0) * 29, 1, 3, 3), 1)
        label_fake_local = torch.full((inputs.size(0) * 29, 1, 3, 3), 0)


        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

            targets_mi_local = make_one_hot_by_time_local(targets, 500, 29)
            target_mi_local = targets_mi_local.unsqueeze(2).unsqueeze(3).repeat(1, 1, 3, 3)#They would be concatenated with features(Local)

            target_mi = make_one_hot_global(targets, 500)#They would be concatenated with final representations(Global)
            label_fake = label_fake.cuda()
            label_real = label_real.cuda()

            label_real_local = label_real_local.cuda()
            label_fake_local = label_fake_local.cuda()

        outputs,resnet_feature = model(inputs)

        _, preds = torch.max(outputs.data, 1)

        optimizer.zero_grad()
        optimizer_info.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward(retain_graph=True)

        # Paired samples(Global)
        info_real_output_global = model2(target_mi, outputs)
        loss_real_global = BCEcriterion(info_real_output_global.squeeze(), label_real)
        loss_real_global.backward(retain_graph=True)

        # Unaired samples(Global)
        info_fake_output_global = model2(target_mi, torch.cat(
            (outputs[2:, ...], outputs[0:2, ...]), dim=0))
        loss_fake_global = BCEcriterion(info_fake_output_global.squeeze(), label_fake)
        loss_fake_global.backward(retain_graph=True)

        # Paired samples(Local)
        info_real_output_local = model3(torch.cat((target_mi_local, resnet_feature), 1))
        loss_real_local = BCEcriterion(info_real_output_local, label_real_local)
        loss_real_local.backward(retain_graph=True)

        # Unaired samples(Local)
        info_fake_output_local = model3(torch.cat((target_mi_local, torch.cat(
            (resnet_feature[29:, ...], resnet_feature[0:29, ...]), dim=0)), 1))
        loss_fake_local = BCEcriterion(info_fake_output_local, label_fake_local)
        loss_fake_local.backward()

        optimizer.step()
        optimizer_info.step()

        # stastics
        running_loss += loss.item() * inputs.size(0)
        batch_correct = (preds == targets.data).sum().item()
        running_corrects += batch_correct
        running_all += len(inputs)

        error_info_global = loss_real_global.item() + loss_fake_global.item()
        global_loss += error_info_global * inputs.size(0)

        D_real = info_real_output_global.mean().item()
        D_fake = info_fake_output_global.mean().item()

        D_real_local = info_real_output_local.mean().item()
        D_fake_local = info_fake_output_local.mean().item()

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
            print('global info loss rnn: D_real:{:.4f}\tD_fake:{:.4f}'.format(
                D_real,
                D_fake
            ))
            print('local info loss cnn: D_real:{:.4f}\tD_fake:{:.4f}'.format(
                D_real_local,
                D_fake_local
            ))

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

def test(model, model2,model3, save_path, dset_loaders, criterion, epoch, phase, args, logger, use_gpu, save=True):

    model.eval()
    global best_acc
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
            outputs,_ = model(inputs)

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
                torch.save(model2.module.state_dict(), '{}/epoch{}_info_rnn_acc{}.pt'.format(save_path,str(epoch),str(best_acc)))
                torch.save(model3.module.state_dict(), '{}/epoch{}_info_local_acc{}.pt'.format(save_path,str(epoch),str(best_acc)))

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
    info = GlobalInfo()
    info_local = LocalDiscriminator()
    reload_model(net, logger, args.path)
    reload_model(info, logger, args.path2)
    reload_model(info_local, logger, args.path3)
    # define loss function and optimizer
    model = torch.nn.DataParallel(net)
    model2 = torch.nn.DataParallel(info)
    model3 = torch.nn.DataParallel(info_local)
    if use_gpu:
        model = model.cuda()
        model2 = model2.cuda()
        model3 = model3.cuda()
    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.BCELoss()

    params = net.parameters()
    optimizer = optim.Adam(params, lr=args.lr,weight_decay=5e-5)
    optimizer_info = optim.Adam(list(info.parameters())+list(info_local.parameters()), lr=args.lr,weight_decay=5e-5)

    dset_loaders, dset_sizes = data_loader(args)
    if args.test:
        test(model, model2,model3, save_path, dset_loaders, criterion, 0, 'test',  args, logger, use_gpu,  False)
        return

    for epoch in range(args.s_epochs, args.epochs):
        train(model, model2,model3, dset_loaders, criterion,criterion2, epoch, 'train', optimizer,optimizer_info,args, logger, use_gpu)
        test(model, model2,model3, save_path,dset_loaders, criterion, epoch, 'val', args, logger, use_gpu, True)

if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    test_adam(args, use_gpu)


