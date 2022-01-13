#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2021/12/23 21:00
# @Author : ''
# @FileName: train.py
import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from script import mkdir_file, setup_seed, Logger, load_checkpoint
from model import resnet18
from data_loader import CIFAR10, SVHN, get_transform

setup_seed(42)


def eval(sm_net, val_loader, val_num, criterion_fun, device):
    # ----------------------------------Evaluation process-----------------------------
    sm_net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    val_losses = 0.0
    with torch.no_grad():
        val_bar = tqdm(val_loader)
        for val_data in val_bar:
            val_images, val_labels = val_data
            [_, outputs] = sm_net(val_images.to(device))
            val_loss = criterion_fun(outputs, val_labels.to(device))
            val_losses += val_loss.item()
            pred_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(pred_y, val_labels.to(device)).sum().item()

    val_acc = acc / val_num
    val_losses = val_losses / val_num
    return val_losses, val_acc


def main():
    # define hyper parameters
    model_name = 'resnet18'
    dataset_name = 'cifar10'
    batch_size = 128
    downsample_size = 8
    lr = 0.1
    num_classes = 10
    epochs = 150
    beta = 2
    stepsize = 30

    # define save path including model save path and logs path
    save_dir = './logs/' + model_name
    save_path = './model/'
    mkdir_file(save_path)
    mkdir_file(save_dir)

    # define load model from file
    model_ckp = save_path + model_name + '_tm_' + str(downsample_size) + dataset_name + '.pth'
    # initialize a instance for summary writer from tensorboard
    tb_writer = SummaryWriter()

    # initialize a instance in Logger
    sys.stdout = Logger(os.path.join(save_dir,
                                     'log_' + model_name + '_sm_' + "beta_" + str(beta) + dataset_name + '.txt'))

    # define device and print used device during the training
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sys.stdout.write('Use device: {}\n'.format(device))

    # ------------------------data processing-----------------------------
    net_count = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of network
    sys.stdout.write("Use {} dataloader workers every process\n".format(net_count))

    # define dataset
    if dataset_name == "cifar10":
        train_dataset = CIFAR10(root="../datasets/",
                                train=True,
                                download=True,
                                transform=get_transform("train"),
                                downsample_size=downsample_size)

        val_dataset = CIFAR10(root="../datasets/",
                              train=False,
                              download=True,
                              transform=get_transform("val"),
                              downsample_size=downsample_size)

    elif dataset_name == "SVHN":
        train_dataset = SVHN(root="../datasets/",
                             split='train',
                             download=True,
                             transform=get_transform("train"),
                             downsample_size=downsample_size)

        val_dataset = SVHN(root="../datasets/",
                           split='test',
                           download=True,
                           transform=get_transform("val"),
                           downsample_size=downsample_size)
    else:
        pass

    train_num = len(train_dataset)
    val_num = len(val_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=net_count)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=net_count)

    # define model
    tm_net = resnet18(num_classes=num_classes)
    sm_net = resnet18(num_classes=num_classes)

    # load model
    if os.path.isfile(model_ckp):
        sys.stdout.write("Load teacher weights file: {}\n".format(model_ckp))
        tm_net = load_checkpoint(tm_net, model_ckp, device)
        sys.stdout.write("Load student weights file: {}\n".format(model_ckp))
        sm_net = load_checkpoint(sm_net, model_ckp, device)
    else:
        assert "Not have teacher model!!!"

    # freeze teacher model
    for params in tm_net.parameters():
        params.requires_grad = False

    # optimizer = optim.Adam(sm_net.parameters(), lr=lr, weight_decay=5e-4)
    optimizer = optim.SGD(sm_net.parameters(),
                          lr=lr,
                          weight_decay=5e-04,
                          momentum=0.9)

    best_acc = 0.0
    train_steps = len(train_loader)

    if stepsize > 0:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode="min",
                                                   factor=0.5,
                                                   patience=stepsize,
                                                   verbose=True,
                                                   min_lr=1.e-8,
                                                   threshold=0.1)
        # scheduler = lr_scheduler.MultiStepLR(optimizer,
        #                                      milestones=[50, 80, 100, 140, 160, 180],
        #                                      gamma=0.1)

    # ---------------------------------Train process---------------------------------
    for epoch in range(epochs):
        # training process
        tm_net.eval()
        sm_net.train()
        running_loss, entropy_loss, mse_loss = 0.0, 0.0, 0.0
        train_acc = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, lr1_images, labels = data

            optimizer.zero_grad()

            features, soft = tm_net(images.to(device))
            l1_fea, soft_l1 = sm_net(lr1_images.to(device))

            loss_entropy = F.cross_entropy(soft_l1, labels.to(device))
            loss_mse = F.mse_loss(l1_fea, features)

            loss = loss_entropy + beta * loss_mse

            loss.backward()
            optimizer.step()

            # evaluate train samples in teacher model
            pred_y = torch.max(soft, dim=1)[1]
            train_acc += torch.eq(pred_y, labels.to(device)).sum().item()

            # print info
            running_loss += loss.item()
            entropy_loss += loss_entropy.item()
            mse_loss += loss_mse.item()
            train_bar.desc = "Train epoch [{}/{}], " \
                             "Total loss:{:.3f}," \
                             "Loss:{:.3f}," \
                             "loss_mse:{:.3f}". \
                format(epoch + 1, epochs, loss, loss_entropy, loss_mse)

        val_losses, val_acc = eval(sm_net, val_loader, val_num, criterion_fun, device)

        sys.stdout.write("Use {} batch for training, {} images for validation.\n".
                         format(train_num, val_num))
        sys.stdout.write('[epoch %d]: Training loss: %.3f, '
                         'Teacher acc: %.3f, '
                         'Eval loss: %.3f, '
                         'Eval accuracy: %.3f\n'
                         % (epoch + 1, running_loss / train_steps, train_acc / train_num,
                            val_losses, val_acc))

        # tensorbord log
        tags = ["train_loss", "entropy_loss", "mse_loss",
                "eval_loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], running_loss / train_steps, epoch)
        tb_writer.add_scalar(tags[1], entropy_loss / train_steps, epoch)
        tb_writer.add_scalar(tags[2], mse_loss / train_steps, epoch)
        tb_writer.add_scalar(tags[3], val_losses, epoch)
        tb_writer.add_scalar(tags[4], val_acc, epoch)
        tb_writer.add_scalar(tags[5], optimizer.param_groups[0]["lr"], epoch)

        if stepsize > 0:
            scheduler.step(running_loss / train_steps)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(sm_net.state_dict(),
                       save_path + model_name + '_sm_' + "beta_" + str(beta) + dataset_name + '.pth')

    print('Training best accuracy: ', best_acc)
    print('Finished Training!')


if __name__ == '__main__':
    main()
