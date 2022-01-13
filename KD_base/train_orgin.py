#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2021/12/23 21:00
# @Author : ''
# @FileName: train.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import itertools
from torch.optim import lr_scheduler
from tqdm import tqdm
from script import mkdir_file, setup_seed, MyDatasetManger, Logger
from data_loader import CIFAR10, SVHN, get_transform
from model import resnet18
from loss import ArcNet

setup_seed(42)


def main():
    # define hyper parameters
    model_name = "resnet18"
    dataset_name = "cifar10"  # cifar10, SVHN
    attr = "face"
    batch_size = 128
    downsample_size = 32
    lr = 0.1
    num_classes = 10
    epochs = 200
    stepsize = 30

    save_dir = './logs/' + model_name + "_" + str(downsample_size) + dataset_name
    save_path = './model/'
    mkdir_file(save_path)
    mkdir_file(save_dir)

    model_ckp = save_path + model_name + '_tm_' + str(downsample_size) + dataset_name + '.pth'
    tb_writer = SummaryWriter()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use device: ', device)  # print used device during the training
    # # define dataset path
    # image_path = "../datasets/" + dataset_name + "/"
    # assert os.path.join(image_path), "{} path does not exist!".format(image_path)
    # define net count
    net_count = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of network
    sys.stdout.write("Dataset name: {} \n".format(dataset_name))
    sys.stdout.write("Use {} dataloader workers every process \n".format(net_count))

    sys.stdout = Logger(os.path.join(save_dir, 'log_' + dataset_name + '.txt'))
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

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=net_count)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False, num_workers=net_count)
    val_num = len(val_dataset)
    tm_net = resnet18(num_classes=num_classes)
    if os.path.exists(model_ckp):
        sys.stdout.write("load trunk weights file: {} \n".format(model_ckp))
        pre_weights = torch.load(model_ckp)
        tm_net.load_state_dict(pre_weights, strict=True)
    else:
        assert "Not have teacher model!!!"

    tm_net = tm_net.to(device)

    # define basic soft max function and center loss with two optimizer
    if attr == "arcface":
        arcface_net = ArcNet(cls_num=num_classes, feature_dim=512).to(device)
        # define loss function
        loss_fun = nn.NLLLoss()
        optimizer = optim.SGD(itertools.chain(tm_net.parameters(), arcface_net.parameters()),
                              lr=lr,
                              weight_decay=5e-4,
                              momentum=0.9)
        # optimizer = optim.Adam(itertools.chain(tm_net.parameters(), arcface_net.parameters()),
        #                        lr=lr)
    else:
        loss_entropy = nn.CrossEntropyLoss()
        optimizer = optim.SGD(tm_net.parameters(),
                              lr=lr,
                              weight_decay=5e-4,
                              momentum=0.9)
        # optimizer = optim.Adam(tm_net.parameters(),
        #                        lr=lr)

    best_acc = 0.0
    train_steps = len(train_loader)

    if stepsize > 0:
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=0.1)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode="min",
                                                   factor=0.5,
                                                   patience=8,
                                                   verbose=True,
                                                   min_lr=1.e-8,
                                                   threshold=0.1)
        # scheduler = lr_scheduler.MultiStepLR(optimizer,
        #                                      milestones=[60, 100, 140, 170, 200, 230, 260, 290],
        #                                      gamma=0.1)

    # ---------------------------------Train process---------------------------------
    for epoch in range(epochs):
        # training process
        tm_net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            _, images, labels = data

            optimizer.zero_grad()

            [features, out] = tm_net(images.to(device))
            if attr == "arcface":
                out = arcface_net(features)
                loss = loss_fun(out, labels.to(device))  # arcface loss
            else:
                loss = loss_entropy(out, labels.to(device))

            loss.backward()
            optimizer.step()

            # print info
            running_loss += loss.item()
            train_bar.desc = "Train epoch [{}/{}], loss_arc:{:.3f}". \
                format(epoch + 1, epochs, loss)

        if stepsize > 0:
            # scheduler.step()
            scheduler.step(running_loss / train_steps)

        # ----------------------------------Evaluation process-----------------------------
        tm_net.eval()
        if attr == "arcface":
            arcface_net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_losses = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for val_images, val_labels in val_bar:
                [feas, outs] = tm_net(val_images.to(device))
                if attr == "arcface":
                    outs = arcface_net(feas)
                    # calculate loss
                    val_loss = loss_fun(outs, val_labels.to(device))  # arcface loss
                else:
                    val_loss = loss_entropy(outs, val_labels.to(device))
                val_losses += val_loss.item()
                pred_y = torch.max(outs, dim=1)[1]
                acc += torch.eq(pred_y, val_labels.to(device)).sum().item()

        val_acc = acc / val_num
        val_losses = val_losses / val_num
        sys.stdout.write("Use {} batch for training, {} images for validation.\n".format(len(train_dataset), val_num))
        sys.stdout.write('[epoch %d]: Training loss: %.3f, Eval loss: %.3f, Eval accuracy: %.3f\n'
                         % (epoch + 1, running_loss / train_steps, val_losses, val_acc))

        # tensorbord log
        tags = ["train_loss", "eval_loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], running_loss / train_steps, epoch)
        tb_writer.add_scalar(tags[1], val_losses, epoch)
        tb_writer.add_scalar(tags[2], acc, epoch)
        tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(tm_net.state_dict(), save_path + model_name + '_tm_' + str(downsample_size) + dataset_name + '.pth')

    sys.stdout.write('Training best accuracy:{} \n'.format(best_acc))
    sys.stdout.write('Finished Training!')


if __name__ == '__main__':
    main()
