#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2021/12/23 21:00
# @Author : ''
# @FileName: utils.py
import os
import sys
import cv2
import json
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def setup_seed(an_int):
    torch.manual_seed(an_int)
    torch.cuda.manual_seed(an_int)
    torch.cuda.manual_seed_all(an_int)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mkdir_file(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_checkpoint(model, checkpoint_path, device):
    """
    Loads weights from checkpoint
        :param model: a pytorch nn student
        :param str checkpoint_path: address/path of a file
        :return: pytorch nn student with weights loaded from checkpoint
    """
    model_ckp = torch.load(checkpoint_path)
    model.load_state_dict(model_ckp)
    return model.to(device)


def get_transform(mode):
    if mode == 'train':  # use for HR and LR during the train process
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if mode == 'val':  # use for LR during the test process
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    else:
        assert "The {} of get_transform is not define!".format(mode)


class MyDataset(Dataset):
    def __init__(self, txt_path, dataset_name, resize_tuple=((8, 8),),
                 downsampling_prob=0.1,
                 train=True, transform=None, interpolation=Image.BICUBIC):
        super(MyDataset, self).__init__()
        self._txt_path = txt_path
        self._dataset_name = dataset_name
        self._root = os.path.dirname(self._txt_path)
        self._resize_tuple = resize_tuple
        self._downsampling_prob = downsampling_prob
        self._train = train
        self._transform = transform
        self._interpolation = interpolation
        self._class_to_idx, self._classes = self._find_classes()
        self._imgs_path = self._read_imgs()
        self._loader = self._get_loader
        self._curriculum_index = 0
        self._step_iterations = 35000
        self._curriculum = True

    def _find_classes(self):
        classes = [d for d in os.listdir(os.path.join(self._root, 'train'))
                   if os.path.isdir(os.path.join(os.path.join(self._root, 'train'), d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        idx_to_class = {i: classes[i] for i in range(len(classes))}
        json_str = json.dumps(idx_to_class, indent=4)
        with open("classes_indices.json", "w") as f:
            f.write(json_str)
        return class_to_idx, classes

    def _read_imgs(self):
        imgs = []
        if self._train:
            with open(os.path.join(self._txt_path,
                                   self._dataset_name + '_train.txt'), 'r') as fh:
                for line in fh:
                    line = line.strip('\n')
                    line = line.rstrip()
                    words = line.split('/')[-2]
                    imgs.append((line, words))  # imgs中包含有train图像路径和标签
        else:
            with open(os.path.join(self._txt_path,
                                   self._dataset_name + '_val.txt'), 'r') as fh:
                for line in fh:
                    line = line.strip('\n')
                    line = line.rstrip()
                    words = line.split('/')[-2]
                    # words = line.split('\\')[-2]
                    imgs.append((line, words))  # imgs中包含有val图像路径和标签
        return imgs

    def _lower_resolution(self, img, resolution):
        w_i, h_i = img.size
        # r = h_i/float(w_i)
        # if self._train:
        #     res = torch.rand(1).item()
        #     res = 3 + 5*res
        #     res = 2**int(res)
        # else:
        #     res = resolution
        # if res >= w_i or res >= h_i:
        #     return img
        # if h_i < w_i:
        #     h_n = res
        #     w_n = h_n/float(r)
        # else:
        #     w_n = res
        #     h_n = w_n*float(r)
        # img2 = img.resize((int(w_n), int(h_n)), self._interpolation)
        # img2 = img2.resize((w_i, h_i), self._interpolation)
        img2 = img.resize((int(resolution), int(resolution)), self._interpolation)
        img2 = img2.resize((w_i, h_i), self._interpolation)
        return img2

    @staticmethod
    def _get_loader(path):
        return Image.fromarray(cv2.imread(path))

    def __getitem__(self, index):
        fn, label_name = self._imgs_path[index]
        # print(fn, label_name)
        label = self._class_to_idx[label_name]
        orgin_img = self._loader(fn)
        if self._train:
            if self._curriculum:
                self._curriculum_index += 1
                if torch.rand(1).item() <= self._downsampling_prob:
                    img1 = self._lower_resolution(orgin_img, self._resize_tuple[0][0])
                else:
                    img1 = orgin_img
            else:
                img1 = self._lower_resolution(orgin_img, self._resize_tuple[0][0])
            if self._transform is not None:
                orgin_img = self._transform(orgin_img)
                img1 = self._transform(img1)
            else:
                assert "transform method is not defined!"
            if (self._curriculum_index % self._step_iterations) == 0 and self._downsampling_prob < 1.0:
                self._downsampling_prob += 0.1
            return orgin_img, img1, label

        else:
            img_copy = self._lower_resolution(orgin_img, self._resize_tuple[0][0])
            if self._transform is not None:
                orgin_img = self._transform(img_copy)
            else:
                assert "transform method is not defined!"
        return orgin_img, label

    def __len__(self):
        return len(self._imgs_path)


class MyDatasetManger():
    def __init__(self, dataset_path, resize_tuple=((8, 8),),
                 downsampling_prob=0.1,
                 dataset_name=None, batch_size=32, num_of_workers=0):
        self._batch_size = batch_size
        self._num_of_workers = num_of_workers
        self._datasets_path = dataset_path
        self._resize_tuple = resize_tuple
        self._downsampling_prob = downsampling_prob
        self._dataset_name = dataset_name
        self._datasets = self._init_datasets()
        self._data_loaders = self._init_data_loaders()

    def _init_datasets(self):
        train_data = MyDataset(self._datasets_path,
                               self._dataset_name,
                               resize_tuple=self._resize_tuple,
                               downsampling_prob=self._downsampling_prob,
                               train=True,
                               transform=get_transform(mode='train'),
                               interpolation=Image.BICUBIC)

        val_data = MyDataset(self._datasets_path,
                             self._dataset_name,
                             downsampling_prob=self._downsampling_prob,
                             resize_tuple=self._resize_tuple,
                             train=False,
                             transform=get_transform(mode='val'),
                             interpolation=Image.BICUBIC)

        return train_data, val_data

    def _init_data_loaders(self):
        train_data_loader = DataLoader(dataset=self._datasets[0],
                                       batch_size=self._batch_size,
                                       shuffle=True,
                                       num_workers=self._num_of_workers)

        val_data_loader = DataLoader(dataset=self._datasets[1],
                                     batch_size=100,
                                     shuffle=False,
                                     num_workers=self._num_of_workers)
        return train_data_loader, val_data_loader

    def get_loaders(self):
        return self._data_loaders


class Logger(object):
    """
    Write console output to external text file.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_file(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
