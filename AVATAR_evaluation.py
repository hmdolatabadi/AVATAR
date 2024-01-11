import argparse
import logging
import os
import lmdb
import pickle
from io import BytesIO
from PIL import Image
import random

import torch

import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.ResNet import ResNet18, ResNet50
from models.DenseNet import DenseNet121
from models.vgg import VGG
from models.wideresnet import WideResNet

from utils import CutMix, MixUp, Cutout, CutMixCrossEntropyLoss

from fast_autoaugment.FastAutoAugment.archive import fa_reduced_cifar10
from fast_autoaugment.FastAutoAugment.augmentations import apply_augment
from ISS_augs import *
from tqdm import tqdm
from utils import AverageMeter

class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img

def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)

class ImageFolderLMDB(torch.utils.data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env     = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                                 readonly=True, lock=False,
                                 readahead=False, meminit=False)
        
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys   = loads_data(txn.get(b'__keys__'))

        self.transform        = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        buffer = BytesIO(imgbuf)
        img    = Image.open(buffer).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        im2arr = np.array(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return im2arr, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    
    # diffusion models
    parser.add_argument('--path', type=str, default='./data/', help='Path to the data files')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--arch', type=str, default='ResNet18', help='Model Architecture')
    
    parser.add_argument('--use_cutout', action='store_true')
    parser.add_argument('--use_mixup', action='store_true')
    parser.add_argument('--use_cutmix', action='store_true')
    parser.add_argument('--use_fa', action='store_true')

    parser.add_argument('--grayscale', default=False, action='store_true', help='grayscale compression')
    parser.add_argument('--jpeg', default=None, type=int, help='JPEG quality factor')
    parser.add_argument('--bdr', default=None, type=int, help='bit depth')
    parser.add_argument('--TrainAUG', default='', help='Train augmentations')
    parser.add_argument('--lowpass', default='', help='filtering')
    parser.add_argument('--ISS_both_train_test', action='store_true', default=False)
    
    # data
    parser.add_argument('--domain', type=str, default='cifar10', help='which domain: cifar10, cifar100, celebahq, imagenet')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=120, help='training epochs')
    parser.add_argument('--unlearnable_alg', type=str, required=True, help='which unlearnable algorithm to sanitize [UNL, NTGA, ADV]')

    args = parser.parse_args()

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    args.device = device

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args

def train_model(args):

    print(f'Training {args.arch} for a {args.unlearnable_alg} {args.domain} stored in {args.path}...')

    if args.domain == 'cifar10':
        # Prepare Dataset
        train_transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]

        test_transform = [
            transforms.ToTensor()
        ]

        train_transform = transforms.Compose(train_transform)
        test_transform  = transforms.Compose(test_transform)

        train_transform = aug_train(args.jpeg, args.grayscale, args.bdr, args.TrainAUG, args.lowpass)
        test_transform  = aug_test(args.ISS_both_train_test, args.jpeg, args.grayscale, args.bdr)

        if args.use_fa:
            # FastAutoAugment
            print('Using Fast Auto')
            train_transform.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
        elif args.use_cutout:
            print('Using Cutout')
            train_transform.transforms.append(Cutout(16))


        train_dataset = datasets.CIFAR10(root='../datasets', train=True, download=True, transform=train_transform)

        # Get Target Dataset
        unlearnable_dataset   = np.load(f'{args.path}')
        train_dataset.data    = unlearnable_dataset['data']
        train_dataset.targets = unlearnable_dataset['targets']


        if args.use_cutmix:
            print('Using Cutmix')
            train_dataset = CutMix(dataset=train_dataset, num_class=10)
        elif args.use_mixup:
            print('Using Mixup')
            train_dataset = MixUp(dataset=train_dataset, num_class=10)

        test_dataset = datasets.CIFAR10(root='../datasets', train=False, download=True, transform=test_transform)
        
    elif args.domain == 'cifar100':
        # Prepare Dataset
        train_transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]

        test_transform = [
            transforms.ToTensor()
        ]

        train_transform = transforms.Compose(train_transform)
        test_transform  = transforms.Compose(test_transform)

        if args.use_fa:
            # FastAutoAugment
            print('Using Fast Auto')
            train_transform.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
        elif args.use_cutout:
            print('Using Cutout')
            train_transform.transforms.append(Cutout(16))

        
        if args.unlearnable_alg == 'CLEAN':
            train_dataset = datasets.CIFAR100(root='../datasets', train=True, download=True, transform=train_transform)
        else:
            train_dataset = datasets.CIFAR100(root='../datasets', train=True, download=True, transform=train_transform)
            
            # Get Unlearnable Dataset
            unlearnable_dataset = np.load(f'{args.path}')
            train_dataset.data, train_dataset.targets = unlearnable_dataset['data'], unlearnable_dataset['targets']
            
        if args.use_cutmix:
            print('Using Cutmix')
            train_dataset = CutMix(dataset=train_dataset, num_class=10)
        elif args.use_mixup:
            print('Using Mixup')
            train_dataset = MixUp(dataset=train_dataset, num_class=10)

        test_dataset = datasets.CIFAR100(root='../datasets', train=False, download=True, transform=test_transform)

    elif args.domain == 'svhn':
        # Prepare Dataset
        train_transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]

        test_transform = [
            transforms.ToTensor()
        ]

        train_transform = transforms.Compose(train_transform)
        test_transform  = transforms.Compose(test_transform)

        if args.use_fa:
            # FastAutoAugment
            print('Using Fast Auto')
            train_transform.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
        elif args.use_cutout:
            print('Using Cutout')
            train_transform.transforms.append(Cutout(16))

        
        if args.unlearnable_alg == 'CLEAN':
            train_dataset = datasets.SVHN(root='../datasets', split='train', download=True, transform=train_transform)
        else:
            train_dataset = datasets.SVHN(root='../datasets', split='train', download=True, transform=train_transform)
            
            # Get Unlearnable Dataset
            unlearnable_dataset = np.load(f'{args.path}')
            train_dataset.data, train_dataset.labels = unlearnable_dataset['data'], unlearnable_dataset['targets']
            
        if args.use_cutmix:
            print('Using Cutmix')
            train_dataset = CutMix(dataset=train_dataset, num_class=10)
        elif args.use_mixup:
            print('Using Mixup')
            train_dataset = MixUp(dataset=train_dataset, num_class=10)

        test_dataset = datasets.SVHN(root='../datasets', split='test', download=True, transform=test_transform)
        
    elif args.domain == 'imagenet':
        # Prepare Dataset
        train_transform = [transforms.RandomResizedCrop([224, 224]),
                           transforms.RandomHorizontalFlip(),
                           transforms.ColorJitter(brightness=0.1,
                                                  contrast=0.1,
                                                  saturation=0.1),
                           transforms.ToTensor()]

        test_transform = [transforms.Resize([256, 256]),
                          transforms.CenterCrop([224, 224]),
                          transforms.ToTensor()]

        train_transform = transforms.Compose(train_transform)
        test_transform  = transforms.Compose(test_transform)

        # Get Unlearnable Dataset
        train_dataset = ImageFolderLMDB(args.path, transform=train_transform)
        test_dataset  = ImageFolderLMDB('./data/MiniIN_val.lmdb', transform=test_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False, num_workers=12)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=args.batch_size * 2, shuffle=False, pin_memory=True, drop_last=False, num_workers=12)

    # Train DNN on Unlearnable Dataset
    if args.arch == 'ResNet18':
        model = ResNet18(num_classes=args.num_classes)
    elif args.arch == 'ResNet50':
        model = ResNet50(num_classes=args.num_classes)
    elif args.arch == 'WideResNet34':
        model = WideResNet(depth=34, widen_factor=2, num_classes=args.num_classes)
    elif args.arch == 'VGG16':
        model = VGG('VGG16', num_classes=args.num_classes)
    elif args.arch == 'DenseNet121':
        model = DenseNet121(num_classes=args.num_classes)
    else:
        raise ValueError(f'No such model name: {args.arch}')

    model     = model.to(args.device)
    criterion = CutMixCrossEntropyLoss() if (args.use_cutmix or args.use_mixup) else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, weight_decay=0.0005, momentum=0.9)

    lr_drop_epochs = [80, 100]
    best_acc       = 0

    for epoch in range(args.epochs):

        lr = 0.1
        for lr_drop_epoch in lr_drop_epochs:
            if epoch >= lr_drop_epoch:
                lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Train
        model.train()
        acc_meter  = AverageMeter()
        loss_meter = AverageMeter()
        pbar = tqdm(train_loader, total=len(train_loader), disable=True)
        
        for images, labels in pbar:
            images, labels = images.to(args.device), labels.to(args.device)
            model.zero_grad()
            optimizer.zero_grad()

            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)

            if args.use_cutmix or args.use_mixup:
                _, labels = torch.max(labels.data, 1)
                
            acc = (predicted == labels).sum().item()/labels.size(0)
            acc_meter.update(acc)
            loss_meter.update(loss.item())
            pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))

        # Eval
        model.eval()
        correct, total = 0, 0
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            with torch.no_grad():
                logits = model(images)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        tqdm.write('Clean Accuracy %.2f\n' % (acc*100))

        if acc >= best_acc:
            best_acc   = acc
            ckpt_state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                }

            # save checkpoint
            torch.save(ckpt_state, os.path.join(args.log_dir, 'best_model.pth'))
    
    tqdm.write('Best Clean Accuracy %.2f\n' % (best_acc*100))

if __name__ == '__main__':
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = True

    args         = parse_args_and_config()
    log_dir      = os.path.join('results', args.arch, args.unlearnable_alg, 'T' + args.path.split('_')[3], 'seed' + str(args.seed))
    args.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)

    train_model(args)