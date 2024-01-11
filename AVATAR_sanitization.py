import argparse
import logging
import yaml
import os
import time
import utils
import lmdb
import pickle
from io import BytesIO
from utils import str2bool
from PIL import Image

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from glob import glob

from runners.diffpure_ddpm import Diffusion
from runners.diffpure_guided import GuidedDiffusion
from runners.diffpure_sde import RevGuidedDiffusion
from runners.diffpure_ode import OdeGuidedDiffusion
from runners.diffpure_ldsde import LDGuidedDiffusion

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    
    # diffusion models
    parser.add_argument('--path', type=str, default='./data/', help='Path to the data files')
    parser.add_argument('--save_path', type=str, default='./data/', help='Path to save the sanitized data files')
    parser.add_argument('--config', type=str, default='cifar10.yml', help='Path to the config file')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=400, help='Sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')
    parser.add_argument('--rand_t', type=str2bool, default=False, help='Decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='sde', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='score_sde', help='[guided_diffusion, score_sde]')
    parser.add_argument('--eot_iter', type=int, default=20, help='only for rand version of autoattack')
    parser.add_argument('--use_bm', action='store_true', help='whether to use brownian motion')

    # LDSDE
    parser.add_argument('--sigma2', type=float, default=1e-3, help='LDSDE sigma2')
    parser.add_argument('--lambda_ld', type=float, default=1e-2, help='lambda_ld')
    parser.add_argument('--eta', type=float, default=5., help='LDSDE eta')
    parser.add_argument('--step_size', type=float, default=1e-2, help='step size for ODE Euler method')
    
    # data
    parser.add_argument('--domain', type=str, default='cifar10', help='which domain: celebahq, cat, car, imagenet')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--unlearnable_alg', type=str, required=True, help='which unlearnable algorithm to sanitize [UNL, NTGA, ADV]')

    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = utils.dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    args.image_folder = os.path.join(args.exp, args.image_folder)
    os.makedirs(args.image_folder, exist_ok=True)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args, new_config

def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)

def dumps_data(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj)

def save_lmdb(args, data, targets):
    name            = f'{args.unlearnable_alg}_MiniIN_Sanitized_{args.t}'
    write_frequency = 5000

    lmdb_path = os.path.join('./data/', "%s.lmdb" % name)
    isdir     = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)

    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)

    for idx in range(data.shape[0]):
        
        image  = Image.fromarray(np.uint8((data[idx]).clip(0, 255)))
        buffer = BytesIO()
        image.save(buffer, format="png", quality=100)
        val   = buffer.getvalue()
        label = targets[idx]

        # Create a tuple of image and label
        imglabel_tuple = (val, label)

        txn.put(u'{}'.format(idx).encode('ascii'), dumps_data(imglabel_tuple))
        
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, data.shape[0]))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_data(keys))
        txn.put(b'__len__', dumps_data(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

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

class SDE_Sanitizer(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args

        # diffusion model
        print(f'diffusion_type: {args.diffusion_type}')
        if args.diffusion_type == 'ddpm':
            self.runner = GuidedDiffusion(args, config, device=config.device, dataset='imagenet32' if 'in32' in args.config else 'imagenet')
        elif args.diffusion_type == 'sde':
            self.runner = RevGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'ode':
            self.runner = OdeGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'ldsde':
            self.runner = LDGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'celebahq-ddpm':
            self.runner = Diffusion(args, config, device=config.device)
        else:
            raise NotImplementedError('unknown diffusion type')

        self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.tag = None

    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=config.device)

    def set_tag(self, tag=None):
        self.tag = tag

    def forward(self, x):
        counter = self.counter.item()
        if counter % 5 == 0:
            print(f'diffusion times: {counter}')

        # imagenet [3, 224, 224] -> [3, 256, 256] -> [3, 224, 224]
        if 'imagenet' in self.args.domain:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        elif 'webface' in self.args.domain:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        start_time = time.time()
        x_re = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag)
        minutes, seconds = divmod(time.time() - start_time, 60)

        if 'imagenet' in self.args.domain:
            x_re = F.interpolate(x_re, size=(224, 224), mode='bilinear', align_corners=False)
        elif 'webface' in self.args.domain:
            x_re = F.interpolate(x_re, size=(112, 112), mode='bilinear', align_corners=False)
        

        if counter % 5 == 0:
            print(f'x shape (before diffusion models): {x.shape}')
            print(f'x shape (before classifier): {x_re.shape}')
            print("Sampling time per batch: {:0>2}:{:05.2f}".format(int(minutes), seconds))

        out = (x_re + 1) * 0.5

        self.counter += 1

        return out

def sanitize(args, config):

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

        if args.unlearnable_alg == 'CLEAN':
            train_dataset = datasets.CIFAR10(root='../datasets', train=True, download=True, transform=test_transform)
        else:
            train_dataset = datasets.CIFAR10(root='../datasets', train=True, download=True, transform=test_transform)
            
            # Get Unlearnable Dataset
            unlearnable_dataset = np.load(f'{args.path}')
            train_dataset.data, train_dataset.targets = unlearnable_dataset['data'], unlearnable_dataset['targets']

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

        if args.unlearnable_alg == 'CLEAN':
            train_dataset = datasets.CIFAR100(root='../datasets', train=True, download=True, transform=test_transform)
        else:
            train_dataset = datasets.CIFAR100(root='../datasets', train=True, download=True, transform=test_transform)
            
            # Get Unlearnable Dataset
            unlearnable_dataset = np.load(f'{args.path}')
            train_dataset.data, train_dataset.targets = unlearnable_dataset['data'], unlearnable_dataset['targets']

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

        if args.unlearnable_alg == 'CLEAN':
            train_dataset = datasets.SVHN(root='../datasets', split='train', download=True, transform=test_transform)
        else:
            train_dataset = datasets.SVHN(root='../datasets', split='train', download=True, transform=test_transform)
            
            # Get Unlearnable Dataset
            unlearnable_dataset = np.load(f'{args.path}')
            train_dataset.data, train_dataset.labels = unlearnable_dataset['data'], unlearnable_dataset['targets']


    elif args.domain == 'imagenet':

        test_transform = [transforms.ToTensor(),]
        test_transform = transforms.Compose(test_transform)

        # Get Unlearnable Dataset
        train_dataset = ImageFolderLMDB(args.path, transform=test_transform)

    elif args.domain == 'webface':

        clean_files = sorted(glob(os.path.join(args.path, '*')))
        pois_cls    = np.load('./data/WebFace/poisoned_classes_non_overlapping.npy')
        pois_files  = [clean_files[idx].split('/')[-1] for idx in pois_cls] 
        pois_count  = 0
        img_dataset = []
        labels      = []

        for idx, file in enumerate(pois_files):
            target = pois_cls[idx]
            images = sorted(glob(os.path.join(args.path, file, '*.jpg')))
            for image in images:
                img_dataset.append(np.array(Image.open(image).convert('RGB'), dtype=np.uint8))
                labels.append(int(target))
                pois_count += 1
                
        img_dataset   = torch.tensor(np.array(img_dataset).transpose(0, 3, 1, 2).astype(np.float32))/255.
        targets       = torch.tensor(np.array(labels))
        train_dataset = torch.utils.data.TensorDataset(img_dataset, targets)

    data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=4)

    # load model
    print(f'starting the model and loader for {args.unlearnable_alg} and {args.t}...')

    sanitizer_model = SDE_Sanitizer(args, config)
    sanitizer_model.eval().to(config.device)

    pbar             = tqdm(data_loader, total=len(data_loader))

    if args.domain == 'cifar10' or args.domain == 'cifar100':

        sanitized_images = np.zeros_like(train_dataset.data)
        sanitizer_model.reset_counter()

        for images, labels in pbar:
            
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                images = sanitizer_model(images)
                images = images.cpu()

            torch.cuda.empty_cache() 
            sanitized_images[(sanitizer_model.counter - 1) * args.batch_size: min(len(train_dataset), sanitizer_model.counter * args.batch_size)] = (255 * images.numpy().transpose(0, 2, 3, 1)).astype(np.uint8)

        diff = args.config.split('.')[0]
        np.savez(os.path.join(args.save_path, f'{args.unlearnable_alg}_{args.domain.upper()}_Sanitized_{args.t}_{diff}.npz'), data=sanitized_images, targets=train_dataset.targets)

    elif args.domain == 'svhn':

        sanitized_images = np.zeros_like(train_dataset.data)
        sanitizer_model.reset_counter()

        for images, labels in pbar:
            
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                images = sanitizer_model(images)
                images = images.cpu()

            torch.cuda.empty_cache() 
            sanitized_images[(sanitizer_model.counter - 1) * args.batch_size: min(len(train_dataset), sanitizer_model.counter * args.batch_size)] = (255 * images.numpy()).astype(np.uint8)

        diff = args.config.split('.')[0]
        np.savez(os.path.join(args.save_path, f'{args.unlearnable_alg}_{args.domain.upper()}_Sanitized_{args.t}_{diff}.npz'), data=sanitized_images, targets=train_dataset.labels)


    elif args.domain == 'imagenet':
        sanitized_images = np.zeros((len(train_dataset), 224, 224, 3), dtype=np.uint8)
        sanitizer_model.reset_counter()
        targets = []
        for images, labels in pbar:
            
            images = images.cuda()
            targets.append(labels)
            with torch.no_grad():
                images = sanitizer_model(images)
                images = images.cpu()
            torch.cuda.empty_cache() 
            sanitized_images[(sanitizer_model.counter - 1) * args.batch_size: min(len(train_dataset), sanitizer_model.counter * args.batch_size)] = (255 * images.numpy().transpose(0, 2, 3, 1)).astype(np.uint8)

        targets = torch.cat(targets).numpy()
        save_lmdb(args, data=sanitized_images, targets=targets)

    elif args.domain == 'webface':

        sanitized_images = np.zeros((len(train_dataset), 112, 112, 3))
        sanitizer_model.reset_counter()
        targets = []

        for images, labels in pbar:
            
            images = images.cuda()
            targets.append(labels.numpy())
            with torch.no_grad():
                images = sanitizer_model(images)
                images = images.cpu()

            torch.cuda.empty_cache() 
            sanitized_images[(sanitizer_model.counter - 1) * args.batch_size: min(len(train_dataset), sanitizer_model.counter * args.batch_size)] = (255 * images.numpy().transpose(0, 2, 3, 1)).astype(np.uint8)

        diff = args.config.split('.')[0]
        np.savez(os.path.join(args.save_path, f'{args.unlearnable_alg}_{args.domain.upper()}_Sanitized_{args.t}_{diff}.npz'), data=sanitized_images, targets=np.array(targets))


if __name__ == '__main__':

    args, config = parse_args_and_config()
    log_dir = os.path.join(args.image_folder, 'resnet18', args.unlearnable_alg, 'seed' + str(args.seed), 'data' + str(args.data_seed))
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir

    sanitize(args, config)