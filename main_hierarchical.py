from __future__ import print_function

import os
import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from networks.resnet_big import HierarchicalSupConResNet
from losses import HierarchySupConLoss
from utils.debug_utils import check_tensor, check_gradients
from util import set_optimizer, save_model

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100'], help='dataset')
    parser.add_argument('--feat_dim', type=int, default=128,
                        help='feature dimension')
    parser.add_argument('--level_weights', type=str, default='0.4,0.6',
                        help='weights for hierarchical levels')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/'
    
    
    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * 0.1
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if not os.path.isdir(opt.model_path):
        os.makedirs(opt.model_path)

    return opt

def set_loader(opt):
    # CIFAR100 mean and std
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    normalize = transforms.Normalize(mean=mean, std=std)

    # Define data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    # Load CIFAR100 dataset
    train_dataset = CIFAR100Hierarchy(root=opt.data_folder,
                                    transform=TwoCropTransform(train_transform),
                                    download=True)

    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader

def set_model(opt):
    # Define model with hierarchical outputs at layers 2 and 4
    model = HierarchicalSupConResNet(
        name=opt.model,
        head='mlp',
        feat_dim=opt.feat_dim,
        is_output_layer=[False, True, False, True],
    )
    
    # Parse level weights from string
    level_weights = [float(w) for w in opt.level_weights.split(',')]
    
    # Define loss with weights for each level
    criterion = HierarchySupConLoss(
        level_weights=level_weights,
        temperature=opt.temp,
        contrast_mode='all',
    )

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

def train(train_loader, model, criterion, optimizer, epoch, opt):
    """Train for one epoch"""
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        # Unpack hierarchical labels
        superclass_labels, class_labels = labels
        
        # Handle the two augmented views
        images = torch.cat([images[0], images[1]], dim=0)  # [2*B, C, H, W]
        # Duplicate labels for both views
        superclass_labels = superclass_labels.repeat(2)  # [2*B]
        class_labels = class_labels.repeat(2)  # [2*B]
        
        # Stack labels for hierarchical loss
        labels = torch.stack([superclass_labels, class_labels], dim=1)  # [2*B, 2]
        
        # Move to GPU
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        
        # Forward pass
        features = model(images)  # [2*B, num_levels, feat_dim]
        
        # Split features for contrastive loss
        bsz = labels.shape[0] // 2
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        # Reshape for contrastive loss: [B, num_levels, num_views=2, feat_dim]
        features = torch.stack([f1, f2], dim=2)
        
        # Get labels for first half (since we duplicated them)
        labels = labels[:bsz]
        
        # Compute loss
        loss = criterion(features, labels)
        losses.update(loss.item(), bsz)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print progress
        if (idx + 1) % opt.print_freq == 0:
            print(f'Train: [{epoch}][{idx + 1}/{len(train_loader)}]\t'
                  f'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'loss {loss.item():.3f} ({losses.avg:.3f})')
            sys.stdout.flush()
    
    return losses.avg

def main(opt=None):
    sys.argv = ['', '--dataset', 'cifar100', '--model', 'resnet18', '--learning_rate', '0.5', '--batch_size', '1024', '--epochs', '200']
    if opt is None:
        opt = parse_option()

    # Get data loader
    train_loader = set_loader(opt)
    
    # Build model and criterion
    model, criterion = set_model(opt)
    
    # Build optimizer
    optimizer = set_optimizer(opt, model)
    
    # Training loop
    for epoch in range(1, opt.epochs + 1):
        # Train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}, loss {:.3f}'.format(
            epoch, time2 - time1, loss))

        # Save model
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.model_path, f'ckpt_epoch_{epoch}.pth')
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(opt.model_path, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

class CIFAR100Hierarchy(datasets.CIFAR100):
    """CIFAR100 dataset with hierarchical labels"""
    
    def __init__(self, root, transform=None, download=False):
        super().__init__(root=root, transform=transform, download=download)
        
        # Define the mapping of fine labels to coarse labels (20 superclasses)
        self.coarse_labels = torch.tensor([
            4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
            3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
            6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
            0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
            5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
            16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
            10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
            2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
            16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
            18, 1, 2, 15, 6, 0, 17, 8, 14, 13
        ])
        
    def __getitem__(self, index):
        img, fine_label = super().__getitem__(index)
        coarse_label = self.coarse_labels[fine_label]
        return img, (coarse_label, fine_label)

if __name__ == '__main__':
    main() 