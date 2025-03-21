from __future__ import print_function

import sys
import argparse
import time
import math
import os

import torch
import torch.backends.cudnn as cudnn

from main_ce import set_loader as set_loader_ce
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.resnet_big import HierarchicalSupConResNet, LinearClassifier
from torchvision import transforms, datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


class CIFAR100Hierarchy(datasets.CIFAR100):
    """CIFAR100 dataset with hierarchical labels"""
    
    def __init__(self, root, transform=None, train=True, download=False):
        super().__init__(root=root, transform=transform, train=train, download=download)
        
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


def set_loader(opt):
    """Wrapper around main_ce.set_loader that uses CIFAR100Hierarchy"""
    # Get the transforms and parameters from main_ce's set_loader
    train_loader, val_loader = set_loader_ce(opt)
    
    # Get the transforms from the existing loaders
    train_transform = train_loader.dataset.transform
    val_transform = val_loader.dataset.transform
    
    # Create new datasets with hierarchical labels
    train_dataset = CIFAR100Hierarchy(root=opt.data_folder,
                                    transform=train_transform,
                                    train=True,  # Explicitly set train=True
                                    download=True)
    val_dataset = CIFAR100Hierarchy(root=opt.data_folder,
                                  transform=val_transform,
                                  train=False)
    
    # Create new data loaders with the hierarchical datasets
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)
    
    return train_loader, val_loader


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    # Only support CIFAR100 for hierarchical learning
    opt.n_cls = 100
    opt.n_superclass = 20

    return opt


def set_model(opt):
    model = HierarchicalSupConResNet(
        name=opt.model,
        head='mlp',
        feat_dim=128,  # Set to 128 to match checkpoint
        is_output_layer=[False, True, False, True],
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Get the model's output dimensions based on architecture
    if opt.model in ['resnet18', 'resnet34']:
        early_dim = 128  # 128 for layer2 (expansion=1)
        deep_dim = 512   # 512 for layer4 (expansion=1)
    else:  # resnet50, resnet101
        early_dim = 512  # 512 for layer2 (expansion=4)
        deep_dim = 2048  # 2048 for layer4 (expansion=4)
    # early_dim = deep_dim
    concat_dim = early_dim + deep_dim

    # Five classifiers:
    # 1. Superclass classifier using early features (128-dim for ResNet18/34, 512-dim for ResNet50/101)
    # 2. Superclass classifier using deep features (512-dim for ResNet18/34, 2048-dim for ResNet50/101)
    # 3. Superclass classifier using concatenated features (640-dim for ResNet18/34, 2560-dim for ResNet50/101)
    # 4. Fine-grained classifier from deep features (512-dim for ResNet18/34, 2048-dim for ResNet50/101)
    # 5. Fine-grained classifier from concatenated features (640-dim for ResNet18/34, 2560-dim for ResNet50/101)
    
    # Since is_output_layer=[False, False, False, True], there is only one output layer
    # Set early_dim equal to deep_dim
    superclass_classifier = LinearClassifier(name=opt.model, num_classes=opt.n_superclass, feat_dim=deep_dim)
    class_classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls, feat_dim=deep_dim)
    concat_classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls, feat_dim=concat_dim)
    # Add a new classifier for predicting superclasses using early features
    early_superclass_classifier = LinearClassifier(name=opt.model, num_classes=opt.n_superclass, feat_dim=early_dim)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        superclass_classifier = superclass_classifier.cuda()
        class_classifier = class_classifier.cuda()
        concat_classifier = concat_classifier.cuda()
        early_superclass_classifier = early_superclass_classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError('This code requires GPU')

    return model, (superclass_classifier, class_classifier, concat_classifier, early_superclass_classifier), criterion


def train(train_loader, model, classifiers, criterion, optimizers, epoch, opt):
    """one epoch training"""
    model.eval()
    superclass_classifier, class_classifier, concat_classifier, early_superclass_classifier = classifiers
    superclass_optimizer, class_optimizer, concat_optimizer, early_superclass_optimizer = optimizers
    
    superclass_classifier.train()
    class_classifier.train()
    concat_classifier.train()
    early_superclass_classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    superclass_losses = AverageMeter()
    class_losses = AverageMeter()
    concat_losses = AverageMeter()
    early_superclass_losses = AverageMeter()
    superclass_top1 = AverageMeter()
    class_top1 = AverageMeter()
    concat_top1 = AverageMeter()
    early_superclass_top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        superclass_labels, class_labels = labels
        superclass_labels = superclass_labels.cuda(non_blocking=True)
        class_labels = class_labels.cuda(non_blocking=True)
        bsz = class_labels.shape[0]
        
        # Print shapes for debugging
        # print('Images shape:', images.shape)
        # print('Labels shape:', len(labels))
        # print('Batch size:', bsz)
        # print('Superclass labels shape:', superclass_labels.shape)
        # print('Class labels shape:', class_labels.shape)

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), superclass_optimizer)
        warmup_learning_rate(opt, epoch, idx, len(train_loader), class_optimizer)
        warmup_learning_rate(opt, epoch, idx, len(train_loader), concat_optimizer)
        warmup_learning_rate(opt, epoch, idx, len(train_loader), early_superclass_optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)  # List of features from different levels

        # Superclass classification from level 1 features (64-dim)
        superclass_output = superclass_classifier(features[-1].detach())
        superclass_loss = criterion(superclass_output, superclass_labels)

        # Class classification from level 2 features (128-dim)
        class_output = class_classifier(features[-1].detach())
        class_loss = criterion(class_output, class_labels)

        # Class classification from concatenated features
        concat_features = torch.cat([features[0].detach(), features[-1].detach()], dim=1)  # Concatenate level 1 and 2
        concat_output = concat_classifier(concat_features)
        concat_loss = criterion(concat_output, class_labels)

        # Early superclass classification from level 1 features (64-dim)
        early_superclass_output = early_superclass_classifier(features[0].detach())
        early_superclass_loss = criterion(early_superclass_output, superclass_labels)

        # update metric
        superclass_losses.update(superclass_loss.item(), bsz)
        class_losses.update(class_loss.item(), bsz)
        concat_losses.update(concat_loss.item(), bsz)
        early_superclass_losses.update(early_superclass_loss.item(), bsz)
        
        # Calculate accuracies
        superclass_acc1 = accuracy(superclass_output, superclass_labels, topk=(1,))  # Only top-1 for superclass (20 classes)
        class_acc1 = accuracy(class_output, class_labels, topk=(1,))  # Top-1 for fine classes (100 classes)
        concat_acc1 = accuracy(concat_output, class_labels, topk=(1,))  # Top-1 for concatenated
        early_superclass_acc1 = accuracy(early_superclass_output, superclass_labels, topk=(1,))  # Top-1 for early superclass (20 classes)
        
        superclass_top1.update(superclass_acc1[0].item(), bsz)
        class_top1.update(class_acc1[0].item(), bsz)
        concat_top1.update(concat_acc1[0].item(), bsz)
        early_superclass_top1.update(early_superclass_acc1[0].item(), bsz)

        # SGD
        superclass_optimizer.zero_grad()
        superclass_loss.backward()
        superclass_optimizer.step()

        class_optimizer.zero_grad()
        class_loss.backward()
        class_optimizer.step()

        concat_optimizer.zero_grad()
        concat_loss.backward()
        concat_optimizer.step()

        early_superclass_optimizer.zero_grad()
        early_superclass_loss.backward()
        early_superclass_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'S-loss {sloss.val:.3f} ({sloss.avg:.3f})\t'
                  'C-loss {closs.val:.3f} ({closs.avg:.3f})\t'
                  'CC-loss {ccloss.val:.3f} ({ccloss.avg:.3f})\t'
                  'ES-loss {esloss.val:.3f} ({esloss.avg:.3f})\t'
                  'S-Acc@1 {stop1.val:.3f} ({stop1.avg:.3f})\t'
                  'C-Acc@1 {ctop1.val:.3f} ({ctop1.avg:.3f})\t'
                  'CC-Acc@1 {cctop1.val:.3f} ({cctop1.avg:.3f})\t'
                  'ES-Acc@1 {estop1.val:.3f} ({estop1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, sloss=superclass_losses, closs=class_losses,
                   ccloss=concat_losses, esloss=early_superclass_losses, stop1=superclass_top1,
                   ctop1=class_top1, cctop1=concat_top1, estop1=early_superclass_top1))
            sys.stdout.flush()

    return (superclass_losses.avg, class_losses.avg, concat_losses.avg, early_superclass_losses.avg), \
           (superclass_top1.avg, class_top1.avg, concat_top1.avg, early_superclass_top1.avg)


def validate(val_loader, model, classifiers, criterion, opt):
    """validation"""
    model.eval()
    superclass_classifier, class_classifier, concat_classifier, early_superclass_classifier = classifiers
    superclass_classifier.eval()
    class_classifier.eval()
    concat_classifier.eval()
    early_superclass_classifier.eval()

    batch_time = AverageMeter()
    superclass_losses = AverageMeter()
    class_losses = AverageMeter()
    concat_losses = AverageMeter()
    early_superclass_losses = AverageMeter()
    superclass_top1 = AverageMeter()
    class_top1 = AverageMeter()
    concat_top1 = AverageMeter()
    early_superclass_top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            superclass_labels, class_labels = labels
            superclass_labels = superclass_labels.cuda()
            class_labels = class_labels.cuda()
            bsz = class_labels.shape[0]

            # forward
            features = model.encoder(images)  # List of features from different levels

            # Superclass classification from level 1 features (64-dim)
            superclass_output = superclass_classifier(features[-1])
            superclass_loss = criterion(superclass_output, superclass_labels)

            # Class classification from level 2 features (128-dim)
            class_output = class_classifier(features[-1])
            class_loss = criterion(class_output, class_labels)

            # Class classification from concatenated features
            concat_features = torch.cat([features[0], features[-1]], dim=1)  # Concatenate level 1 and 2
            concat_output = concat_classifier(concat_features)
            concat_loss = criterion(concat_output, class_labels)

            # Early superclass classification from level 1 features (64-dim)
            early_superclass_output = early_superclass_classifier(features[0])
            early_superclass_loss = criterion(early_superclass_output, superclass_labels)

            # update metric
            superclass_losses.update(superclass_loss.item(), bsz)
            class_losses.update(class_loss.item(), bsz)
            concat_losses.update(concat_loss.item(), bsz)
            early_superclass_losses.update(early_superclass_loss.item(), bsz)
            
            # Calculate accuracies
            superclass_acc1 = accuracy(superclass_output, superclass_labels, topk=(1,))  # Only top-1 for superclass (20 classes)
            class_acc1 = accuracy(class_output, class_labels, topk=(1,))  # Top-1 for fine classes (100 classes)
            concat_acc1 = accuracy(concat_output, class_labels, topk=(1,))  # Top-1 for concatenated
            early_superclass_acc1 = accuracy(early_superclass_output, superclass_labels, topk=(1,))  # Top-1 for early superclass (20 classes)
            
            superclass_top1.update(superclass_acc1[0].item(), bsz)
            class_top1.update(class_acc1[0].item(), bsz)
            concat_top1.update(concat_acc1[0].item(), bsz)
            early_superclass_top1.update(early_superclass_acc1[0].item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'S-Loss {sloss.val:.4f} ({sloss.avg:.4f})\t'
                      'C-Loss {closs.val:.4f} ({closs.avg:.4f})\t'
                      'CC-Loss {ccloss.val:.4f} ({ccloss.avg:.4f})\t'
                      'ES-Loss {esloss.val:.4f} ({esloss.avg:.4f})\t'
                      'S-Acc@1 {stop1.val:.3f} ({stop1.avg:.3f})\t'
                      'C-Acc@1 {ctop1.val:.3f} ({ctop1.avg:.3f})\t'
                      'CC-Acc@1 {cctop1.val:.3f} ({cctop1.avg:.3f})\t'
                      'ES-Acc@1 {estop1.val:.3f} ({estop1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       sloss=superclass_losses, closs=class_losses,
                       ccloss=concat_losses, esloss=early_superclass_losses, stop1=superclass_top1,
                       ctop1=class_top1, cctop1=concat_top1, estop1=early_superclass_top1))

    print(' * Superclass Acc@1 {stop1.avg:.3f}'.format(stop1=superclass_top1))
    print(' * Class Acc@1 {ctop1.avg:.3f}'.format(ctop1=class_top1))
    print(' * Concat Class Acc@1 {cctop1.avg:.3f}'.format(cctop1=concat_top1))
    print(' * Early Superclass Acc@1 {estop1.avg:.3f}'.format(estop1=early_superclass_top1))
    return (superclass_losses.avg, class_losses.avg, concat_losses.avg, early_superclass_losses.avg), \
           (superclass_top1.avg, class_top1.avg, concat_top1.avg, early_superclass_top1.avg)


def plot_metrics(df, epoch):
    """Plot loss and accuracy curves with enhanced visualizations"""
    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Linear scale plots
    plt.figure(figsize=(20, 15), dpi=300)
    
    # Plot losses - linear scale
    plt.subplot(2, 3, 1)
    plt.plot(df['epoch'], df['superclass_loss'], 'r-', linewidth=2, label='Deep Superclass')
    plt.plot(df['epoch'], df['early_superclass_loss'], 'm-', linewidth=2, label='Early Superclass')
    plt.plot(df['epoch'], df['class_loss'], 'g-', linewidth=2, label='Class')
    plt.plot(df['epoch'], df['concat_loss'], 'b-', linewidth=2, label='Concat')
    plt.title('Test Loss vs Epoch (Linear Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot accuracies - linear scale
    plt.subplot(2, 3, 2)
    plt.plot(df['epoch'], df['superclass_acc'], 'r-', linewidth=2, label='Deep Superclass')
    plt.plot(df['epoch'], df['early_superclass_acc'], 'm-', linewidth=2, label='Early Superclass')
    plt.plot(df['epoch'], df['class_acc'], 'g-', linewidth=2, label='Class')
    plt.plot(df['epoch'], df['concat_acc'], 'b-', linewidth=2, label='Concat')
    plt.title('Test Accuracy vs Epoch (Linear Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot error rates - linear scale
    plt.subplot(2, 3, 3)
    plt.plot(df['epoch'], 100 - df['superclass_acc'], 'r-', linewidth=2, label='Deep Superclass Error')
    plt.plot(df['epoch'], 100 - df['early_superclass_acc'], 'm-', linewidth=2, label='Early Superclass Error')
    plt.plot(df['epoch'], 100 - df['class_acc'], 'g-', linewidth=2, label='Class Error')
    plt.plot(df['epoch'], 100 - df['concat_acc'], 'b-', linewidth=2, label='Concat Error')
    plt.title('Error Rate vs Epoch (Linear Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Error Rate (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot losses - logarithmic scale
    plt.subplot(2, 3, 4)
    plt.semilogy(df['epoch'], df['superclass_loss'], 'r-', linewidth=2, label='Deep Superclass')
    plt.semilogy(df['epoch'], df['early_superclass_loss'], 'm-', linewidth=2, label='Early Superclass')
    plt.semilogy(df['epoch'], df['class_loss'], 'g-', linewidth=2, label='Class')
    plt.semilogy(df['epoch'], df['concat_loss'], 'b-', linewidth=2, label='Concat')
    plt.title('Test Loss vs Epoch (Log Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot superclass accuracy comparison
    plt.subplot(2, 3, 5)
    plt.plot(df['epoch'], df['early_superclass_acc'] - df['superclass_acc'], 'm-', linewidth=2, 
             label='Early vs Deep Superclass')
    plt.title('Early vs Deep Superclass Accuracy Difference', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy Difference (%)', fontsize=12)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot error rates - logarithmic scale
    plt.subplot(2, 3, 6)
    plt.semilogy(df['epoch'], 100 - df['superclass_acc'], 'r-', linewidth=2, label='Deep Superclass Error')
    plt.semilogy(df['epoch'], 100 - df['early_superclass_acc'], 'm-', linewidth=2, label='Early Superclass Error')
    plt.semilogy(df['epoch'], 100 - df['class_acc'], 'g-', linewidth=2, label='Class Error')
    plt.semilogy(df['epoch'], 100 - df['concat_acc'], 'b-', linewidth=2, label='Concat Error')
    plt.title('Error Rate vs Epoch (Log Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Error Rate % (log scale)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'plots/hierarchical_metrics_epoch_{epoch}.png')
    plt.savefig(f'plots/hierarchical_metrics_latest.png')  # Always save the latest version
    
    # Close the figure to free memory
    plt.close()
    
    print(f"Hierarchical plots saved to 'plots/hierarchical_metrics_epoch_{epoch}.png'")

def visualize_predictions(val_loader, model, classifiers, epoch, num_images=4):
    """Visualize predictions on random test images"""
    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    model.eval()
    superclass_classifier, class_classifier, concat_classifier, early_superclass_classifier = classifiers
    superclass_classifier.eval()
    class_classifier.eval()
    concat_classifier.eval()
    early_superclass_classifier.eval()
    
    # CIFAR-100 class names
    superclass_names = [
        'aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
        'household electrical devices', 'household furniture', 'insects', 'large carnivores',
        'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
        'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles',
        'small mammals', 'trees', 'vehicles 1', 'vehicles 2'
    ]
    
    class_names = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
        'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
        'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
        'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]
    
    # Get a batch of images
    images, (superclass_labels, class_labels) = next(iter(val_loader))
    
    # Select random indices
    batch_size = images.shape[0]
    indices = random.sample(range(batch_size), min(num_images, batch_size))
    
    # Create figure with a single row
    fig, axes = plt.subplots(1, num_images, figsize=(16, 5), dpi=200)
    
    # CIFAR100 mean and std for denormalization
    mean = torch.tensor((0.5071, 0.4867, 0.4408))
    std = torch.tensor((0.2675, 0.2565, 0.2761))
    
    with torch.no_grad():
        # Get features and predictions
        features = model.encoder(images.cuda())
        
        # Get predictions from each classifier
        superclass_output = superclass_classifier(features[-1])
        class_output = class_classifier(features[-1])
        concat_features = torch.cat([features[0], features[-1]], dim=1)
        concat_output = concat_classifier(concat_features)
        early_superclass_output = early_superclass_classifier(features[0])
        
        # Get predicted classes
        _, superclass_preds = superclass_output.cpu().max(1)
        _, class_preds = class_output.cpu().max(1)
        _, concat_preds = concat_output.cpu().max(1)
        _, early_superclass_preds = early_superclass_output.cpu().max(1)
    
    # Create color-coding for predictions
    def get_color(pred, truth):
        return 'green' if pred == truth else 'red'
    
    for idx, i in enumerate(indices):
        # Denormalize image
        img = images[i].cpu()
        img = img * std[:, None, None] + mean[:, None, None]
        img = torch.clamp(img, 0, 1)
        
        # Plot image
        axes[idx].imshow(img.permute(1, 2, 0))
        axes[idx].axis('off')
        
        # Get class names
        true_superclass_name = superclass_names[superclass_labels[i]]
        true_class_name = class_names[class_labels[i]]
        pred_superclass_name = superclass_names[superclass_preds[i]]
        pred_class_name = class_names[class_preds[i]]
        pred_concat_class_name = class_names[concat_preds[i]]
        pred_early_superclass_name = superclass_names[early_superclass_preds[i]]
        
        # Add predictions as title with color-coding
        superclass_color = get_color(superclass_preds[i], superclass_labels[i])
        class_color = get_color(class_preds[i], class_labels[i])
        concat_color = get_color(concat_preds[i], class_labels[i])
        early_superclass_color = get_color(early_superclass_preds[i], superclass_labels[i])
        
        title = f'Ground Truth:\n' + \
                f'Super: {true_superclass_name}\n' + \
                f'Class: {true_class_name}\n\n' + \
                f'Predictions:\n' + \
                f'Super: {pred_superclass_name} [{"✓" if superclass_preds[i] == superclass_labels[i] else "✗"}]\n' + \
                f'Class: {pred_class_name} [{"✓" if class_preds[i] == class_labels[i] else "✗"}]\n' + \
                f'Concat: {pred_concat_class_name} [{"✓" if concat_preds[i] == class_labels[i] else "✗"}]\n' + \
                f'ESuper: {pred_early_superclass_name} [{"✓" if early_superclass_preds[i] == superclass_labels[i] else "✗"}]'
                
        axes[idx].set_title(title, fontsize=9)
    
    plt.suptitle(f'Predictions at Epoch {epoch}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f'plots/hierarchical_predictions_epoch_{epoch}.png')
    plt.savefig(f'plots/hierarchical_predictions_latest.png')  # Always save the latest version
    plt.close()
    
    print(f"Prediction visualizations saved to 'plots/hierarchical_predictions_epoch_{epoch}.png'")

def main(opt=None):
    sys.argv = ['', '--dataset', 'cifar100', '--model', 'resnet50', '--learning_rate', '0.1', '--batch_size', '512', '--epochs', '300', '--ckpt', './save/ckpt_epoch_200.pth']
    if opt is None:
        opt = parse_option()
    print(opt)
    best_acc = 0
    
    # Create dataframe to store metrics
    metrics_df = pd.DataFrame(columns=[
        'epoch', 'superclass_loss', 'class_loss', 'concat_loss', 'early_superclass_loss',
        'superclass_acc', 'class_acc', 'concat_acc', 'early_superclass_acc'
    ])
    
    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifiers, criterion = set_model(opt)

    # build optimizer
    superclass_optimizer = set_optimizer(opt, classifiers[0])
    class_optimizer = set_optimizer(opt, classifiers[1])
    concat_optimizer = set_optimizer(opt, classifiers[2])
    early_superclass_optimizer = set_optimizer(opt, classifiers[3])
    optimizers = (superclass_optimizer, class_optimizer, concat_optimizer, early_superclass_optimizer)
    
    # Get initial test metrics
    val_losses, val_accs = validate(val_loader, model, classifiers, criterion, opt)
    
    # Ensure tensor values are detached from CUDA and converted to float
    val_losses = [loss.detach().cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in val_losses]
    val_accs = [acc.detach().cpu().item() if isinstance(acc, torch.Tensor) else acc for acc in val_accs]
    
    new_row = pd.DataFrame([{
        'epoch': 0,
        'superclass_loss': val_losses[0],
        'class_loss': val_losses[1],
        'concat_loss': val_losses[2],
        'early_superclass_loss': val_losses[3],
        'superclass_acc': val_accs[0],
        'class_acc': val_accs[1],
        'concat_acc': val_accs[2],
        'early_superclass_acc': val_accs[3]
    }])
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
    
    # Plot initial metrics
    plot_metrics(metrics_df, 0)
    visualize_predictions(val_loader, model, classifiers, 0)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, superclass_optimizer, epoch)
        adjust_learning_rate(opt, class_optimizer, epoch)
        adjust_learning_rate(opt, concat_optimizer, epoch)
        adjust_learning_rate(opt, early_superclass_optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        losses, accs = train(train_loader, model, classifiers, criterion,
                          optimizers, epoch, opt)
        time2 = time.time()
        
        # Ensure tensor values are detached from CUDA and converted to float
        losses = [loss.detach().cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in losses]
        accs = [acc.detach().cpu().item() if isinstance(acc, torch.Tensor) else acc for acc in accs]
        
        print('Train epoch {}, total time {:.2f}, superclass loss {:.3f}, class loss {:.3f}, concat loss {:.3f}, early superclass loss {:.3f}, '
              'superclass accuracy {:.3f}, class accuracy {:.3f}, concat accuracy {:.3f}, early superclass accuracy {:.3f}'.format(
               epoch, time2 - time1, losses[0], losses[1], losses[2], losses[3], accs[0], accs[1], accs[2], accs[3]))

        # eval for one epoch
        val_losses, val_accs = validate(val_loader, model, classifiers, criterion, opt)
        
        # Ensure tensor values are detached from CUDA and converted to float
        val_losses = [loss.detach().cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in val_losses]
        val_accs = [acc.detach().cpu().item() if isinstance(acc, torch.Tensor) else acc for acc in val_accs]
        
        if val_accs[2] > best_acc:
            best_acc = val_accs[2]
        
        # Store metrics
        new_row = pd.DataFrame([{
            'epoch': epoch,
            'superclass_loss': val_losses[0],
            'class_loss': val_losses[1],
            'concat_loss': val_losses[2],
            'early_superclass_loss': val_losses[3],
            'superclass_acc': val_accs[0],
            'class_acc': val_accs[1],
            'concat_acc': val_accs[2],
            'early_superclass_acc': val_accs[3]
        }])
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
        
        # Plot metrics every 20 epochs
        if epoch % 20 == 0 or epoch == opt.epochs:
            print(f"\nGenerating plots for epoch {epoch}")
            plot_metrics(metrics_df, epoch)
            visualize_predictions(val_loader, model, classifiers, epoch)
            print(f"Plots saved for epoch {epoch}\n")
            
        # Save metrics to csv
        metrics_df.to_csv('training_metrics.csv', index=False)
            
    print('best accuracy: {:.3f}'.format(best_acc))
    return best_acc


if __name__ == '__main__':
    main()
