from __future__ import print_function

import sys
import argparse
import time
import math

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
                                    download=True)
    val_dataset = CIFAR100Hierarchy(root=opt.data_folder,
                                  transform=val_transform,
                                  download=True)
    
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
    parser.add_argument('--num_workers', type=int, default=12,
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
    parser.add_argument('--model', type=str, default='resnet18')
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
        is_output_layer=[False, True, False, True]  # Enable second output layer to match checkpoint
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Three classifiers:
    # 1. Superclass classifier
    # 2. Fine-grained classifier from fine features
    # 3. Fine-grained classifier from concatenated features
    
    
    superclass_classifier = LinearClassifier(name=opt.model, num_classes=opt.n_superclass, feat_dim=128)
    class_classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls, feat_dim=512)
    concat_classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls, feat_dim=640)  # 128*2 features

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
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError('This code requires GPU')

    return model, (superclass_classifier, class_classifier, concat_classifier), criterion


def train(train_loader, model, classifiers, criterion, optimizers, epoch, opt):
    """one epoch training"""
    model.eval()
    superclass_classifier, class_classifier, concat_classifier = classifiers
    superclass_optimizer, class_optimizer, concat_optimizer = optimizers
    
    superclass_classifier.train()
    class_classifier.train()
    concat_classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    superclass_losses = AverageMeter()
    class_losses = AverageMeter()
    concat_losses = AverageMeter()
    superclass_top1 = AverageMeter()
    class_top1 = AverageMeter()
    concat_top1 = AverageMeter()

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

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)  # List of features from different levels

        # Superclass classification from level 1 features (64-dim)
        superclass_output = superclass_classifier(features[0].detach())
        superclass_loss = criterion(superclass_output, superclass_labels)

        # Class classification from level 2 features (128-dim)
        class_output = class_classifier(features[1].detach())
        class_loss = criterion(class_output, class_labels)

        # Class classification from concatenated features
        concat_features = torch.cat([features[0].detach(), features[1].detach()], dim=1)  # Concatenate level 1 and 2
        concat_output = concat_classifier(concat_features)
        concat_loss = criterion(concat_output, class_labels)

        # update metric
        superclass_losses.update(superclass_loss.item(), bsz)
        class_losses.update(class_loss.item(), bsz)
        concat_losses.update(concat_loss.item(), bsz)
        
        # Calculate accuracies
        superclass_acc1 = accuracy(superclass_output, superclass_labels, topk=(1,))  # Only top-1 for superclass (20 classes)
        class_acc1 = accuracy(class_output, class_labels, topk=(1,))  # Top-1 for fine classes (100 classes)
        concat_acc1 = accuracy(concat_output, class_labels, topk=(1,))  # Top-1 for concatenated
        
        superclass_top1.update(superclass_acc1[0].item(), bsz)
        class_top1.update(class_acc1[0].item(), bsz)
        concat_top1.update(concat_acc1[0].item(), bsz)

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
                  'S-Acc@1 {stop1.val:.3f} ({stop1.avg:.3f})\t'
                  'C-Acc@1 {ctop1.val:.3f} ({ctop1.avg:.3f})\t'
                  'CC-Acc@1 {cctop1.val:.3f} ({cctop1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, sloss=superclass_losses, closs=class_losses,
                   ccloss=concat_losses, stop1=superclass_top1, ctop1=class_top1,
                   cctop1=concat_top1))
            sys.stdout.flush()

    return (superclass_losses.avg, class_losses.avg, concat_losses.avg), \
           (superclass_top1.avg, class_top1.avg, concat_top1.avg)


def validate(val_loader, model, classifiers, criterion, opt):
    """validation"""
    model.eval()
    superclass_classifier, class_classifier, concat_classifier = classifiers
    superclass_classifier.eval()
    class_classifier.eval()
    concat_classifier.eval()

    batch_time = AverageMeter()
    superclass_losses = AverageMeter()
    class_losses = AverageMeter()
    concat_losses = AverageMeter()
    superclass_top1 = AverageMeter()
    class_top1 = AverageMeter()
    concat_top1 = AverageMeter()

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
            superclass_output = superclass_classifier(features[0])
            superclass_loss = criterion(superclass_output, superclass_labels)

            # Class classification from level 2 features (128-dim)
            class_output = class_classifier(features[1])
            class_loss = criterion(class_output, class_labels)

            # Class classification from concatenated features
            concat_features = torch.cat([features[0], features[1]], dim=1)  # Concatenate level 1 and 2
            concat_output = concat_classifier(concat_features)
            concat_loss = criterion(concat_output, class_labels)

            # update metric
            superclass_losses.update(superclass_loss.item(), bsz)
            class_losses.update(class_loss.item(), bsz)
            concat_losses.update(concat_loss.item(), bsz)
            
            # Calculate accuracies
            superclass_acc1 = accuracy(superclass_output, superclass_labels, topk=(1,))  # Only top-1 for superclass (20 classes)
            class_acc1 = accuracy(class_output, class_labels, topk=(1,))  # Top-1 for fine classes (100 classes)
            concat_acc1 = accuracy(concat_output, class_labels, topk=(1,))  # Top-1 for concatenated
            
            superclass_top1.update(superclass_acc1[0].item(), bsz)
            class_top1.update(class_acc1[0].item(), bsz)
            concat_top1.update(concat_acc1[0].item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'S-Loss {sloss.val:.4f} ({sloss.avg:.4f})\t'
                      'C-Loss {closs.val:.4f} ({closs.avg:.4f})\t'
                      'CC-Loss {ccloss.val:.4f} ({ccloss.avg:.4f})\t'
                      'S-Acc@1 {stop1.val:.3f} ({stop1.avg:.3f})\t'
                      'C-Acc@1 {ctop1.val:.3f} ({ctop1.avg:.3f})\t'
                      'CC-Acc@1 {cctop1.val:.3f} ({cctop1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       sloss=superclass_losses, closs=class_losses,
                       ccloss=concat_losses, stop1=superclass_top1,
                       ctop1=class_top1, cctop1=concat_top1))

    print(' * Superclass Acc@1 {stop1.avg:.3f}'.format(stop1=superclass_top1))
    print(' * Class Acc@1 {ctop1.avg:.3f}'.format(ctop1=class_top1))
    print(' * Concat Class Acc@1 {cctop1.avg:.3f}'.format(cctop1=concat_top1))
    return (superclass_losses.avg, class_losses.avg, concat_losses.avg), \
           (superclass_top1.avg, class_top1.avg, concat_top1.avg)


def plot_metrics(df, epoch):
    """Plot loss and accuracy curves"""
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['superclass_loss'], label='Superclass')
    plt.plot(df['epoch'], df['class_loss'], label='Class')
    plt.plot(df['epoch'], df['concat_loss'], label='Concat')
    plt.title('Test Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['superclass_acc'], label='Superclass')
    plt.plot(df['epoch'], df['class_acc'], label='Class')
    plt.plot(df['epoch'], df['concat_acc'], label='Concat')
    plt.title('Test Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'metrics_epoch_{epoch}.png')
    plt.close()

def visualize_predictions(val_loader, model, classifiers, num_images=4):
    """Visualize predictions on random test images"""
    model.eval()
    superclass_classifier, class_classifier, concat_classifier = classifiers
    superclass_classifier.eval()
    class_classifier.eval()
    concat_classifier.eval()
    
    # Get a batch of images
    images, (superclass_labels, class_labels) = next(iter(val_loader))
    
    # Select random indices
    batch_size = images.shape[0]
    indices = random.sample(range(batch_size), min(num_images, batch_size))
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    # CIFAR100 mean and std for denormalization
    mean = torch.tensor((0.5071, 0.4867, 0.4408))
    std = torch.tensor((0.2675, 0.2565, 0.2761))
    
    with torch.no_grad():
        # Get features and predictions
        features = model.encoder(images.cuda())
        
        # Get predictions from each classifier
        superclass_output = superclass_classifier(features[0])
        class_output = class_classifier(features[1])
        concat_features = torch.cat([features[0], features[1]], dim=1)
        concat_output = concat_classifier(concat_features)
        
        # Get predicted classes
        _, superclass_preds = superclass_output.cpu().max(1)
        _, class_preds = class_output.cpu().max(1)
        _, concat_preds = concat_output.cpu().max(1)
    
    for idx, i in enumerate(indices):
        # Denormalize image
        img = images[i].cpu()
        img = img * std[:, None, None] + mean[:, None, None]
        img = torch.clamp(img, 0, 1)
        
        # Plot image
        axes[idx].imshow(img.permute(1, 2, 0))
        axes[idx].axis('off')
        
        # Add predictions as title
        title = f'True: (super={superclass_labels[i]}, class={class_labels[i]})\n'
        title += f'Pred: super={superclass_preds[i]}, class={class_preds[i]}, concat={concat_preds[i]}'
        axes[idx].set_title(title, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'predictions.png')
    plt.close()

def main(opt=None):
    sys.argv = ['', '--dataset', 'cifar100', '--model', 'resnet18', '--learning_rate', '1', '--batch_size', '512', '--epochs', '100', '--ckpt', './save/ckpt_epoch_200.pth']
    if opt is None:
        opt = parse_option()
    print(opt)
    best_acc = 0
    
    # Create dataframe to store metrics
    metrics_df = pd.DataFrame(columns=[
        'epoch', 'superclass_loss', 'class_loss', 'concat_loss',
        'superclass_acc', 'class_acc', 'concat_acc'
    ])
    
    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifiers, criterion = set_model(opt)

    # build optimizer
    superclass_optimizer = set_optimizer(opt, classifiers[0])
    class_optimizer = set_optimizer(opt, classifiers[1])
    concat_optimizer = set_optimizer(opt, classifiers[2])
    optimizers = (superclass_optimizer, class_optimizer, concat_optimizer)
    
    # Get initial test metrics
    val_losses, val_accs = validate(val_loader, model, classifiers, criterion, opt)
    metrics_df = metrics_df.append({
        'epoch': 0,
        'superclass_loss': val_losses[0],
        'class_loss': val_losses[1],
        'concat_loss': val_losses[2],
        'superclass_acc': val_accs[0],
        'class_acc': val_accs[1],
        'concat_acc': val_accs[2]
    }, ignore_index=True)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, superclass_optimizer, epoch)
        adjust_learning_rate(opt, class_optimizer, epoch)
        adjust_learning_rate(opt, concat_optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        losses, accs = train(train_loader, model, classifiers, criterion,
                          optimizers, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, superclass loss {:.3f}, class loss {:.3f}, concat loss {:.3f}, '
              'superclass accuracy {:.3f}, class accuracy {:.3f}, concat accuracy {:.3f}'.format(
               epoch, time2 - time1, losses[0], losses[1], losses[2], accs[0], accs[1], accs[2]))

        # eval for one epoch
        val_losses, val_accs = validate(val_loader, model, classifiers, criterion, opt)
        if val_accs[2] > best_acc:
            best_acc = val_accs[2]
        
        # Store metrics
        metrics_df = metrics_df.append({
            'epoch': epoch,
            'superclass_loss': val_losses[0],
            'class_loss': val_losses[1],
            'concat_loss': val_losses[2],
            'superclass_acc': val_accs[0],
            'class_acc': val_accs[1],
            'concat_acc': val_accs[2]
        }, ignore_index=True)
        
        # Plot metrics every 5 epochs
        if epoch % 5 == 0:
            plot_metrics(metrics_df, epoch)
            visualize_predictions(val_loader, model, classifiers)
            
        # Save metrics to csv
        metrics_df.to_csv('training_metrics.csv', index=False)
            
    print('best accuracy: {:.3f}'.format(best_acc))
    
    # Final plots
    plot_metrics(metrics_df, opt.epochs)
    visualize_predictions(val_loader, model, classifiers)


if __name__ == '__main__':
    main()
