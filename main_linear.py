from __future__ import print_function

import sys
import argparse
import time
import math
import random

import torch
import torch.backends.cudnn as cudnn

from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.resnet_big import SupConResNet, LinearClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


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
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='dataset')

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

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

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
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError('This code requires GPU')

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def visualize_predictions(val_loader, model, classifier, epoch, num_images=4):
    """Visualize predictions on random test images"""
    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    model.eval()
    classifier.eval()
    
    # CIFAR-100 class names (if using CIFAR-100)
    if hasattr(val_loader.dataset, 'classes'):
        class_names = val_loader.dataset.classes
    else:
        # For CIFAR-100
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
    dataiter = iter(val_loader)
    images, labels = next(dataiter)
    
    # Select random indices
    batch_size = images.shape[0]
    indices = random.sample(range(batch_size), min(num_images, batch_size))
    
    # Create figure
    fig, axes = plt.subplots(1, num_images, figsize=(16, 5), dpi=200)
    
    # CIFAR mean and std for denormalization
    mean = torch.tensor((0.5071, 0.4867, 0.4408))
    std = torch.tensor((0.2675, 0.2565, 0.2761))
    
    with torch.no_grad():
        # Get features and predictions
        features = model.encoder(images.cuda())
        outputs = classifier(features)
        
        # Get predicted classes
        _, predicted = outputs.cpu().max(1)
    
    for idx, i in enumerate(indices):
        # Denormalize image
        img = images[i].cpu()
        img = img * std[:, None, None] + mean[:, None, None]
        img = torch.clamp(img, 0, 1)
        
        # Plot image
        axes[idx].imshow(img.permute(1, 2, 0))
        axes[idx].axis('off')
        
        # Get class names
        true_class = class_names[labels[i]]
        pred_class = class_names[predicted[i]]
        
        # Add predictions as title
        title = f'True: {true_class}\nPred: {pred_class}'
        if labels[i] == predicted[i]:
            title += ' ✓'
        else:
            title += ' ✗'
            
        axes[idx].set_title(title, fontsize=10)
    
    plt.suptitle(f'Predictions at Epoch {epoch}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f'plots/predictions_epoch_{epoch}.png')
    plt.savefig(f'plots/predictions_latest.png')  # Always save the latest version
    plt.close()
    
    print(f"Prediction visualizations saved to 'plots/predictions_epoch_{epoch}.png'")


def plot_metrics(df, epoch):
    """
    Plot training and validation metrics with enhanced visualizations.
    
    Args:
        df: DataFrame containing the metrics
        epoch: Current epoch number for saving the plot
    """
    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Set higher DPI for better resolution
    plt.figure(figsize=(20, 15), dpi=300)
    
    # Plot losses - linear scale
    plt.subplot(2, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    plt.title('Loss vs Epoch (Linear Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Plot accuracies - linear scale
    plt.subplot(2, 2, 2)
    plt.plot(df['epoch'], df['train_acc'], 'b-', linewidth=2, label='Train Accuracy')
    plt.plot(df['epoch'], df['val_acc'], 'r-', linewidth=2, label='Validation Accuracy')
    plt.title('Accuracy vs Epoch (Linear Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot losses - logarithmic scale
    plt.subplot(2, 2, 3)
    plt.semilogy(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='Train Loss')
    plt.semilogy(df['epoch'], df['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    plt.title('Loss vs Epoch (Log Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot learning rate
    plt.subplot(2, 2, 4)
    if 'learning_rate' in df.columns:
        plt.plot(df['epoch'], df['learning_rate'], 'g-', linewidth=2)
        plt.title('Learning Rate vs Epoch', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
    else:
        # If learning rate is not tracked, show accuracy in log scale instead
        plt.plot(df['epoch'], 100 - df['train_acc'], 'b-', linewidth=2, label='Train Error (100 - Acc)')
        plt.plot(df['epoch'], 100 - df['val_acc'], 'r-', linewidth=2, label='Validation Error (100 - Acc)')
        plt.title('Error Rate vs Epoch (Log Scale)', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Error Rate % (log scale)', fontsize=12)
        plt.yscale('log')
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'plots/metrics_epoch_{epoch}.png')
    plt.savefig(f'plots/metrics_latest.png')  # Always save the latest version with a consistent name
    
    # Close the figure to free memory
    plt.close()
    
    print(f"Plots saved to 'plots/metrics_epoch_{epoch}.png'")


def main():
    best_acc = 0
    sys.argv = ['', '--dataset', 'cifar100', '--model', 'resnet50', '--learning_rate', '0.1', '--batch_size', '512', '--epochs', '200', '--ckpt', './save/ckpt_epoch_200.pth']
    opt = parse_option()
    
    # Create dataframe to store metrics
    metrics_df = pd.DataFrame(columns=[
        'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'learning_rate'
    ])
    
    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)
    
    # Get initial validation metrics
    val_loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
    
    # Ensure values are detached from CUDA and converted to float
    if isinstance(val_acc, torch.Tensor):
        val_acc = val_acc.detach().cpu().item()
    if isinstance(val_loss, torch.Tensor):
        val_loss = val_loss.detach().cpu().item()
    
    # Store initial metrics (epoch 0)
    new_row = pd.DataFrame([{
        'epoch': 0,
        'train_loss': float('nan'),  # No training loss for epoch 0
        'train_acc': float('nan'),   # No training accuracy for epoch 0
        'val_loss': val_loss,
        'val_acc': val_acc,
        'learning_rate': optimizer.param_groups[0]['lr']
    }])
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
    
    # Plot initial metrics
    plot_metrics(metrics_df, 0)
    visualize_predictions(val_loader, model, classifier, 0)
    
    # Create a plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        train_loss, train_acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, train_acc))

        # eval for one epoch
        val_loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc
        
        # Ensure values are detached from CUDA and converted to float
        if isinstance(train_acc, torch.Tensor):
            train_acc = train_acc.detach().cpu().item()
        if isinstance(train_loss, torch.Tensor):
            train_loss = train_loss.detach().cpu().item()
        if isinstance(val_acc, torch.Tensor):
            val_acc = val_acc.detach().cpu().item()
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.detach().cpu().item()
            
        # Store metrics
        new_row = pd.DataFrame([{
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        }])
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
        
        # Plot metrics every 10 epochs and at the end
        if epoch % 10 == 0 or epoch == opt.epochs:
            print(f"\nGenerating plots for epoch {epoch}")
            plot_metrics(metrics_df, epoch)
            visualize_predictions(val_loader, model, classifier, epoch)
            print(f"Plots saved for epoch {epoch}\n")
            
        # Save metrics to csv
        metrics_df.to_csv('training_metrics.csv', index=False)

    print('best accuracy: {:.2f}'.format(best_acc))
    return best_acc


if __name__ == '__main__':
    main()
