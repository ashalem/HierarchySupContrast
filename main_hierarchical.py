from __future__ import print_function

import os
import time
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import math

from util import TwoCropTransform, AverageMeter
from networks.resnet_big import HierarchicalSupConResNet
from losses import HierarchySupConLoss
from utils.debug_utils import check_tensor, check_gradients

def set_loader():
    # Data loading parameters
    batch_size = 1024  
    num_workers = 12  
    data_folder = './datasets/'

    # CIFAR100 mean and std
    # Mean and standard deviation values for CIFAR100 dataset normalization
    mean = (0.5071, 0.4867, 0.4408)  # RGB channel means
    std = (0.2675, 0.2565, 0.2761)   # RGB channel standard deviations
    normalize = transforms.Normalize(mean=mean, std=std)  # Normalization transform

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
    train_dataset = CIFAR100Hierarchy(root=data_folder,
                                    transform=TwoCropTransform(train_transform),
                                    download=True)

    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    return train_loader

def set_model():
    # Define model with hierarchical outputs at layers 2 and 4
    model = HierarchicalSupConResNet(
        name='resnet18',
        head='mlp',
        feat_dim=128,
        is_output_layer=[False, False, False, True],
    )
    
    # Define loss with weights for each level
    criterion = HierarchySupConLoss(
        level_weights=[1],  # More Weight for the second level
        temperature=0.1,  # Reduced from 0.1 to make loss more sensitive
        contrast_mode='all',
    )

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

def set_optimizer_and_scheduler(model):
    # Increased learning rate from 0.15 to 0.3
    optimizer = torch.optim.SGD(model.parameters(),
                               lr=0.1,  # Doubled the learning rate
                               momentum=0.9,
                               weight_decay=1e-4)
    
    # Define warmup scheduler + more aggressive LR decay
    num_epochs = 100  # Total epochs
    warmup_epochs = 3  # Warmup period
    
    def lr_schedule(epoch):
        # Linear warmup for warmup_epochs
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        # Cosine decay after warmup
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    
    return optimizer, scheduler

def train(train_loader, model, criterion, optimizer, epoch):
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
        if idx % 10 == 0:
            print(f'Train: [{epoch}][{idx}/{len(train_loader)}]\t'
                  f'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'loss {loss.item():.3f} ({losses.avg:.3f})')
    
    return losses.avg

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

def main():
    # Get data loader
    train_loader = set_loader()
    
    # Build model and criterion
    model, criterion = set_model()
    
    # Build optimizer and scheduler
    optimizer, scheduler = set_optimizer_and_scheduler(model)
    
    # Training loop
    for epoch in range(1, 201):  # 200 epochs
        # Train for one epoch
        loss = train(train_loader, model, criterion, optimizer, epoch)
            
        # Step the scheduler
        scheduler.step()

        # Save model
        if epoch % 50 == 0:
            save_file = f'./save/ckpt_epoch_{epoch}.pth'
            if not os.path.exists('./save'):
                os.makedirs('./save')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, save_file)

    return

if __name__ == '__main__':
    main() 