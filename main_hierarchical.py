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

def set_loader():
    # Data loading parameters
    batch_size = 256
    num_workers = 2
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
        feat_dim=64,
        is_output_layer=[False, True, False, True],
        scale_up=False  # Default to scaling down to 64 dimensions
    )
    
    # Define loss with weights for each level
    criterion = HierarchySupConLoss(
        level_weights=[0.5, 0.5],  # Try equal weights first
        temperature=0.07  # Lower temperature for sharper contrasts
    )

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

def train(train_loader, model, criterion, optimizer, epoch):
    """One epoch training"""
    model.train()
    print(f"\nModel device: {next(model.parameters()).device}")
    print(f"Criterion device: {criterion.level_weights.device}")

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        # Print initial devices
        # print(f"\nInitial devices:")
        # print(f"Images device: {images[0].device}")
        # print(f"Labels devices: {labels[0].device}, {labels[1].device}")

        # Unpack hierarchical labels
        superclass_labels, class_labels = labels
        labels = torch.stack([superclass_labels, class_labels], dim=1)
        # print(f"Stacked labels device: {labels.device}")

        images = torch.cat([images[0], images[1]], dim=0)
        # print(f"Concatenated images device: {images.device}")
        
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # print(f"\nAfter CUDA transfer:")
            # print(f"Images device: {images.device}")
            # print(f"Labels device: {labels.device}")
        
        bsz = labels.shape[0]

        # Forward pass
        features = model(images)
        # print(f"Features after model device: {features.device}")
        
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(2), f2.unsqueeze(2)], dim=2)
        # print(f"Features after processing device: {features.device}")
        
        # Compute loss
        loss = criterion(features, labels)
        # print(f"Loss device: {loss.device}")
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print info
        if (idx + 1) % 10 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

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
    # Add this line at the start of main
    torch.autograd.set_detect_anomaly(True)
    
    # Set up data loader
    train_loader = set_loader()

    # Set up model and criterion
    model, criterion = set_model()

    # Use Adam optimizer with a higher initial learning rate
    optimizer = torch.optim.Adam(model.parameters(),
                               lr=1e-2,  # Higher initial learning rate
                               betas=(0.9, 0.999),  # Default momentum parameters for Adam
                               eps=1e-8,            # Small constant for numerical stability
                               weight_decay=1e-4    # L2 regularization factor
    
    # Simple cosine schedule without warmup
    total_epochs = 100

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_epochs,
        eta_min=1e-4  # Minimum learning rate
    )

    # Training loop
    for epoch in range(1, total_epochs + 1):
        # Train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch)
        time2 = time.time()
        print('Epoch {}, total time {:.2f}, loss {:.3f}, lr {:.6f}'.format(
            epoch, time2 - time1, loss, optimizer.param_groups[0]['lr']))
        
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

if __name__ == '__main__':
    main() 