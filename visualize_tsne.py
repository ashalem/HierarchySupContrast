from __future__ import print_function

import os
import sys
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from networks.resnet_big import SupConResNet, HierarchicalSupConResNet
from main_linear_hierarchical import CIFAR100Hierarchy

def parse_option():
    parser = argparse.ArgumentParser('argument for TSNE visualization')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='model architecture')
    parser.add_argument('--supcon_ckpt', type=str, required=True,
                        help='path to standard SupCon checkpoint')
    parser.add_argument('--hier_ckpt', type=str, required=True,
                        help='path to Hierarchical SupCon checkpoint')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='number of samples to use for TSNE')
    parser.add_argument('--perplexity', type=float, default=30.0,
                        help='perplexity parameter for TSNE')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    
    opt = parser.parse_args()
    
    # Set up paths
    opt.data_folder = './datasets/'
    
    return opt

def set_loader(opt):
    """Create data loaders for CIFAR-100 and hierarchical CIFAR-100"""
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    normalize = transforms.Normalize(mean=mean, std=std)
    
    # Simple transformation for evaluation (no augmentation)
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Regular CIFAR-100 dataset
    cifar100_dataset = datasets.CIFAR100(root=opt.data_folder,
                                       train=False,
                                       transform=transform,
                                       download=True)
    
    # Hierarchical CIFAR-100 dataset
    hierarchy_dataset = CIFAR100Hierarchy(root=opt.data_folder,
                                        transform=transform,
                                        train=False,
                                        download=True)
    
    # Set seed for reproducibility
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    
    # Create loaders
    cifar100_loader = torch.utils.data.DataLoader(
        cifar100_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)
    
    hierarchy_loader = torch.utils.data.DataLoader(
        hierarchy_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)
    
    return cifar100_loader, hierarchy_loader

def set_model(opt):
    """Load both models and their checkpoints"""
    # Standard SupCon model
    supcon_model = SupConResNet(name=opt.model)
    supcon_ckpt = torch.load(opt.supcon_ckpt, map_location='cpu')
    supcon_state_dict = supcon_ckpt['model']
    
    # Hierarchical SupCon model
    hier_model = HierarchicalSupConResNet(
        name=opt.model,
        head='mlp',
        feat_dim=128,
        is_output_layer=[False, True, False, True],
    )
    hier_ckpt = torch.load(opt.hier_ckpt, map_location='cpu')
    hier_state_dict = hier_ckpt['model']
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            supcon_model.encoder = torch.nn.DataParallel(supcon_model.encoder)
            hier_model.encoder = torch.nn.DataParallel(hier_model.encoder)
        else:
            # Handle state dict keys for non-DataParallel models
            new_supcon_state_dict = {}
            for k, v in supcon_state_dict.items():
                k = k.replace("module.", "")
                new_supcon_state_dict[k] = v
            supcon_state_dict = new_supcon_state_dict
            
            new_hier_state_dict = {}
            for k, v in hier_state_dict.items():
                k = k.replace("module.", "")
                new_hier_state_dict[k] = v
            hier_state_dict = new_hier_state_dict
            
        supcon_model = supcon_model.cuda()
        hier_model = hier_model.cuda()
        cudnn.benchmark = True
        
        supcon_model.load_state_dict(supcon_state_dict)
        hier_model.load_state_dict(hier_state_dict)
    else:
        print("CUDA not available. Using CPU (not recommended).")
        supcon_model.load_state_dict(supcon_state_dict)
        hier_model.load_state_dict(hier_state_dict)
    
    return supcon_model, hier_model

def extract_features(model, dataloader, n_samples, is_hier=False):
    """Extract features from the model for n_samples"""
    model.eval()
    
    # Lists to store features and labels
    all_features = []
    all_labels = []
    all_superclass_labels = []
    
    # Track how many samples we've processed
    samples_collected = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            # Break if we have enough samples
            if samples_collected >= n_samples:
                break
            
            # Get current batch size
            batch_size = images.size(0)
            
            # Limit the number of samples if needed
            if samples_collected + batch_size > n_samples:
                # Only take what we need
                take_samples = n_samples - samples_collected
                images = images[:take_samples]
                if is_hier:
                    superclass_labels, class_labels = labels
                    superclass_labels = superclass_labels[:take_samples]
                    class_labels = class_labels[:take_samples]
                    labels = (superclass_labels, class_labels)
                else:
                    labels = labels[:take_samples]
                batch_size = take_samples
            
            # Move to GPU
            images = images.cuda(non_blocking=True)
            
            # Extract features
            if is_hier:
                features = model.encoder(images)  # List of features from different layers
                # Concatenate early (features[0]) and deep (features[-1]) layers
                concat_features = torch.cat([features[0], features[-1]], dim=1)
                all_features.append(concat_features.cpu().numpy())
                superclass_labels, class_labels = labels
                all_labels.append(class_labels.numpy())
                all_superclass_labels.append(superclass_labels.numpy())
            else:
                features = model.encoder(images)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
            
            # Update count
            samples_collected += batch_size
    
    # Concatenate all features and labels
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    
    if is_hier:
        all_superclass_labels = np.concatenate(all_superclass_labels)
        return all_features, all_labels, all_superclass_labels
    else:
        return all_features, all_labels

def visualize_tsne(features, labels, title, filename, perplexity=30.0):
    """Create and save a TSNE visualization"""
    # Normalize features
    scaled_features = StandardScaler().fit_transform(features)
    
    # Apply TSNE
    print(f"Computing t-SNE for {title}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(scaled_features)
    
    # Get unique labels for color mapping
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # Create a colormap with enough distinct colors
    if n_classes <= 20:  # For superclasses
        cmap = plt.cm.get_cmap('tab20', n_classes)
    else:  # For regular classes (100)
        # Create a circular colormap for better distinction with many classes
        cmap = plt.cm.hsv
    
    # Create the plot
    plt.figure(figsize=(12, 10), dpi=300)
    
    # Plot each class with a different color
    for i, label in enumerate(unique_labels):
        idx = labels == label
        plt.scatter(tsne_result[idx, 0], tsne_result[idx, 1], 
                   c=[cmap(i / n_classes)], 
                   label=f'Class {label}',
                   alpha=0.7, 
                   s=20,
                   edgecolors='none')
    
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE dimension 1', fontsize=12)
    plt.ylabel('t-SNE dimension 2', fontsize=12)
    
    # Only show legend for superclasses (too many classes otherwise)
    if n_classes <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"TSNE visualization saved to {filename}")

def main():
    # Parse command line arguments
    opt = parse_option()
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots/tsne', exist_ok=True)
    
    # Set up data loaders
    cifar100_loader, hierarchy_loader = set_loader(opt)
    
    # Set up models
    supcon_model, hier_model = set_model(opt)
    
    # Extract features for standard SupCon
    supcon_features, supcon_labels = extract_features(
        supcon_model, cifar100_loader, opt.n_samples, is_hier=False
    )
    
    # Extract features for hierarchical SupCon
    hier_features, hier_labels, hier_superclass_labels = extract_features(
        hier_model, hierarchy_loader, opt.n_samples, is_hier=True
    )
    
    # Create TSNE visualizations
    visualize_tsne(
        supcon_features, 
        supcon_labels, 
        f"Standard SupCon - Class Distribution (n={opt.n_samples})",
        f"plots/tsne/supcon_classes_tsne.png",
        perplexity=opt.perplexity
    )
    
    visualize_tsne(
        hier_features, 
        hier_labels, 
        f"Hierarchical SupCon - Class Distribution (n={opt.n_samples})",
        f"plots/tsne/hier_classes_tsne.png",
        perplexity=opt.perplexity
    )
    
    visualize_tsne(
        hier_features, 
        hier_superclass_labels, 
        f"Hierarchical SupCon - Superclass Distribution (n={opt.n_samples})",
        f"plots/tsne/hier_superclasses_tsne.png",
        perplexity=opt.perplexity
    )
    
    print("All t-SNE visualizations completed!")

if __name__ == '__main__':
    main() 