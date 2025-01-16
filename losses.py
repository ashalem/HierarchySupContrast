"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class HierarchySupConLoss(nn.Module):
    """Hierarchical Supervised Contrastive Loss.

    This loss function extends the standard SupConLoss to handle multiple feature vectors per sample,
    each corresponding to different hierarchical levels or steps in the network. It computes the supervised
    contrastive loss for each level, applies a normalized weight to each, and aggregates them to produce
    the final loss value.

    Args:
        level_weights (list or tuple of floats): Weights for each hierarchical level. Length must match
            the number of levels in the features.
        temperature (float, optional): Temperature parameter for scaling the logits. Default is 0.07.
        contrast_mode (str, optional): Mode for contrastive loss, either 'all' or 'one'. Default is 'all'.
        base_temperature (float, optional): Base temperature for scaling. Default is 0.07.
    """
    def __init__(self, level_weights, temperature=0.05, contrast_mode='all', base_temperature=0.07):
        super(HierarchySupConLoss, self).__init__()
        if not isinstance(level_weights, (list, tuple)):
            raise TypeError(f'Expected level_weights to be a list or tuple, but got {type(level_weights)}')
        self.num_levels = len(level_weights)
        
        # Convert to tensor and register as a buffer so it moves with the module
        weights = torch.tensor(level_weights, dtype=torch.float32)
        if weights.sum() == 0:
            raise ValueError('Sum of level_weights must be greater than 0.')
        normalized_weights = weights / weights.sum()  # Normalize weights
        self.register_buffer('level_weights', normalized_weights)  # This will move with the module to GPU

        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.supcon_loss = SupConLoss(
            temperature=self.temperature,
            contrast_mode=self.contrast_mode,
            base_temperature=self.base_temperature
        )

    def forward(self, features, labels):
        """
        Compute the hierarchical supervised contrastive loss.

        Args:
            features (Tensor): Hidden vectors of shape [batch_size, num_levels, num_views, feature_dim].
            labels (Tensor): Ground truth labels of shape [batch_size, num_levels]. Each label corresponds to a level.

        Returns:
            Tensor: A scalar loss value.
        """
        # print(f"\nHierarchySupConLoss forward:")
        # print(f"Features shape: {features.shape}, device: {features.device}")
        # print(f"Labels shape: {labels.shape}, device: {labels.device}")
        # print(f"Level weights device: {self.level_weights.device}")
        # print(f"SupConLoss temperature device: {self.supcon_loss.temperature}")
        
        batch_size = features.shape[0]
        num_levels = features.shape[1]
        num_views = features.shape[2]
        
        # print(f"Batch size: {batch_size}, Num levels: {num_levels}, Num views: {num_views}")
        
        # Calculate loss for each level
        level_losses = []
        for levelIdx in range(num_levels):
            level_features = features[:, levelIdx, :, :]
            level_labels = labels[:, levelIdx]
            # print(f"\nLevel {levelIdx}:")
            # print(f"Level features shape: {level_features.shape}, device: {level_features.device}")
            # print(f"Level labels shape: {level_labels.shape}, device: {level_labels.device}")
            
            loss = self.supcon_loss(level_features, level_labels)
            # print(f"Level {levelIdx} loss: {loss.item():.4f}, device: {loss.device}")
            level_losses.append(loss)
            
        level_losses = torch.stack(level_losses)
        # print(f"Stacked losses device: {level_losses.device}")
        weighted_loss = torch.sum(level_losses * self.level_weights)
        # print(f"Final weighted loss: {weighted_loss.item():.4f}, device: {weighted_loss.device}")
        
        return weighted_loss    