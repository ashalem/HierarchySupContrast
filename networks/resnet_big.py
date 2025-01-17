"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block_class, num_blocks, in_channel=3, zero_init_residual=False):
        """
        Initializes the ResNet model.

        Args:
            block (nn.Module): The block type to be used (e.g., BasicBlock or Bottleneck).
            num_blocks (list of int): A list containing the number of blocks for each of the four layers.
            in_channel (int, optional): Number of input channels. Default is 3.
            zero_init_residual (bool, optional): If True, initializes the last BatchNorm layer in each residual branch to zero. Default is False.

        Attributes:
            in_planes (int): Number of input planes for the first convolutional layer.
            conv1 (nn.Conv2d): First convolutional layer.
            bn1 (nn.BatchNorm2d): Batch normalization layer after the first convolutional layer.
            layer1 (nn.Sequential): First layer of residual blocks.
            layer2 (nn.Sequential): Second layer of residual blocks.
            layer3 (nn.Sequential): Third layer of residual blocks.
            layer4 (nn.Sequential): Fourth layer of residual blocks.
            avgpool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer.

        Initializes the weights of the convolutional and batch normalization layers.
        If zero_init_residual is True, initializes the last BatchNorm layer in each residual branch to zero.
        """
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block_class, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block_class, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block_class, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block_class, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block_class, planes, num_blocks, stride):
        """
        Creates a sequential layer composed of multiple blocks.

        Args:
            block (nn.Module): The block class to be used for creating the layer.
            planes (int): The number of output channels for each block.
            num_blocks (int): The number of blocks to be stacked in the layer.
            stride (int): The stride to be used for the first block in the layer.

        Returns:
            nn.Sequential: A sequential container of the stacked blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block_class(self.in_planes, planes, stride))
            self.in_planes = planes * block_class.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out
    
    
class RepeatAdapter(nn.Module):
    def __init__(self, repeat_factor):
        super().__init__()
        self.repeat_factor = repeat_factor
    
    def forward(self, x):
        return x.repeat_interleave(self.repeat_factor, dim=-1)

class PoolAdapter(nn.Module):
    def __init__(self, pool_factor):
        super().__init__()
        self.pool_factor = pool_factor
    
    def forward(self, x):
        return x.view(-1, self.pool_factor, x.size(-1) // self.pool_factor).mean(dim=1)

class HierarchicalResNet(ResNet):
    def __init__(self, block_class, num_blocks, is_output_layer=[False, False, False, True], 
                 in_channel=3, zero_init_residual=False, feat_dim=128):
        super(HierarchicalResNet, self).__init__(block_class, num_blocks, in_channel, zero_init_residual)
        self.is_output_layer = is_output_layer
        self.num_output_layers = sum(is_output_layer)
        
        # Add dimension matching
        self.dims = [64 , 128, 256, 512]
        
        # Set target dimension based on scaling direction
        self.target_dim = feat_dim
        
        # Create heads for each output layer as ModuleList
        heads = []
        for i, is_output in enumerate(is_output_layer):
            if not is_output:
                continue
            heads.append(nn.Sequential(
                nn.Linear(self.dims[i], self.dims[i]),
                nn.ReLU(inplace=True),
                nn.Linear(self.dims[i], self.target_dim)
            ))
        self.heads = nn.ModuleList(heads)
        
    
    def forward(self, x, layer=100):
        if self.num_output_layers == 0:
            return None

        stacked_out_tensor = []
        head_idx = 0
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if self.is_output_layer[0]:
            prepared_out = self.avgpool(out)
            prepared_out = torch.flatten(prepared_out, 1)
            prepared_out = self.heads[head_idx](prepared_out)
            stacked_out_tensor.append(prepared_out)
            head_idx += 1
            
        out = self.layer2(out)
        if self.is_output_layer[1]:
            prepared_out = self.avgpool(out)
            prepared_out = torch.flatten(prepared_out, 1)
            prepared_out = self.heads[head_idx](prepared_out)
            stacked_out_tensor.append(prepared_out)
            head_idx += 1
            
        out = self.layer3(out)
        if self.is_output_layer[2]:
            prepared_out = self.avgpool(out)
            prepared_out = torch.flatten(prepared_out, 1)
            prepared_out = self.heads[head_idx](prepared_out)
            stacked_out_tensor.append(prepared_out)
            head_idx += 1
            
        out = self.layer4(out)
        if self.is_output_layer[3]:
            prepared_out = self.avgpool(out)
            prepared_out = torch.flatten(prepared_out, 1)
            prepared_out = self.heads[head_idx](prepared_out)
            stacked_out_tensor.append(prepared_out)
            head_idx += 1

        stacked = torch.stack(stacked_out_tensor, dim=1)
        return stacked

def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


# A dictionary that maps the model name to the model class and the number of output channels.
# Those output channels are used for the projection head as the input dimension.
model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        # model_fun is the model class, e.g., resnet50
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class HierarchicalSupConResNet(SupConResNet):
    def __init__(self, name='resnet18', head='mlp', feat_dim=128, 
                 is_output_layer=[False, False, False, True]):
        super(HierarchicalSupConResNet, self).__init__(name, head, feat_dim)
        self.num_output_layers = sum(is_output_layer)
        self.encoder = HierarchicalResNet(
            BasicBlock, [2, 2, 2, 2],
            is_output_layer=is_output_layer,
            feat_dim=feat_dim
        )

    def forward(self, x):
        stacked_out_tensor = self.encoder(x)
        
        if self.num_output_layers != stacked_out_tensor.shape[1]:
            raise ValueError(
                f"Number of output layers ({self.num_output_layers}) does not match "
                f"the number of output layers in the encoder ({stacked_out_tensor.shape[1]})"
            )
        
        stacked_normalized = []
        for i in range(self.num_output_layers):
            after_head = stacked_out_tensor[:, i, :]
            normalized_tensor = F.normalize(after_head, dim=1)
            stacked_normalized.append(normalized_tensor)
            
        normalized_tensor_stacked = torch.stack(stacked_normalized, dim=1)
        return normalized_tensor_stacked


class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
