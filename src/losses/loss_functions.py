# loss_functions.py

import torch
import torch.nn as nn
import torchvision

class MSELoss(nn.Module):
    """Mean Squared Error Loss"""
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss()
    
    def forward(self, outputs, targets):
        return self.loss(outputs, targets)


class MAELoss(nn.Module):
    """Mean Absolute Error Loss (L1 Loss)"""
    def __init__(self):
        super(MAELoss, self).__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, outputs, targets):
        return self.loss(outputs, targets)


class HuberLoss(nn.Module):
    """Huber Loss"""
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.loss = nn.SmoothL1Loss()
    
    def forward(self, outputs, targets):
        return self.loss(outputs, targets)


class SSIMLoss(nn.Module):
    """Structural Similarity Index (SSIM) Loss"""
    def __init__(self, window_size=11, reduction='mean'):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.reduction = reduction
    
    def forward(self, outputs, targets):
        return 1 - self.ssim(outputs, targets)
    
    def ssim(self, outputs, targets):
        # Using a simplified SSIM implementation for demonstration
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = torch.mean(outputs, dim=(2, 3), keepdim=True)
        mu_y = torch.mean(targets, dim=(2, 3), keepdim=True)

        sigma_x = torch.var(outputs, dim=(2, 3), unbiased=False, keepdim=True)
        sigma_y = torch.var(targets, dim=(2, 3), unbiased=False, keepdim=True)
        sigma_xy = torch.mean((outputs - mu_x) * (targets - mu_y), dim=(2, 3), keepdim=True)

        ssim_index = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        return ssim_index.mean() if self.reduction == 'mean' else ssim_index

import torch
import torch.nn as nn
import torchvision

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        
        # Move blocks to GPU if available
        for bl in blocks:
            bl.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            for p in bl:
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move inputs and targets to the same device as the blocks
        input = input.to(device)
        target = target.to(device)
        
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean.to(device)) / self.std.to(device)
        target = (target-self.mean.to(device)) / self.std.to(device)
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

    
class PerceptualLoss(nn.Module):
    """Perceptual Loss (using a pre-trained VGG model)"""
    def __init__(self, model, layers=None):
        super(PerceptualLoss, self).__init__()
        self.model = model
        self.layers = layers if layers else ['relu1_2', 'relu2_2', 'relu3_3']
        self.criterion = nn.MSELoss()
        self.model.eval()

    def forward(self, outputs, targets):
        features_outputs = self.extract_features(outputs)
        features_targets = self.extract_features(targets)
        loss = 0.0
        for f_out, f_tar in zip(features_outputs, features_targets):
            loss += self.criterion(f_out, f_tar)
        return loss

    def extract_features(self, x):
        features = []
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.layers:
                features.append(x)
        return features


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (a variant of L1 Loss)"""
    def __init__(self, epsilon=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        diff = outputs - targets
        loss = torch.sqrt(diff ** 2 + self.epsilon ** 2)
        return loss.mean()


class LossSelector(nn.Module):
    """Selects and combines the appropriate loss functions based on input"""
    def __init__(self, loss_config=None, **kwargs):
        super(LossSelector, self).__init__()

        if loss_config is None:
            loss_config = {"mse": 1.0}

        self.losses = []
        self.weights = []
        for loss_type, weight in loss_config.items():
            self.losses.append(self._get_loss_function(loss_type, **kwargs))
            self.weights.append(weight)

    def _get_loss_function(self, loss_type, **kwargs):
        """Returns the appropriate loss function based on the loss_type"""
        loss_type = loss_type.lower()

        if loss_type == "mse":
            return MSELoss()
        elif loss_type == "mae":
            return MAELoss()
        elif loss_type == "huber":
            return HuberLoss(**kwargs)
        elif loss_type == "ssim":
            return SSIMLoss(**kwargs)
        elif loss_type == "perceptual":
            return VGGPerceptualLoss(**kwargs)
        elif loss_type == "charbonnier":
            return CharbonnierLoss(**kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, outputs, targets):
        total_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(outputs, targets)
        return total_loss

