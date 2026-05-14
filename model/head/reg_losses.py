# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from model.head.loss_ssim import ssim
shape = (256, 256)


class LossFunction_Dense(nn.Module):
    def __init__(self):
        super(LossFunction_Dense, self).__init__()
        self.gradient_loss = gradient_loss()
        self.ssim_loss =L_SSIM()
        self.feat_ncc_loss =feat_NCC_loss()

    def forward(self, y, y_f, tgt, src, mov_encs,fix_aligneds,flow):

        hyper_L1  = 1           
        hyper_ssim = 1
        hyper_grad = 15   
        hyper_ncc =0.01
        # TODO: similarity loss
        l1_1 = torch.nn.functional.l1_loss(tgt, y)
        l1_2 = torch.nn.functional.l1_loss(src, y_f)
        l1=hyper_L1*(l1_1+l1_2)
        
        ssim_1=self.ssim_loss(tgt, y)
        ssim_2=self.ssim_loss(src, y_f)
        ssim=hyper_ssim*(ssim_1 + ssim_2)
        # TODO: feature loss
        [mov_enc0, mov_enc1,mov_enc2,mov_enc3]=mov_encs
        [fix_aligned0,fix_aligned1,fix_aligned2,fix_aligned3]=fix_aligneds
        
        ncc_4 = self.feat_ncc_loss(mov_enc3, fix_aligned3)
        ncc   = hyper_ncc *(ncc_4)

        # TODO: gradient loss
        grad = hyper_grad*self.gradient_loss(flow)
        
        loss =  ssim + grad + l1 + ncc
        return loss, ssim, l1, grad, ncc

                     
class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        

    def forward(self, image_A, image_fused):
        
        Loss_SSIM = (1-ssim(image_A, image_fused))
        return Loss_SSIM
class gradient_loss(nn.Module):
    def __init__(self):
        super(gradient_loss, self).__init__()

    def forward(self, s, penalty='l2'):
        dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
        dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])
        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
        d = torch.mean(dx) + torch.mean(dy)
        return d / 2.0

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class feat_NCC_loss(nn.Module):
    def __init__(self):
        super(feat_NCC_loss, self).__init__()
        
    def forward(self, z1, z2, alpha=0.005):
        """
        Args:
            z1: Tensor of shape (B, C, H, W)  
                First feature map.
            z2: Tensor of shape (B, C, H, W)  
                Second feature map.
            alpha: float  
                Weight factor for penalizing off-diagonal correlation terms.

        Returns:
            torch.float: Loss value.
        """

        b, c, h, w = z1.shape

        # Flatten spatial dimensions: (B, C, H, W) ¡ú (B, L, C)
        z1 = z1.view(b, c, -1).transpose(1, 2)  # (B, L, C)
        z2 = z2.view(b, c, -1).transpose(1, 2)  # (B, L, C)

        return self.bt_loss_sub(z1, z2, alpha)
            
    def bt_loss_sub(self, z1, z2, alpha):
        """
        Args:
            z1: Tensor of shape (B, N, C)
            z2: Tensor of shape (B, N, C)
            alpha: float
        
        Returns:
            torch.float: Loss value.
        """

        # Compute batch NCC matrix ¡ú (B, C, C)
        ncc_mat = self.calc_batch_ncc_matrix(z1, z2)

        # Extract diagonal (similarity of corresponding channels)
        on_diag = ncc_mat.diagonal(dim1=1, dim2=2)  # (B, C)

        # Extract off-diagonal elements (cross-channel correlation)
        off_diag = self.off_diagonal(ncc_mat)  # (B, C*C - C)

        # Diagonal terms should approach 1; off-diagonals should approach 0
        loss = (on_diag - 1).square().sum(dim=1) + alpha * off_diag.square().sum(dim=1)

        return loss.mean()
        
    def calc_batch_ncc_matrix(self, a, b):
        """
        Compute the Normalized Cross-Correlation (NCC) matrix.

        Args:
            a, b: tensors with shape (B, d1, d2)

        Returns:
            Tensor of shape (B, d2, d2): NCC matrix.
        """
        
        d1 = a.shape[1]

        # Normalize along the d1 dimension
        z_a = self.standardize(a, dim=1)
        z_b = self.standardize(b, dim=1)

        # NCC computation: (B, d2, d2)
        ncc_mat = torch.bmm(z_a.transpose(1, 2), z_b) / (d1 - 1)
        return ncc_mat

    def off_diagonal(self, x):
        """
        Extract all non-diagonal elements from (B, L, L) matrices.

        Args:
            x: Tensor of shape (B, L, L)

        Returns:
            Tensor of shape (B, L*L - L): all off-diagonal entries.
        """
        batch_size = x.shape[0]
        l = x.shape[1]

        x = x.view(batch_size, -1)
        x = x[:, :-1].view(batch_size, l - 1, l + 1)

        return x[:, :, 1:].reshape(batch_size, -1)

    def standardize(self, x, dim, eps=1e-5):
        """
        Standardize tensor along a given dimension.

        Args:
            x: input tensor
            dim: dimension along which to compute mean and std
            eps: numerical stability constant

        Returns:
            Standardized tensor.
        """
        std, mean = torch.std_mean(x, dim=dim, keepdim=True)
        return (x - mean) / (eps + std)


