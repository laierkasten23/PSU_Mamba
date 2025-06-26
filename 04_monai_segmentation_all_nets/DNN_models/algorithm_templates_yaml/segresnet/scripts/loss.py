# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compusernce with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
# @Time    : 2019/9/9 
# @Author  : Elliott Zheng  
# @Email   : admin@hypercube.top

# https://github.com/elliottzheng/AdaptiveWingLoss/blob/master/adaptive_wing_loss.py


from monai.losses import DiceCELoss

import math
import torch
from torch import nn

# torch.log  and math.log is e based
class WingLoss(nn.Module):
    """
    Wing loss function for regression tasks.

    Args:
        omega (float): Parameter that controls the width of the linear region in the loss function. Default is 10.
        epsilon (float): Parameter that prevents division by zero in the loss function. Default is 2.

    Returns:
        torch.Tensor: Computed loss value.

    """

    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))
    

class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        '''
        Initializes the AdaptiveWingLoss module.

        :param omega: The parameter omega for the loss calculation. Default is 14.
        :param theta: The parameter theta for the loss calculation. Default is 0.5.
        :param epsilon: The parameter epsilon for the loss calculation. Default is 1.
        :param alpha: The parameter alpha for the loss calculation. Default is 2.1.
        '''
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        Calculates the forward pass of the AdaptiveWingLoss.

        :param pred: The predicted tensor of shape BxNxHxH.
        :param target: The target tensor of shape BxNxHxH.
        :return: The calculated loss value.
        '''
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))
    

class DiceCELoss2(DiceCELoss):
    def ce(self, input: torch.Tensor, target: torch.Tensor):
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()

        return self.cross_entropy(input, target)

'''
if __name__ == "__main__":
    loss_func = AdaptiveWingLoss()
    y = torch.ones(2, 68, 64, 64)
    y_hat = torch.zeros(2, 68, 64, 64)
    y_hat.requires_grad_(True)
    loss = loss_func(y_hat, y)
    loss.backward()
    print(loss)


if __name__ == "__main__":
    loss_func = WingLoss()
    y = torch.ones(2, 68, 64, 64)
    y_hat = torch.zeros(2, 68, 64, 64)
    y_hat.requires_grad_(True)
    loss = loss_func(y_hat, y)
    loss.backward()
    print(loss)

'''

    

