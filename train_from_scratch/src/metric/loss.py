import torch 
from torch import nn
from src.utils import sum_tensor 



def get_tp_fp_fn_tn(net_output, groundtruth, axes=None, mask=None, square=False):
    """
    Calculate true positives (tp), false positives (fp), false negatives (fn), and true negatives (tn) for a given network output and ground truth.

    Args:
        net_output (torch.Tensor): The network output tensor of shape (b, c, x, y(, z)).
        groundtruth (torch.Tensor): The ground truth tensor of shape (b, 1, x, y(, z)) or (b, x, y(, z)) or one hot encoding (b, c, x, y(, z)).
        axes (tuple, optional): The axes along which to perform summation. Default is None, which means no summation. can be (, ) = no summation
        mask (torch.Tensor, optional): The mask tensor of shape (b, 1, x, y(, z)). It must have 1 for valid pixels and 0 for invalid pixels. Default is None.
        square (bool, optional): If True, fp, tp, and fn will be squared before summation. Default is False.

    Returns:
        torch.Tensor: The true positives (tp) tensor.
        torch.Tensor: The false positives (fp) tensor.
        torch.Tensor: The false negatives (fn) tensor.
        torch.Tensor: The true negatives (tn) tensor.
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = groundtruth.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            groundtruth = groundtruth.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, groundtruth.shape)]):
            # if this is the case then groundtruth is probably already a one hot encoding
            y_onehot = groundtruth
        else:
            groundtruth = groundtruth.long()
            y_onehot = torch.zeros(shp_x, device=net_output.device)
            y_onehot.scatter_(1, groundtruth, 1)

    # y_onehot is b, c, x, y(, z)
    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn



class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        Soft Dice Loss function for semantic segmentation.

        Args:
            apply_nonlin (callable): Non-linear function to be applied to the input tensor `x`.
            batch_dice (bool): If True, compute dice loss over the batch dimension as well.
            do_bg (bool): If True, include background class in the dice loss computation.
            smooth (float): Smoothing factor to avoid division by zero.

        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        """
        Compute the forward pass of the Soft Dice Loss.

        Args:
            x (torch.Tensor): Predicted segmentation tensor.
            y (torch.Tensor): Ground truth segmentation tensor.
            loss_mask (torch.Tensor): Optional mask tensor to apply element-wise multiplication.

        Returns:
            torch.Tensor: Negative Soft Dice Loss.

        """
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc

class IoULoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        Intersection over Union (IoU) Loss.

        Args:
            apply_nonlin (callable): Optional non-linear function to be applied to the input before computing the loss.
            batch_dice (bool): Flag indicating whether to compute dice loss per batch or per image.
            do_bg (bool): Flag indicating whether to include the background class in the loss calculation.
            smooth (float): Smoothing factor to avoid division by zero.
            square (bool): Flag indicating whether to square the inputs before computing the loss.

        References:
            - Paper: https://link.springer.com/chapter/10.1007/978-3-319-50835-1_22 (page 5)
        """
        super(IoULoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        """
        Compute the IoU loss.

        Args:
            x (torch.Tensor): Predicted output tensor.
            y (torch.Tensor): Target tensor.
            loss_mask (torch.Tensor): Optional mask tensor to apply element-wise multiplication.

        Returns:
            torch.Tensor: Computed IoU loss.

        """
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn,_ = get_tp_fp_fn_tn(x, y, axes, loss_mask, self.square)

        iou = (tp + self.smooth) / (tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                iou = iou[1:]
            else:
                iou = iou[:, 1:]
        iou = iou.mean()

        return -iou
