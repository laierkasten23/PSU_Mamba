import torch
import numpy as np
import torch.nn.functional as F

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

# https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/nnunet/utilities/tensor_utilities.py
def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


# https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/nnunet/utilities/nd_softmax.py
def softmax_helper(x): 
    return lambda x: F.softmax(x, 1)