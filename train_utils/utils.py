import os
import random
import torch
import numpy as np
import torch.nn as nn
import torchvision.utils
from torchvision import datasets, transforms


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)


def save_checkpoint(net, epoch, args):
    """
    Save the checkpoint for an opacus processed model
    """
    state = {
        'net': net._module.state_dict(),
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
    }

    torch.save(state, os.path.join(args.result_dir, f'dp_trained_model.pth'))


def save_checkpoint_non_opacus(net, epoch, args):
    """
    Save the checkpoint for an opacus processed model
    """
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
    }

    torch.save(state, os.path.join(args.result_dir, f'trained_model.pth'))


def keep_layer(model, keep_list):
    """
        Args: keep_list ['fc', 'bn']
    """
    for name, param in model.named_parameters():
        param.requires_grad = False
        for keep in keep_list:
            if keep in name:
                param.requires_grad = True
    return model


def imshow(images, spc, img_specs):
    num_classes = img_specs['num_classes']
    img_width = img_specs['width']
    img_height = img_specs['height']
    num_channel = img_specs['num_channel']

    # Convert PyTorch tensor to NumPy array
    image_array = images
    image_array = image_array.reshape(num_classes, spc, num_channel, img_width, img_height)

    # Create a 10x10 grid for displaying 100 images
    fig, axes = plt.subplots(num_classes, spc, figsize=(spc, num_classes))

    for i in range(num_classes):
        for j in range(spc):
            if num_channel == 1:
                axes[i, j].imshow(image_array[i, j, 0], cmap='gray')
            else:
                axes[i, j].imshow(image_array[i, j].transpose(1, 2, 0))
            axes[i, j].axis('off')

    plt.tight_layout()  # Ensure proper spacing
    plt.show()
