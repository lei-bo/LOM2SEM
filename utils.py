import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show_tensor_images(image_tensor, num_images=8, nrow=4):
    """
    Plot image tensors in a grid.
    Args:
        image_tensor: A tensor of images with shape (N,C,H,W).
        num_images: Number of images to plot.
        nrow: Number of images displayed in each row of the grid.
    """
    image_tensor = image_tensor.detach().cpu()
    image_grid = make_grid(image_tensor[:num_images], nrow=nrow, padding=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis("off")
    plt.show()


class AvgMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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


class Recorder(object):
    """Record lists of variables"""
    def __init__(self, names):
        self.names = names
        self.record = {}
        for name in self.names:
            self.record[name] = []

    def update(self, vals):
        for name, val in zip(self.names, vals):
            self.record[name].append(val)
