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