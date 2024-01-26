import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show_tensor_images(image_tensor, title, num_images=8, nrow=4, save_path=None):
    """
    Plot image tensors in a grid.
    Args:
        image_tensor: A tensor of images with shape (N,C,H,W).
        title: The title of the plot.
        num_images: Number of images to plot.
        nrow: Number of images displayed in each row of the grid.
        save_path: If save_path is provided, save the plot instead of showing.
    """
    image_tensor = image_tensor.detach().cpu()
    image_grid = make_grid(image_tensor[:num_images], nrow=nrow, padding=5)
    aspect_ratio = image_grid.shape[2] / image_grid.shape[1]
    plt.figure(dpi=150, figsize=(6 * aspect_ratio, 6), constrained_layout=True)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis("off")
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    return image_grid


class ScoreMeter:
    def __init__(self, score_names):
        self.score_names = score_names
        self.score_sum = {name: 0 for name in score_names}
        self.count = 0

    def update(self, scores, n=1):
        for name, score in zip(self.score_names, scores):
            self.score_sum[name] += score * n
        self.count += n

    def reset(self):
        self.__init__(self.score_names)

    def stats_dict(self):
        stats = dict()
        for k, v in self.score_sum.items():
            stats[k] = v / self.count
        return stats

    def stats_string(self):
        stats = self.stats_dict()
        stats_string = ""
        for name, stat in stats.items():
            stats_string += f"{name} {stat:.3f} | "
        return stats_string[:-3]