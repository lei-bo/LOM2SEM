import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show_tensor_images(image_tensor, title, num_images=8, nrow=4):
    """
    Plot image tensors in a grid.
    Args:
        image_tensor: A tensor of images with shape (N,C,H,W).
        title: The title of the plot.
        num_images: Number of images to plot.
        nrow: Number of images displayed in each row of the grid.
    """
    image_tensor = image_tensor.detach().cpu()
    image_grid = make_grid(image_tensor[:num_images], nrow=nrow, padding=5)
    plt.figure(dpi=150, constrained_layout=True)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis("off")
    plt.title(title)
    plt.show()


class ScoreMeter:
    """Computes and stores average scores."""
    def __init__(self):
        self.gen_loss = 0
        self.gen_adv_loss = 0
        self.gen_recon_loss = 0
        self.disc_loss = 0
        self.disc_real_loss = 0
        self.disc_fake_loss = 0
        self.count = 0

    def update(self, gen_loss, gen_adv_loss, gen_recon_loss,
               disc_loss, disc_real_loss, disc_fake_loss, n=1):
        self.gen_loss += gen_loss * n
        self.gen_adv_loss += gen_adv_loss * n
        self.gen_recon_loss += gen_recon_loss * n
        self.disc_loss += disc_loss * n
        self.disc_real_loss += disc_real_loss * n
        self.disc_fake_loss += disc_fake_loss * n
        self.count += n

    def reset(self):
        self.__init__()

    def stats(self):
        stats = []
        for score in [self.gen_loss, self.gen_adv_loss, self.gen_recon_loss,
                      self.disc_loss, self.disc_real_loss, self.disc_fake_loss]:
            stats.append(score / self.count)
        return stats

    def stats_string(self):
        gen_loss, gen_adv_loss, gen_recon_loss, disc_loss, disc_real_loss, disc_fake_loss = self.stats()
        return f"gen_loss {gen_loss:.3f} | gen_adv_loss {gen_adv_loss:.3f} | "\
               f"gen_recon_loss {gen_recon_loss:.3f} | disc_loss {disc_loss:.3f} | "\
               f"disc_real_loss {disc_real_loss:.3f} | disc_fake_loss {disc_fake_loss:.3f}"


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
