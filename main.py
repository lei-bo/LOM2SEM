import numpy as np
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from models import UNet, Discriminator
from utils import show_tensor_images, AvgMeter, Recorder


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# modeling
SIZE = (128, 128)
LAMBDA_RECON = 100

# training
N_EPOCHS = 1000
BATCH_SIZE = 16
LR = 1e-4

# visualization and save
DISPLAY_STEP = 500
RECORD_STEP = 100
NROW = 6 # number of images per row


class ImagePairDataset(Dataset):
    """A dataset consists of pairs of images. Images in the same pair come from
    different folders with the same name.

    Args:
        root (string): Root directory path.
        folder_A (string): Folder of image type A.
        folder_A (string): Folder of image type B.
        split (string): A text file listing the name of images.
    """
    def __init__(self, root: str, folder_A: str, folder_B: str, split: str):
        super(ImagePairDataset, self).__init__()
        self.img_names = np.loadtxt(f"{root}/split/{split}",
                                    dtype=str, delimiter='\n', ndmin=1)
        self.folder_A = f"{root}/{folder_A}"
        self.folder_B = f"{root}/{folder_B}"
        self.transform = T.Compose([T.Resize(SIZE),
                                    T.ToTensor()])

    def __getitem__(self, index: int):
        img_name = self.img_names[index]
        img_A = self._get_image(self.folder_A, img_name)
        img_B = self._get_image(self.folder_B, img_name)
        return img_A, img_B[0][None,:,:]

    def _get_image(self, folder: str, img_name: str):
        img_path = f'{folder}/{img_name}'
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_names)

def train():
    train_set = ImagePairDataset("./data/mecs_steel",
                                 "LOM640", "SEM640", "all.txt")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    gen = UNet(input_channels=3, output_channels=1,
               hidden_channels=32).to(DEVICE)
    disc = Discriminator(input_channels=4).to(DEVICE)
    recon_criterion = nn.L1Loss().to(DEVICE)
    adv_criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    gen_opt = optim.Adam(gen.parameters(), lr=LR)
    disc_opt = optim.Adam(disc.parameters(), lr=LR)

    cur_step = 0
    gen_meter = AvgMeter('gen_loss', ':.5f')
    disc_meter = AvgMeter('disc_loss', ':.5f')
    recorder = Recorder(('iter', 'gen_loss', 'disc_loss'))
    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch+1}/{N_EPOCHS}")
        for real_A, real_B in tqdm(train_loader):
            real_A, real_B = real_A.to(DEVICE), real_B.to(DEVICE)
            fake_B = gen(real_A) # shape (N,1,H,W)

            # Update discriminator
            disc_opt.zero_grad()
            disc_loss = get_disc_loss(disc, real_B, fake_B, real_A,
                                      adv_criterion)
            disc_meter.update(disc_loss.item(), BATCH_SIZE)
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            # Update generator
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, real_B, real_A,
                                    adv_criterion, recon_criterion)
            gen_meter.update(gen_loss.item(), BATCH_SIZE)
            gen_loss.backward()
            gen_opt.step()

            if cur_step % DISPLAY_STEP == 0:
                image_tensor = torch.cat([real_A[:NROW],
                                          torch.cat([real_B]*3, dim=1)[:NROW],
                                          torch.cat([fake_B]*3, dim=1)[:NROW]])
                show_tensor_images(image_tensor, nrow=NROW, num_images=NROW*3)

            if cur_step > 0 and cur_step % RECORD_STEP == 0:
                print(f"Epoch {epoch + 1}: Step {cur_step}: "
                      f"gen_loss: {gen_meter.avg}, "
                      f"disc_loss: {disc_meter.avg}")
                recorder.update([cur_step, gen_meter.avg, disc_meter.avg])
                gen_meter.reset()
                disc_meter.reset()
                torch.save({'gen': gen.state_dict(),
                            'gen_opt': gen_opt.state_dict(),
                            'disc': disc.state_dict(),
                            'disc_opt': disc_opt.state_dict(),
                            'losses': recorder.record
                            }, f"./checkpoints/pix2pix.pth")
            cur_step += 1

def get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion):
    """
    Get generator loss (adversarial loss + reconstruction loss).
    Args:
        gen: The generator model.
        disc: The discriminator model.
        real: Target real images.
        condition: Source real images.
        adv_criterion: The adversarial loss function.
        recon_criterion:

    Returns: Generator loss.
    """
    fake = gen(condition)
    pred = disc(fake, condition)
    gen_adv_loss = adv_criterion(pred, torch.ones_like(pred))
    gen_recon_loss = recon_criterion(fake, real)
    gen_loss = gen_adv_loss + LAMBDA_RECON * gen_recon_loss
    return gen_loss

def get_disc_loss(disc, real, fake, condition, adv_criterion):
    """
    Get discriminator loss.
    Args:
        disc: The discriminator model.
        real: Target real images.
        fake: Generated images.
        condition: Source real images.
        adv_criterion: The adversarial loss function.

    Returns: Discriminator loss.
    """
    fake_pred = disc(fake.detach(), condition)
    fake_loss = adv_criterion(fake_pred, torch.zeros_like(fake_pred))
    real_pred = disc(real, condition)
    real_loss = adv_criterion(real_pred, torch.ones_like(real_pred))
    return real_loss + fake_loss


if __name__ == '__main__':
    train()