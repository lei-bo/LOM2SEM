import numpy as np
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from models import UNet
from utils import show_tensor_images


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EPOCHS = 100
BATCH_SIZE = 4
LR = 1e-4
DISPLAY_STEP = 149


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
        self.transform = T.Compose([T.ToTensor()])

    def __getitem__(self, index: int):
        img_name = self.img_names[index]
        img_A = self._get_image(self.folder_A, img_name)
        img_B = self._get_image(self.folder_B, img_name)
        return img_A, img_B

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
    train_set = ImagePairDataset("./data/mecs_steel", "LOM640", "SEM640", "all.txt")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    model = UNet(input_channels=3, output_channels=1, hidden_channels=32).to(DEVICE)
    recon_criterion = nn.L1Loss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    cur_step = 0
    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch+1}/{N_EPOCHS}")
        for imgs_A, imgs_B in tqdm(train_loader):
            imgs_A, imgs_B = imgs_A.to(DEVICE), imgs_B.to(DEVICE)
            preds_B = model(imgs_A)
            optimizer.zero_grad()
            loss = recon_criterion(preds_B, imgs_B[:,0,:,:][:,None,:,:])
            loss.backward()
            optimizer.step()

            if cur_step % DISPLAY_STEP == 0:
                print(f"Epoch {epoch+1}: Step {cur_step}: loss: {loss.item()}")
                show_tensor_images(torch.cat([imgs_A, imgs_B,
                                              torch.cat([preds_B]*3, dim=1)]),
                                   nrow=BATCH_SIZE, num_images=BATCH_SIZE*3)
            cur_step += 1


if __name__ == '__main__':
    train()