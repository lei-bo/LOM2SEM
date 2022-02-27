import numpy as np
import torchvision.transforms as T
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from utils import show_tensor_images


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

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_A = self._get_image(self.folder_A, img_name)
        img_B = self._get_image(self.folder_B, img_name)
        return img_A, img_B

    def _get_image(self, folder, img_name):
        img_path = f'{folder}/{img_name}'
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_names)

if __name__ == '__main__':
    dataset = ImagePairDataset("./data/mecs_steel", "LOM640", "SEM640", "all.txt")
    img_A, img_B = dataset[0]
    show_tensor_images([img_A, img_B], num_images=4)