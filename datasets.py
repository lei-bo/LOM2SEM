import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class ImagePairDataset(Dataset):
    """A dataset consists of pairs of images. Images in the same pair come from
    different folders with the same name.

    Args:
        root: Root directory path.
        size: The size of input after resizing.
        folder_A: Folder of image type A.
        folder_A: Folder of image type B.
        split: A text file listing the name of images.
    """
    def __init__(self, root: str, size: tuple, folder_A: str, folder_B: str, split: str):
        super(ImagePairDataset, self).__init__()
        self.img_names = np.loadtxt(f"{root}/split/{split}",
                                    dtype=str, delimiter='\n', ndmin=1)
        self.folder_A = f"{root}/{folder_A}"
        self.folder_B = f"{root}/{folder_B}"
        self.transform = T.Compose([T.Resize(size),
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