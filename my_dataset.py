from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class MyDataSet(Dataset):

    def __init__(self, images_path1: list, images_path2: list, images_target: list, transform=None):
        self.images_path1 = images_path1
        self.images_path2 = images_path2
        self.images_target = images_target
        self.transform = transform

    def __len__(self):
        return len(self.images_path1)

    def __getitem__(self, item):
        img1 = Image.open(self.images_path1[item])
        img2 = Image.open(self.images_path2[item])
        target = Image.open(self.images_target[item])

        #img1, img2, target = map(TF.to_tensor, (img1, img2, target))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            target = self.transform(target)

        return img1, img2, target
