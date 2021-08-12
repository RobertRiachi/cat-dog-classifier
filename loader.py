import os
import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, root, file, transforms=None):
        self.root = root
        self.data = pd.read_csv(file, header=None)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_path = os.path.join(self.root, self.data.iloc[index, 0])
        img = io.imread(file_path)
        target = torch.tensor(int(self.data.iloc[index,1]))

        if self.transforms:
            img = self.transforms(img)

        return (img, target)

if __name__ == "__main__":
    root = "/mnt/c/Users/Robert/Downloads/cats-vs-dogs/train"
    file = "/mnt/c/Users/Robert/Downloads/cats-vs-dogs/train_data.csv"
    img_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        ])

    dataset = CustomDataset(root, file, transforms=img_transforms)

    t = dataset.__getitem__(0)[0]

    from torchvision.utils import save_image

    print(t.shape)
    save_image(t, "/mnt/c/Users/Robert/Downloads/cats-vs-dogs/t.jpg")