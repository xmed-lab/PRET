import os
import torch
from PIL import Image
from torchvision import datasets
from torchvision import transforms as pth_transforms


class AutoPad(torch.nn.Module):

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        w, h = img.size[0], img.size[1]
        pad_h = self.size - h
        pad_w = self.size - w
        pad = pth_transforms.Pad((0, 0, pad_w, pad_h), fill=255)
        return pad(img)


class ImagePyramidDataset(datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        target_downsamples=[[0, 'x20']],
        transform = None,
        outd = ''):
        super().__init__(root, transform=transform)

        self.transform = transform

        self.samples = []
        files = os.listdir(root)
        for f in files:
            if not os.path.isdir(os.path.join(root, f)):
                continue
            dirs = os.listdir(os.path.join(root, f))
            dirs = sorted([int(_) for _ in dirs if ".xml" not in _], reverse=True)
            for ds in target_downsamples:
                in_path = os.path.join(root, f, str(dirs[ds[0]]))
                out_path = os.path.join(f, ds[1])
                for img_name in os.listdir(in_path):
                    if '.npy' in img_name:
                        continue
                    # skip background patches by file size 5000
                    if ds[1] == 'x20' and os.path.getsize(os.path.join(in_path, img_name)) < 5000:
                        continue
                    if outd != '' and os.path.exists(os.path.join(outd, out_path)):
                        continue
                    self.samples.append([os.path.join(in_path, img_name), out_path, \
                        os.path.join(out_path, img_name.replace('.jpeg', '.npy'))])

    def __getitem__(self, index: int):

        in_path, out_dir, out_path = self.samples[index]
        if '.npy' in in_path:
            return self.__getitem__(index + 1)
        img = Image.open(in_path).convert("RGB")
        img = self.transform(img)
        return img, out_dir, out_path

