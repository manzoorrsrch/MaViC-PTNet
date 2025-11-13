from pathlib import Path
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

class MultiViewDataset(Dataset):
    def __init__(self, root, split="train", classes=None,
                 img_size=256, num_views=3, exclude_set=None):
        self.root = Path(root) / split
        self.classes = classes or CLASSES
        self.img_size = img_size
        self.num_views = num_views
        self.exclude = exclude_set or set()

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

        self.samples = []
        for ci, c in enumerate(self.classes):
            cdir = self.root / c
            if not cdir.exists():
                continue
            for n in os.listdir(cdir):
                if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    p = str(cdir / n)
                    if p in self.exclude:
                        continue
                    self.samples.append((p, ci))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        views = []
        for _ in range(self.num_views):
            x = self.transform(img)
            views.append(x)
        mv = torch.stack(views, dim=0)  # [V,3,H,W]
        return mv, y, p
