import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class LowLightDataset(Dataset):
    def __init__(self, low_light_dir, high_light_dir, transform=None):
        self.low_light_images = sorted([os.path.join(low_light_dir, f) for f in os.listdir(low_light_dir)])
        self.high_light_images = sorted([os.path.join(high_light_dir, f) for f in os.listdir(high_light_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.low_light_images)

    def __getitem__(self, idx):
        low_img = Image.open(self.low_light_images[idx]).convert('RGB')
        high_img = Image.open(self.high_light_images[idx]).convert('RGB')
        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)
        return low_img, high_img

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
