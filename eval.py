import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import math

class ValidationDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_files = sorted(
            [f for f in os.listdir(lr_dir) if os.path.isfile(os.path.join(lr_dir, f))]
        )
        self.hr_files = sorted(
            [f for f in os.listdir(hr_dir) if os.path.isfile(os.path.join(hr_dir, f))]
        )
        self.transform = transform

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])

        lr_image = Image.open(lr_path).convert("RGB")
        hr_image = Image.open(hr_path).convert("RGB")

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image

def calculate_psnr(output, target):
    mse = torch.mean((output - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(1.0 / math.sqrt(mse.item()))
    return psnr

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    valid_lr_path = "/home/msy/sr/data/DIV2K_valid_LR_bicubic/X2"
    valid_hr_path = "/home/msy/sr/data/DIV2K_valid_HR"

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    valid_dataset = ValidationDataset(valid_lr_path, valid_hr_path, transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    model = Autoencoder()
    model.load_state_dict(torch.load("autoencoder_sr.pth"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()

    total_loss = 0.0
    total_psnr = 0.0
    num_samples = 0

    with torch.no_grad():
        for lr, hr in valid_loader:
            lr, hr = lr.to(device), hr.to(device)

            output = model(lr)

            output = F.interpolate(output, size=hr.shape[2:], mode='bilinear', align_corners=False)

            loss = criterion(output, hr)
            total_loss += loss.item()

            psnr = calculate_psnr(output, hr)
            total_psnr += psnr

            num_samples += 1

    avg_loss = total_loss / num_samples
    avg_psnr = total_psnr / num_samples
    print(f"Average Loss (MSE): {avg_loss:.4f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")