import os
import numpy as np
import torch
from torch import nn
from torchsummary import summary
from torchvision import transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

class CNN(nn.Module):
    def __init__(self, out_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1) #(in_channels, out_channels, kernel_size, stride)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2) # each time maxpooling will shrink the image n times

        # For input size 320x240, after three 2x2 maxpooling, the size will be (320/2^3) x (240/2^3) = 40x30
        self.fc1 = nn.Linear(128 * 38 * 28, 512) # Adjust the input size according to the output of conv layers
        self.fc2 = nn.Linear(512, out_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, 128 * 38 * 28) # Adjust the input size according to the output of conv layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x 

def load_image(filepath):
    image = Image.open(filepath).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((240, 320)),
        transforms.ToTensor()
    ])
    return transform(image)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(2).to(device)
    summary(model, (3, 320, 240))

    image_dir = 'Final/dataset'
    image_filenames = [f'{i:02}.bmp' for i in range(1, 11)]
    images = [load_image(os.path.join(image_dir, filename)) for filename in image_filenames]

    ssim_values = []
    psnr_values = []

    for image_tensor in images:
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
        compressed_image_tensor = model(image_tensor).detach().cpu().numpy().squeeze()  # Remove batch dimension

        print(f"Compressed image tensor shape: {compressed_image_tensor.shape}")

        # Depending on the shape, adjust the axes for transposing
        if len(compressed_image_tensor.shape) == 3:
            compressed_image = compressed_image_tensor.transpose((1, 2, 0))
        else:
            compressed_image = compressed_image_tensor  # Handle case if shape is already 2D

        original_image = image_tensor.squeeze().cpu().numpy().transpose((1, 2, 0))

        ssim_value = ssim(original_image, compressed_image, multichannel=True)
        psnr_value = psnr(original_image, compressed_image)

        ssim_values.append(ssim_value)
        psnr_values.append(psnr_value)

        print(f'SSIM: {ssim_value}, PSNR: {psnr_value}')

    avg_ssim = np.mean(ssim_values)
    avg_psnr = np.mean(psnr_values)

    print(f'Average SSIM: {avg_ssim}, Average PSNR: {avg_psnr}')
