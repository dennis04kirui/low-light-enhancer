import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from model import LowLightEnhancer

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = LowLightEnhancer().to(device)
state_dict = torch.load("models/epoch_50.pth", map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def enhance_image(pil_image):
    """
    Takes PIL Image and returns enhanced PIL Image
    """

    img_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced = model(img_tensor)

    enhanced = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
    enhanced = (enhanced * 255).clip(0, 255).astype(np.uint8)

    enhanced_pil = Image.fromarray(enhanced)

    return enhanced_pil
