import torch
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

AIRCRAFT_TYPES = [
    "Boeing 737", "Boeing 747", "Boeing 777",
    "Airbus A320", "Airbus A380", "Cessna",
    "Fighter jet", "Cargo plane", "Helicopter", "Unknown"
]

weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)
model.eval()

preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

def classify_crop(crop: Image.Image) -> str:
    tensor = preprocess(crop).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
    idx = out.argmax().item() % len(AIRCRAFT_TYPES)
    return AIRCRAFT_TYPES[idx]