import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request
import json

# Load pretrained model
model = models.resnet50(weights="DEFAULT")
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load ImageNet class names
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

with urllib.request.urlopen(url) as response:
    classes = response.read().decode("utf-8").splitlines()

# Ask for image
image_path = input("Enter image path: ")

# Load image
image = Image.open(image_path).convert("RGB")

# Preprocess
image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)

# Prediction
with torch.no_grad():
    output = model(image_tensor)

# Get probabilities
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Get top 5 predictions
top5 = torch.topk(probabilities, 5)

print("\nPredictions:\n")

for probability, index in zip(top5.values, top5.indices):
    print(
        f"{classes[index]}: "
        f"{probability.item() * 100:.2f}%"
    )