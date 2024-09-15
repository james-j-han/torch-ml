import torch

from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image

# Load pretrained model - ResNet50
# model = models.resnet50(pretrained=True)
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# Set the model to evaluation mode
model.eval()

# Define the preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize(256), # Resize to 256x256
    transforms.CenterCrop(224), # Crop to 224x224
    transforms.ToTensor(), # Convert image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize
])

# Load your image
img_path = 'bootsie.jpg'
input_image = Image.open(img_path).convert('RGB')
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

# Run the model
with torch.no_grad(): # Disable gradient calculation for inference
    output = model(input_batch)

# Output is a vector of logits (raw scores), which you can convert to probabilities
probabilities = torch.nn.functional.softmax(output[0], dim=0)

import json
with open("imagenet_classes.json", "r") as f:
    labels = json.load(f)

# Find the top-5 predictions
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(labels[str(top5_catid[i].item())], top5_prob[i].item())