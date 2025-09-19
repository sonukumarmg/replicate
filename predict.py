import torch
import torch.nn as nn
from torchvision import models, transforms
from timm.models.vision_transformer import PatchEmbed
from mamba import Mamba   # assumes you have mamba.py locally
from PIL import Image
import json

# Correct imports for Cog
from cog import BasePredictor, Input, Path

# This is your exact model architecture
class CNNMambaClassifier(nn.Module):
    def __init__(self, num_classes, cnn_backbone='resnet50', mamba_dim=256, mamba_layers=2):
        super().__init__()
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        cnn_out_channels = 2048
        self.patch_embed = PatchEmbed(img_size=7, patch_size=1, in_chans=cnn_out_channels, embed_dim=mamba_dim)
        self.mamba = nn.Sequential(*[Mamba(d_model=mamba_dim, d_state=16) for _ in range(mamba_layers)])
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(mamba_dim, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.patch_embed(x)
        x = self.mamba(x)
        x = x.transpose(1, 2)
        x = self.avg_pool(x).squeeze(-1)
        return self.classifier(x)

class Predictor(BasePredictor):
    def setup(self):
        """This method is called once when the container starts."""
        print("Loading model...")

        # Load the checkpoint file
        model_filename = 'model.pth'
        checkpoint = torch.load(model_filename, map_location=torch.device('cpu'))

        # Load class names and create the model
        self.class_names = checkpoint['class_names']
        num_classes = len(self.class_names)
        self.model = CNNMambaClassifier(num_classes=num_classes)

        # Load the weights and set to evaluation mode
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()

        # Define the image transformations
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print("Model loaded successfully.")

    def predict(self, image: Path = Input(description="Image of the crop leaf to classify")) -> str:
        """This method is called for each prediction."""

        # Open and process the input image
        img = Image.open(image).convert("RGB")
        image_tensor = self.transforms(img).unsqueeze(0)

        # Make a prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Format the output as a JSON string
        confidences = {self.class_names[i]: float(probabilities[i]) for i in range(len(self.class_names))}

        # Find the top prediction
        top_prediction = max(confidences, key=confidences.get)

        # Replicate expects a simple output, so we return a JSON string of all confidences
        return json.dumps(confidences)
