import torch
from torchvision import models
import torch.nn as nn

from models.gater import Gater

input_weights_path = "../artifacts/gater.pth"
output_gater_path = "../artifacts/gater_feature_extractor.pth"


class Identity(nn.Module):
    def forward(self, x):
        return x


model = Gater(num_classes=7)
model.load_state_dict(torch.load(input_weights_path, map_location="mps"))
model.fc = Identity()

torch.save(model.state_dict(), output_gater_path)

print(f"Saved modified model (feature extractor) to {output_gater_path}")
