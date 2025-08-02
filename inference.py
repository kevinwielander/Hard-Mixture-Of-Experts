import torch
from torch.utils.data import DataLoader
from models.gater import Gater
from dataloader import CustomDataset
from utils.helpers import get_device, load_config


def inference(model_path):
    configuration = load_config()
    dataset = CustomDataset(configuration, is_train=False)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    num_classes = len(dataset.class_map)
    device = get_device()

    model = Gater(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(dataloader):
            isinstance(imgs, torch.Tensor)
            imgs = imgs.to(device, dtype=torch.float32)
            labels = labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += imgs.size(0)
            correct += (predicted == labels.squeeze()).sum().item()

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    config = load_config()
    inference(model_path=config['training']['best_model_path'] + config['training']['model_name'] + '.pth',)
