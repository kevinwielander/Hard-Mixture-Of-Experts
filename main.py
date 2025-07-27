import yaml
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import wandb
from dataloader import CustomDataset
from models.base_network import BasicNet


def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main():
    config = load_config()
    wandb.init(project=config['training']['wandb_project'], config=config)

    dataset = CustomDataset(config)
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    num_classes = len(dataset.class_map)
    model = BasicNet(num_classes)
    device = get_device()
    print(f"Using device: {device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

    for epoch in range(config['training']['epochs']):
        model.train()
        for imgs, labels in dataloader:
            imgs = imgs.to(device, dtype=torch.float32)
            labels = labels.squeeze().to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss.item(), "epoch": epoch + 1})
        print(f"Epoch {epoch + 1} done, last batch loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
