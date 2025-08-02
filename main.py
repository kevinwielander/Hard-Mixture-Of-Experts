import yaml
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import wandb
from dataloader import CustomDataset
from models.gater import Gater
from utils.helpers import get_device, load_config


def main():
    config = load_config()
    wandb.init(project=config['training']['wandb_project'], config=config)

    dataset = CustomDataset(config)
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    num_classes = len(dataset.class_map)
    model = Gater(num_classes)
    device = get_device()
    print(f"Using device: {device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

    best_loss = float('inf')
    best_model_path = config['training']['best_model_path'] + config['training']['model_name'] + '.pth'

    patience = config['training'].get('patience', 5)
    min_delta = config['training'].get('min_delta', 0.0)
    epochs_no_improve = 0

    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0.0
        for imgs, labels in dataloader:
            imgs = imgs.to(device, dtype=torch.float32)
            labels = labels.squeeze().to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            wandb.log({"loss": loss.item(), "epoch": epoch + 1})
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} done, avg loss: {avg_loss:.4f}")

        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with loss: {best_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    wandb.save(best_model_path)


if __name__ == "__main__":
    main()
