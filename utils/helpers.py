import torch
import yaml


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)