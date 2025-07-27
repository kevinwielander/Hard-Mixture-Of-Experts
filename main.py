import yaml
from dataloader import CustomDataset


def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    dataset = CustomDataset(config)


if __name__ == "__main__":
    main()