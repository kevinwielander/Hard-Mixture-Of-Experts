import yaml
from dataloader import CustomDataset


def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    dataset = CustomDataset(config)
    print(f"Dataset size: {len(dataset)}")

    img_tensor, class_id = dataset[0]
    print(f"Image Tensor Shape: {img_tensor.shape}, Class ID: {class_id}")


if __name__ == "__main__":
    main()