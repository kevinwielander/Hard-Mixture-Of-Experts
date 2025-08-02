import pickle

import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from dataloader import CustomDataset
from models.gater import GaterFeatureExtractor
from utils.helpers import load_config, get_device
from utils.visualizations import visualize_cluster_centers


def extract_features(model, dataloader, device):
    features = []
    for imgs, _ in dataloader:
        imgs = imgs.to(device, dtype=torch.float32)
        with torch.no_grad():
            feats = model.resnet(imgs)
            features.append(feats.cpu())
    return torch.cat(features)


def cluster_features(features, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(features.numpy())
    return torch.tensor(kmeans.cluster_centers_), kmeans


def assign_cluster(model, x, cluster_centers, device):
    x = x.to(device, dtype=torch.float32)
    with torch.no_grad():
        z = model.resnet.fc = torch.nn.Identity()
        feat = model.resnet(x).cpu()
    dists = torch.norm(feat - cluster_centers, dim=1)
    idx = torch.argmin(dists)
    one_hot = torch.zeros(cluster_centers.size(0))
    one_hot[idx] = 1
    return one_hot


def save_kmeans(cluster_centers, centers_path, kmeans_model):
    centers_path = centers_path + "kmeans_centers.pth"
    torch.save(cluster_centers, centers_path)
    with open(centers_path+"kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans_model, f)


def main():
    config = load_config()
    dataset = CustomDataset(config, is_train=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    device = get_device()
    model = GaterFeatureExtractor(num_classes=len(dataset.class_map))
    model.to(device)

    features = extract_features(model, dataloader, device)
    cluster_centers, kmeans_model = cluster_features(features, k=2)

    save_kmeans(cluster_centers, config['training']['best_model_path'], kmeans_model)

    print("Cluster centers:", cluster_centers)
    print("KMeans model:", kmeans_model)

    visualize_cluster_centers(cluster_centers, save_path=config['training']['best_model_path'] + "cluster_centers.png")


if __name__ == "__main__":
    main()
