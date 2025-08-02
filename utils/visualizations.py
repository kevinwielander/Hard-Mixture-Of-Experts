import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize_cluster_centers(cluster_centers, save_path):
    centers_np = cluster_centers.cpu().numpy() if torch.is_tensor(cluster_centers) else cluster_centers
    pca = PCA(n_components=2)  # reduce dimension
    centers_2d = pca.fit_transform(centers_np)

    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], marker='x', color='red')
    plt.title("Cluster Centers (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(save_path)
    plt.close()
