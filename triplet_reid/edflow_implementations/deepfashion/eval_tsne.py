import sys
sys.path.append(".")
import yaml, os, json
from triplet_reid.edflow_implementations.deepfashion.data import (
        FromCSVWithEmbedding, FromCSVWithMultiEmbedding)
from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt


from triplet_reid.excluders.diagonal import Excluder as DiagonalExcluder
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score


def make_tsne_plot(outpath, dataset):
    indices = np.random.permutation(len(dataset))
    N = 1000
    indices = indices[:N]
    data = list()
    for i in tqdm(indices):
        data.append(dataset[i]["embedding"])
    data = np.stack(data)

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=0, verbose = 1, perplexity = 40, n_iter=300)
    data_2d = tsne.fit_transform(data)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(data_2d[:,0], data_2d[:,1])

    fig.savefig(outpath, dpi = 300)
    print("Wrote ", outpath)

def make_combined_tsne_plot(outpath, dataset1, dataset2, label1, label2):
    indices1 = np.random.permutation(len(dataset1))
    indices2 = np.random.permutation(len(dataset2))
    N = 1000
    indices1 = indices1[:N]
    indices2 = indices2[:N]
    data = list()
    for i in tqdm(indices1):
        data.append(dataset1[i]["embedding"])
    for i in tqdm(indices2):
        data.append(dataset2[i]["embedding"])
    data = np.stack(data)
    print(data.shape)

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=0, verbose = 1)
    data_2d = tsne.fit_transform(data)
    print(data_2d.shape)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    colors = ["r", "g"]
    markers = ["+", "x"]
    alphas = [1.0, 1.0]
    ax.scatter(
            data_2d[:N,0], data_2d[:N,1],
            c = colors[0], label = label1, marker = markers[0], alpha = alphas[0])
    ax.scatter(
            data_2d[N:,0], data_2d[N:,1],
            c = colors[1], label = label2, marker = markers[1], alpha = alphas[1])
    ax.legend()

    fig.savefig(outpath, dpi = 300)
    print("Wrote ", outpath)


def run(embedding_root, postfixes):
    joint_config = {
            "spatial_size":         256,
            "data_root":            "data/deepfashion/images",
            "embedding_root":       embedding_root,
            "embedding_postfixes":  postfixes,
            "data_csv":             "data/deepfashion/test_reconstruction.csv",
            "z_size":               None}
    joint_dataset = FromCSVWithMultiEmbedding(joint_config)
    marginal_config = {
            "spatial_size":         256,
            "data_root":            "data/deepfashion/images",
            "embedding_root":       embedding_root,
            "embedding_postfixes":  postfixes,
            "data_csv":             "data/deepfashion/test_transfer.csv",
            "z_size":               None}
    marginal_dataset = FromCSVWithMultiEmbedding(marginal_config)
    print(len(joint_dataset))
    print(len(marginal_dataset))
    for name, dataset in zip(["joint", "marginal"], [joint_dataset, marginal_dataset]):
        out_path = "tsne_" + name + ".png"
        out_path = os.path.join(embedding_root, out_path)
        make_tsne_plot(out_path, dataset)

    out_path = "tsne_" + "combined" + ".png"
    out_path = os.path.join(embedding_root, out_path)
    make_combined_tsne_plot(out_path, joint_dataset, marginal_dataset, "joint", "marginal")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("embedding_root")
    parser.add_argument("--postfixes", nargs = "+", required = True)
    opt = parser.parse_args()
    run(opt.embedding_root, opt.postfixes)
