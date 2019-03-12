import os, sys, argparse, json

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("retrieval_data")
    parser.add_argument("name")
    parser.add_argument("out_path")
    parser.add_argument("--n_nearest", default = None, type = int)
    parser.add_argument("--show_farthest", default = True, action = "store_true")
    opt = parser.parse_args()
    embedding_name = opt.name

    with open(opt.retrieval_data, "r") as f:
        retrieval_data = json.load(f)
    root = opt.data_root

    n_retrievals = len(retrieval_data["retrieval_names"][0])
    #n_retrievals = 5
    if opt.n_nearest is not None:
        assert 2*opt.n_nearest < n_retrievals
        if opt.show_farthest:
            n_retrievals = 2*opt.n_nearest
        else:
            n_retrievals = opt.n_nearest
    if opt.show_farthest:
        assert n_retrievals % 2 == 0
        n_nearest = int(n_retrievals / 2)
    else:
        n_nearest = n_retrievals

    n_rows = 1 + 10
    n_cols = 1 + n_retrievals
    ar = n_cols / n_rows
    figheight = 10.0
    figwidth = figheight*ar

    fig = plt.figure(figsize = (figwidth, figheight))
    gs = fig.add_gridspec(nrows = n_rows, ncols = n_cols)

    ax = fig.add_subplot(gs[0,0])
    ax.text(0.5, 0.5,
            'query',
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    if not opt.show_farthest:
        ax = fig.add_subplot(gs[0,1:])
        ax.text(0.5, 0.5,
                'retrieval based on\n{}'.format(embedding_name),
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax = fig.add_subplot(gs[0,1:1+n_nearest])
        ax.text(0.5, 0.5,
                'nearest retrieval based on\n{}'.format(embedding_name),
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[0,1+n_nearest:])
        ax.text(0.5, 0.5,
                'farthest retrieval based on\n{}'.format(embedding_name),
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    prng = np.random.RandomState(1)
    query_indices = prng.choice(len(retrieval_data["query_names"]), size = n_rows - 1)

    for i in range(n_rows-1):
        i_axis = i + 1
        i_query = query_indices[i]

        path = retrieval_data["query_names"][i_query]
        path = os.path.join(root, path)
        I = plt.imread(path)
        ax = fig.add_subplot(gs[i_axis,0])
        ax.imshow(I)
        ax.set_xticks([])
        ax.set_yticks([])

        for j in range(n_cols-1):
            if opt.show_farthest and j >= n_nearest:
                j_retrieval = j - 2*n_nearest
            else:
                j_retrieval = j
            path = retrieval_data["retrieval_names"][i_query][j_retrieval]
            path = os.path.join(root, path)
            I = plt.imread(path)

            j_axis = j + 1
            ax = fig.add_subplot(gs[i_axis,j_axis])
            ax.imshow(I)
            ax.set_xticks([])
            ax.set_yticks([])

            match = retrieval_data["retrieval_match"][i_query][j_retrieval]
            if match:
                color = "green"
            else:
                color = "red"
            for which in ["bottom", "top", "right", "left"]:
                ax.spines[which].set_color(color)

    gs.tight_layout(figure = fig, pad = 0)
    path = opt.out_path
    fig.savefig(path, dpi = 300)
