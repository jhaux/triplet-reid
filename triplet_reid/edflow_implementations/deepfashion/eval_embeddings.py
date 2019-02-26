import sys
sys.path.append(".")
import yaml
from triplet_reid.edflow_implementations.deepfashion.data import FromCSVWithEmbedding
from tqdm import trange
import numpy as np


from triplet_reid.excluders.diagonal import Excluder as DiagonalExcluder
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score

if __name__ == "__main__":
    #embedding_root = "data/deepfashion/embeddings"
    #embedding_postfix = "_alpha.npz"

    embedding_root = "data/deepfashion/embeddings_both_df128_450000"
    embedding_postfix = "_z_cond.npy"

    query_config = {
            "spatial_size":         256,
            "data_root":            "data/deepfashion/images",
            "embedding_root":       embedding_root,
            "embedding_postfix":    embedding_postfix,
            "data_csv":       "data/deepfashion/query.csv"}
    query_dataset = FromCSVWithEmbedding(query_config)
    gallery_config = {
            "spatial_size":         256,
            "data_root":            "data/deepfashion/images",
            "embedding_root":       embedding_root,
            "embedding_postfix":    embedding_postfix,
            "data_csv":     "data/deepfashion/gallery.csv"}
    gallery_dataset = FromCSVWithEmbedding(gallery_config)
    print(len(query_dataset))
    print(len(gallery_dataset))

    # load required data
    def load(data):
        keys = ["pid", "name", "embedding"]
        loaded = dict((k, list()) for k in keys)
        for i in trange(len(data)):
            datum = data[i]
            for k in keys:
                loaded[k].append(datum[k])
        for k in keys:
            loaded[k] = np.stack(loaded[k])
        return loaded

    query_data = load(query_dataset)
    gallery_data = load(gallery_dataset)


    # evaluation as in original evaluate.py
    distances = cdist(query_data["embedding"], gallery_data["embedding"],
            metric = "euclidean")

    # Compute the pid matches
    pid_matches = gallery_data["pid"][None] == query_data["pid"][:,None]

    # Get a mask indicating True for those gallery entries that should
    # be ignored for whatever reason (same camera, junk, ...) and
    # exclude those in a way that doesn't affect CMC and mAP.
    Excluder = DiagonalExcluder
    excluder = Excluder(gallery_data["name"])
    mask = excluder(query_data["name"])
    distances[mask] = np.inf
    pid_matches[mask] = False

    # Keep track of statistics. Invert distances to scores using any
    # arbitrary inversion, as long as it's monotonic and well-behaved,
    # it won't change anything.
    aps = []
    cmc = np.zeros(len(gallery_data["pid"]), dtype=np.int32)
    scores = 1 / (1 + distances)
    for i in range(len(distances)):
        ap = average_precision_score(pid_matches[i], scores[i])

        if np.isnan(ap):
            logwarn = print
            logwarn("WARNING: encountered an AP of NaN!")
            logwarn("This usually means a person only appears once.")
            logwarn("In this case, it's because of {}.".format(query_data["name"][i]))
            logwarn("I'm excluding this person from eval and carrying on.")
            continue

        aps.append(ap)
        # Find the first true match and increment the cmc data from there on.
        k = np.where(pid_matches[i, np.argsort(distances[i])])[0][0]
        cmc[k:] += 1

    # Compute the actual cmc and mAP values
    cmc = cmc / len(query_data["pid"])
    mean_ap = np.mean(aps)

    # Print out a short summary.
    print('{} | mAP: {:.2%} | top-1: {:.2%} top-2: {:.2%} | top-5: {:.2%} | top-10: {:.2%}'.format(
        embedding_root, mean_ap, cmc[0], cmc[1], cmc[4], cmc[9]))
