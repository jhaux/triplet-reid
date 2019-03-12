import sys
sys.path.append(".")
import yaml, os, json
from triplet_reid.edflow_implementations.deepfashion.data import FromCSVWithEmbedding
from tqdm import trange
import numpy as np


from triplet_reid.excluders.diagonal import Excluder as DiagonalExcluder
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score


def evaluate(query_dataset, gallery_dataset, embedding_root, embedding_postfix, n_retrievals = 10):
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
    info = '{}/**/*{} | mAP: {:.2%} | top-1: {:.2%} top-2: {:.2%} | top-5: {:.2%} | top-10: {:.2%}'.format(
        embedding_root, embedding_postfix, mean_ap, cmc[0], cmc[1], cmc[4], cmc[9])
    print(info)

    # Retrieval data for easy plotting
    cutoff = 1000
    retrievals = np.argsort(distances, axis = 1)
    print(distances.shape)
    print(retrievals.shape)
    retrievals = np.concatenate(
            [retrievals[:, :n_retrievals], retrievals[:, -n_retrievals-cutoff:-cutoff]], axis = 1)
    query_names = query_data["name"]
    retrieval_names = [
            [gallery_data["name"][retrieve_idx] for retrieve_idx in retrievals[query_idx]]
            for query_idx in range(retrievals.shape[0])]
    retrieval_match = [
            [pid_matches[query_idx, retrieve_idx] for retrieve_idx in retrievals[query_idx]]
            for query_idx in range(retrievals.shape[0])]
    retrieval_data = {
            "query_names": query_names,
            "retrieval_names": retrieval_names,
            "retrieval_match": retrieval_match}
    retrieval_data["info"] = {
            "mAP": mean_ap,
            "cmc": cmc}

    return info, retrieval_data


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def run(embedding_root, embedding_postfix, z_size, n_retrievals = 10):
    query_config = {
            "spatial_size":         256,
            "data_root":            "data/deepfashion/images",
            "embedding_root":       embedding_root,
            "embedding_postfix":    embedding_postfix,
            "data_csv":             "data/deepfashion/query.csv",
            "z_size":               z_size}
    query_dataset = FromCSVWithEmbedding(query_config)
    gallery_config = {
            "spatial_size":         256,
            "data_root":            "data/deepfashion/images",
            "embedding_root":       embedding_root,
            "embedding_postfix":    embedding_postfix,
            "data_csv":             "data/deepfashion/gallery.csv",
            "z_size":               z_size}
    gallery_dataset = FromCSVWithEmbedding(gallery_config)
    print(len(query_dataset))
    print(len(gallery_dataset))
    info, retrieval_data = evaluate(query_dataset, gallery_dataset, embedding_root, embedding_postfix, n_retrievals)
    out_path = "retrieval_data" + embedding_postfix + ".json"
    out_path = os.path.join(embedding_root, out_path)
    with open(out_path, "w") as f:
        json.dump(retrieval_data, f, cls = NumpyEncoder)
    print("Wrote {}".format(out_path))


def old():
    #embedding_root = "data/deepfashion/embeddings"
    #embedding_postfix = "_alpha.npz"

    #embedding_root = "data/deepfashion/embeddings_both_df128_450000"
    embedding_root = "data/deepfashion/embeddings_first_stage_450000"
    #embedding_postfix = "_z_cond.npy"
    embedding_postfix = "_z_posterior_parameters.npy"

    z_size = None
    if embedding_postfix == "_z_posterior_parameters.npy":
        z_size = 128

    run(embedding_root, embedding_postfix, z_size)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("embedding_root")
    parser.add_argument("embedding_postfix")
    parser.add_argument("--z_size", default = None)
    parser.add_argument("--n_retrievals", default = 10)
    opt = parser.parse_args()
    run(opt.embedding_root, opt.embedding_postfix, opt.z_size, opt.n_retrievals)
