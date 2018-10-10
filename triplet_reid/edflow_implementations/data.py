from hbu_journal.data.ntu import PretrainDataset
from hbu_journal.data.ntugems import CachedNTUGems
import numpy as np
from edflow.data.dataset import (
        ProcessedDataset, JoinedDataset,
        ConcatenatedDataset, SubDataset,
        ExtraLabelsDataset, RandomlyJoinedDataset)
from edflow.iterators.batches import save_image
from hbu_journal.data.data import TargetProcessing

def center_crop(**kwargs):
    image = kwargs["target"]
    h,w,c = image.shape
    image = image[:,w//4:w//4+w//2,:3]
    return {"image": image}

def add_labels(data, idx):
    # reid code uses the keys
    # "image", "pid", "name", "pid"
    labels = dict()
    S = data.labels["S"][idx]
    P = data.labels["P"][idx]
    file_id = data.labels["file_id"][idx]
    labels["pid"] = "S{:03}P{:03}".format(S, P)
    labels["name"] = file_id
    # furthermore, keypoints and box are not properly prepared to be stacked,
    # replace with zero since we do not need them anyway
    labels["keypoints"] = 0
    labels["box"] = 0
    return labels


# NTUGems

def get_split_indices(data):
    splits = dict()

    # fixed training person ids
    train_person_ids = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 35}
    train_indices = [i for i in range(len(data)) if data.labels["P"][i] in train_person_ids]
    splits["train"] = train_indices

    # remaining as validation
    train_set = set(train_indices)
    val_indices = [i for i in range(len(data)) if not i in train_set]

    # for each combination of person, camera, action randomly select
    # one frame for query and ten frames for gallery
    prng = np.random.RandomState(1)
    val_pids = sorted(set(data.labels["P"][i] for i in val_indices))
    val_cids = sorted(set(data.labels["C"][i] for i in val_indices))
    val_aids = sorted(set(data.labels["A"][i] for i in val_indices))
    query_indices = list()
    gallery_indices = list()
    for pid in val_pids:
        pid_indices = [i for i in val_indices if data.labels["P"][i] == pid]
        for cid in val_cids:
            cid_indices = [i for i in pid_indices if data.labels["C"][i] == cid]
            for aid in val_aids:
                aid_indices = [i for i in cid_indices if data.labels["A"][i] == aid]
                # select query examples
                query_examples = prng.choice(aid_indices, 1, replace = False)
                query_indices += list(query_examples)
                # remove query examples
                aid_indices = [i for i in aid_indices if not i in query_examples]
                # select gallery examples
                gallery_examples = prng.choice(aid_indices, 10, replace = False)
                gallery_indices += list(gallery_examples)
    splits["query"] = query_indices
    splits["gallery"] = gallery_indices
    return splits


def get_data(config, split):
    data = CachedNTUGems()
    indices = get_split_indices(data)[split]
    data = SubDataset(data, indices)

    # apply generic preprocessing (in particular resizing and masking)
    data = ProcessedDataset(data, TargetProcessing(config.get("spatial_size", 256), 1))
    # crop images
    data = ProcessedDataset(data, center_crop)
    # add required labels
    data = ExtraLabelsDataset(data, add_labels)
    return data


def Train(config):
    data = get_data(config, "train")
    # get multiple views of same pid
    #data = JoinedDataset(data, "pid", config.get("n_views", 4))
    data = RandomlyJoinedDataset(data, "pid", config.get("n_views", 4))
    return data


def Query(config):
    data = get_data(config, "query")
    return data
    

def Gallery(config):
    data = get_data(config, "gallery")
    return data


def Eval(config):
    data = ConcatenatedDataset(Query(config), Gallery(config))
    return data


if __name__ == "__main__":
    config = {"spatial_size": 256}
    d = Train(config)

    example = d[0]
    print(example.keys())
    print(len(example["image"]))
    print(example["pid"])
    print(example["name"])

    headers = ["", "Train", "Query", "Gallery"]
    datas = [Train(config), Query(config), Gallery(config)]
    pids = ["pids"]+[len(set(d.labels["pid"])) for d in datas]
    frames = ["frames"]+[len(d) for d in datas]
    rows = [headers, pids, frames]

    row_format ="{:>15}" * (len(headers))
    for row in rows:
        print(row_format.format(*row))
