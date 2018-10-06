from hbu_journal.data.ntu import PretrainDataset
import numpy as np
from edflow.data.dataset import (
        ProcessedDataset, JoinedDataset,
        ConcatenatedDataset, SubDataset)
from edflow.iterators.batches import save_image


def center_crop(**kwargs):
    image = kwargs["target"]
    h,w,c = image.shape
    image = image[:,w//4:w//4+w//2,:3]
    return {"image": image}


def PretrainReidNTU(config):
    # base dataset
    ntu = PretrainDataset(config)

    # center crop
    ntu = ProcessedDataset(ntu, center_crop)

    # get multiple views of same pid
    dataset = JoinedDataset(ntu, "pid", config.get("n_views", 4))

    return dataset


def QueryGalleryNTU(config):
    n_examples = 100
    prng = np.random.RandomState(1)
    query = PretrainDataset(config)
    query = SubDataset(query, prng.choice(len(query), n_examples))
    gallery = PretrainDataset(config)
    gallery = SubDataset(gallery, prng.choice(len(gallery), n_examples))

    data = ConcatenatedDataset(query, gallery)
    data = ProcessedDataset(data, center_crop)
    return data
    

if __name__ == "__main__":
    d = PretrainReidNTU({"spatial_size": 256})

    example = d[0]
    print(example.keys())
    print(example["image"].shape)
    print(example["pid"].shape)
    print(example["image_0"].shape)
    
    for i in range(4):
        save_image(example["image_{}".format(i)], "tmp_{}.png".format(i))
