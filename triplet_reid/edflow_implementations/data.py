from hbu_journal.data.ntu import PretrainDataset
import numpy as np
from edflow.data.dataset import ProcessedDataset, JoinedDataset
from edflow.iterators.batches import save_image


def center_crop(**kwargs):
    image = kwargs["target"]
    h,w,c = image.shape
    image = image[:,w//4:w//4+w//2,:]
    return {"image": image}


def stack(**kwargs):
    r = dict()
    for k in ["image", "pid"]:
        k_keys = kwargs[k]
        k_values = [kwargs[k_key] for k_key in k_keys]
        r[k] = np.stack(k_values, axis = 0)
    return r


def PretrainReidNTU(config):
    # base dataset
    ntu = PretrainDataset(config)

    # center crop
    ntu = ProcessedDataset(ntu, center_crop)

    # get multiple views of same pid
    dataset = JoinedDataset(ntu, "pid", config.get("n_views", 4))

    # stack images and pid
    dataset = ProcessedDataset(dataset, stack)

    return dataset
    

if __name__ == "__main__":
    d = PretrainReidNTU({"spatial_size": 256})

    example = d[0]
    print(example.keys())
    print(example["image"].shape)
    print(example["pid"].shape)
    print(example["image_0"].shape)
    
    for i in range(4):
        save_image(example["image_{}".format(i)], "tmp_{}.png".format(i))
