import sys, os
import numpy as np
from edflow.iterators.batches import load_image
from edflow.iterators.batches import DatasetMixin
from edflow.iterators.batches import resize_float32 as resize
from edflow.util import PRNGMixin


def add_choices(labels):
    labels = dict(labels)
    cid_labels = np.asarray(labels["character_id"])
    cids = np.unique(cid_labels)
    cid_indices = dict()
    for cid in cids:
        cid_indices[cid] = np.nonzero(cid_labels == cid)[0]
        verbose = False
        if verbose:
            if len(cid_indices[cid]) <= 1:
                raise ValueError("No choice for {}: {}".format(cid, cid_indices[cid]))

    labels["choices"] = list()
    for i in range(len(labels["character_id"])):
        cid = labels["character_id"][i]
        choices = cid_indices[cid]
        labels["choices"].append(choices)
    return labels


def center_crop(image):
    h,w,c = image.shape
    image = np.array(image[:,w//4:w//4+w//2,:3])
    return image


class StochasticPairs(DatasetMixin, PRNGMixin):
    def __init__(self, config):
        self.size = config["spatial_size"]
        self.root = config["data_root"]
        self.csv = config["data_csv"]
        with open(self.csv) as f:
            lines = f.read().splitlines()
        self._length = len(lines)
        lines = [l.split(",", 1) for l in lines]
        self.labels = {
                "character_id": [l[0] for l in lines],
                "relative_file_path_": [l[1] for l in lines],
                "file_path_": [os.path.join(self.root, l[1]) for l in lines]}
        self.labels = add_choices(self.labels)
        self.config = config

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = load_image(image_path)
        image = resize(image, self.size)
        image = center_crop(image)
        return image

    def get_example(self, i):
        choices = self.labels["choices"][i]
        if len(choices) > 1:
            choices = [c for c in choices if c != i]
        js = self.prng.choice(choices, size = self.config["n_views"]-1)
        indices = np.concatenate([[i],js])

        output = {
                "image": [self.preprocess_image(self.labels["file_path_"][k]) for k in indices],
                "pid": [self.labels["character_id"][k] for k in indices],
                "name": [self.labels["relative_file_path_"][k] for k in indices]}
        for k in output:
            output[k] = np.stack(output[k], axis = 0)
        return output


class FromCSV(DatasetMixin, PRNGMixin):
    def __init__(self, config):
        self.size = config["spatial_size"]
        self.root = config["data_root"]
        self.csv = config["data_csv"]
        with open(self.csv) as f:
            lines = f.read().splitlines()
        self._length = len(lines)
        lines = [l.split(",", 1) for l in lines]
        self.labels = {
                "character_id": [l[0] for l in lines],
                "relative_file_path_": [l[1] for l in lines],
                "file_path_": [os.path.join(self.root, l[1]) for l in lines]}

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = load_image(image_path)
        image = resize(image, self.size)
        image = center_crop(image)
        return image

    def get_example(self, i):
        output = {
                "image": self.preprocess_image(self.labels["file_path_"][i]),
                "pid": self.labels["character_id"][i],
                "name": self.labels["relative_file_path_"][i]}
        return output


class Eval(DatasetMixin):
    def __init__(self, config):
        self.query_csv = config["data_query_csv"]
        self.gallery_csv = config["data_gallery_csv"]

        query_config = dict(config)
        query_config["data_csv"] = self.query_csv
        gallery_config = dict(config)
        gallery_config["data_csv"] = self.gallery_csv

        self.query_data = FromCSV(query_config)
        self.gallery_data = FromCSV(gallery_config)
        self._length = len(self.query_data)+len(self.gallery_data)

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = load_image(image_path)
        image = resize(image, self.size)
        image = center_crop(image)
        return image

    def get_example(self, i):
        if i < len(self.query_data):
            idx = i
            example = self.query_data.get_example(idx)
            example["dataset_index_"] = 0
        else:
            idx = i - len(self.query_data)
            example = self.gallery_data.get_example(idx)
            example["dataset_index_"] = 1
        return example