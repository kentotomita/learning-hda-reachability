"""Sample reachset data and save to json file."""

import os
import sys
import numpy as np
import json

sys.path.append("../")
from src.reachset import read_reachset


if __name__ == "__main__":
    data_dir = "../out/20230805_133102"
    fpath = os.path.join(data_dir, "reachset.json")

    n_sample = 1000

    with open(fpath, "r") as f:
        reachset = json.load(f)

    reachset_data = reachset["data"]

    print("Number of data points: ", len(reachset_data))

    # sample data
    idxs = np.random.choice(len(reachset_data), n_sample, replace=False)
    reachset_data = [reachset_data[i] for i in idxs]

    # save sampled data
    reachset["data"] = reachset_data
    fpath = os.path.join(data_dir, f"reachset_sample_{n_sample}.json")
    with open(fpath, "w") as f:
        json.dump(reachset, f, indent=4)

    print("Sampled data saved to {}".format(fpath))
