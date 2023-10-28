import sys
import numpy as np

sys.path.append("../")
from src.reachset import read_reachset


def main():
    _, _, reachset = read_reachset("../out/20230803_021318/reachset.json")

    n_infeas = 0
    for data in reachset:
        rf = np.array(data["rf"])
        if np.all(np.isnan(rf)):
            n_infeas += 1

    print("Number of data points: ", len(reachset))
    print("Number of infeasible points: ", n_infeas)


if __name__ == "__main__":
    main()
