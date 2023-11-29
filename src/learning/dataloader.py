import torch
import numpy as np
from torch.utils.data import Dataset

from ..learning.transform import FOV, transform_ic, transform_reachsetparam
debug = False


class ReachsetDataset(Dataset):
    def __init__(self, reachset: np.ndarray):
        self.reachset = reachset

    def __len__(self):
        return len(self.reachset)

    def __getitem__(self, idx):
        """Get a single data point from the dataset.

        Returns:
            data (torch.tensor): original reachset data point
            ic_ (torch.tensor): transformed initial condition
            reach_ (torch.tensor): transformed reachset
        """

        data = self.reachset[idx, :]

        # unpack data
        alt, vx, vz, m, tgo = data[:5]
        z = np.log(m)
        xmax, _ = data[5:7]
        xmin, _ = data[7:9]
        x_ymax, ymax = data[9:11]

        # --------------------
        # Process Initial Condition (IC) = Input
        # --------------------
        alt_, vx_, vz_, z_, tgo_ = transform_ic(alt, vx, vz, z, tgo)
        ic_ = np.array([alt_, vx_, vz_, z_, tgo_])

        # --------------------
        # Process Reachset = Output
        # --------------------
        xmin_, xmax_, ymax_, x_ymax_ = transform_reachsetparam(xmin, xmax, ymax, x_ymax, alt, fov=FOV)
        reach_ = np.array([xmin_, xmax_, ymax_, x_ymax_])

        # convert to torch tensor
        ic_ = torch.tensor(ic_, dtype=torch.float64)
        reach_ = torch.tensor(reach_, dtype=torch.float64)

        return data, ic_, reach_


class FeasibilityDataset(Dataset):
    def __init__(self, reachset: np.ndarray):
        self.reachset = reachset

    def __len__(self):
        return len(self.reachset)

    def __getitem__(self, idx):
        """Get a single data point from the dataset.

        Returns:
            data (torch.tensor): original reachset data point
            ic_ (torch.tensor): transformed initial condition
            feasibility (torch.tensor): feasibility
        """

        data = self.reachset[idx, :]

        # unpack data
        alt, vx, vz, m, tgo = data[:5]
        z = np.log(m)
        feasibility = data[5]

        # --------------------
        # Process Initial Condition (IC) = Input
        # --------------------
        alt_, vx_, vz_, z_, tgo_ = transform_ic(alt, vx, vz, z, tgo)
        ic_ = np.array([alt_, vx_, vz_, z_, tgo_])

        # convert to torch tensor
        ic_ = torch.tensor(ic_, dtype=torch.float64)
        feasibility = torch.tensor(feasibility, dtype=torch.float64)

        return data, ic_, feasibility
