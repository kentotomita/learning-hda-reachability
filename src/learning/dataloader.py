import torch
import numpy as np
from torch.utils.data import Dataset

from ..learning.transform import FOV, transform_ic, transform_reachsetparam

debug = False
class ReachsetDataset(Dataset):
    def __init__(self, reachset: dict):

        self.reachset = reachset

    def __len__(self):
        return len(self.reachset)
    
    def __getitem__(self, idx):
        """Get a single data point from the dataset.
        
        Returns:
            ic_ (torch.tensor): transformed initial condition
            xs (torch.tensor): transformed reachset x
            ys (torch.tensor): transformed reachset y
            fov_radius (torch.tensor): transformed fov radius
            feasible (bool): whether the reachset exists
        """

        data = self.reachset[idx]

        # unpack data
        x0 = data['x0']
        tgo = data['tgo'][0]
        xs = data['rf'][:, 0]
        ys = data['rf'][:, 1]

        alt = x0[2]
        vx = x0[3]
        vz = x0[5]
        z = x0[6]

        # --------------------
        # Process Initial Condition (IC) = Input
        # --------------------
        alt_, vx_, vz_, z_, tgo_ = transform_ic(alt, vx, vz, z, tgo)
        ic_ = np.array([alt_, vx_, vz_, z_, tgo_])

        # some parameters
        fov_radius = alt * np.tan(FOV/2)

        # --------------------
        # Process Reachset = Output
        # --------------------

        # check feasibility
        if np.all(np.isnan(xs)):
            feasible = 0
            xmin = np.nan
            xmax = np.nan

        else:
            feasible = 1

            # replace nan with random sample
            nan_idx = np.isnan(xs) | np.isnan(ys)
            n_nan = np.sum(nan_idx)
            if n_nan > 0:
                # random sample from non-nan data
                idx = np.random.choice(np.where(~nan_idx)[0], n_nan)
                # replace nan with random sample
                xs[nan_idx] = xs[idx]
                ys[nan_idx] = ys[idx]
        
            # sort xs and ys by xs
            idx = np.argsort(xs)
            xs = xs[idx]
            ys = ys[idx]
            
            assert not np.isnan(xs).any(), f"xs contains nan: {xs}"
            assert not np.isnan(ys).any(), f"ys contains nan: {ys}"

            if debug and np.max(ys) < 1e-2:
                print('x0: ', x0)
                print('tgo: ', tgo)
                print('xs: ', xs)
                print('ys: ', ys)

            xmin = np.min(xs)
            xmax = np.max(xs)

            
        # convert to torch tensor
        ic_ = torch.tensor(ic_, dtype=torch.float64)
        xs = torch.tensor(xs, dtype=torch.float64)
        ys = torch.tensor(ys, dtype=torch.float64)
        xmin = torch.tensor(xmin, dtype=torch.float64)
        xmax = torch.tensor(xmax, dtype=torch.float64)
        fov_radius.double()
        feasible = torch.tensor(feasible, dtype=torch.bool)
        
        return ic_, xs, ys, xmin, xmax, alt, fov_radius, feasible
    


        
