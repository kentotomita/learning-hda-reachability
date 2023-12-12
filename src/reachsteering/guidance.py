import numpy as np
from torch.nn import Module
import pygmo as pg
from pygmo_plugins_nonfree import snopt7
import cvxpy as cp
import time
import sys
sys.path.append('../')
from .problems import ReachSteeringCtrl
from .objectives import ic2mean_safety_npy
from ..lcvx import LCvxMinFuel, get_vars
from ..landers import Lander
from ..safetymaps import SafetyMap


class HdaGuidance:
    """Base class for HDA guidance."""
    def __init__(self, lander: Lander, sfmap_model: SafetyMap):
        self.lander = lander
        self.sfmap_model = sfmap_model
        self.t_list = []
        self.X_list = []
        self.U_list = []
        self.sfmap_list = []

    def construct_trajectory(self):
        """Construct trajectory from the solution."""
        X = []
        U = []
        t = []
        for i in range(len(self.t_list)):
            if i < len(self.t_list) - 1:
                t_arr = self.t_list[i]
                X_arr = self.X_list[i]
                U_arr = self.U_list[i]
                leg_mask = t_arr < self.t_list[i+1][0]
                t.extend(t_arr[leg_mask])
                X.extend(X_arr[leg_mask, :])
                U.extend(U_arr[leg_mask[:-1], :])
            else:
                t.extend(self.t_list[i])
                X.extend(self.X_list[i])
                U.extend(self.U_list[i])

        return np.array(t), np.array(X), np.array(U)

    def solve_hda(self, x0: np.ndarray, tof: float, T: float, dt: float):
        """Solve HDA guidance.

        Args:
            x0 (np.ndarray): initial condition; [rx, ry, rz, vx, vy, vz, m]
            tof (float): time-of-flight
            T (float): safety map update period
            dt (float): time step

        Returns:
            t_list (List[np.ndarray]): list of time sequence; each element is a time sequence for a single leg; shape (N,).
            X_list (List[np.ndarray]): list of state sequence; each element is a state sequence for a single leg; shape (N, 7); each row is [rx, ry, rz, vx, vy, vz, m].
            U_list (List[np.ndarray]): list of control sequence; each element is a control sequence for a single leg; shape (N, 3); each row is [ux, uy, uz].
            sfmap_list (List[np.ndarray]): list of safety map; each element is a safety map for a single leg; shape (N, M, 3); each row is [x, y, safety].
        """

        self.sfmap_list.append(self.sfmap_model.get_sfmap(x0[2]))
        t0 = 0.0
        tgo = tof
        while tgo > T:
            print(f"x0: {x0}, tgo: {tgo}")
            t, X, U, sfmap_next, x0_next = self.solve_single_leg(x0, t0, tgo, T, dt)
            self.t_list.append(t)
            self.X_list.append(X)
            self.U_list.append(U)
            self.sfmap_list.append(sfmap_next)
            x0 = np.copy(x0_next)
            tgo -= T
            t0 += T
    
    def solve_single_leg(self, x0: np.ndarray, t0: float, tgo: float, T: float, dt: float):
        """Solve HDA guidance for a single leg.

        Args:
            x0 (np.ndarray): initial condition; [rx, ry, rz, vx, vy, vz, m]
            tgo (float): time-to-go
            T (float): time horizon
            dt (float): time step

        Returns:
            t (np.ndarray): time sequence; shape (N,)
            X (np.ndarray): state sequence; shape (N, 7); each row is [rx, ry, rz, vx, vy, vz, m]
            U (np.ndarray): control sequence; shape (N, 3); each row is [ux, uy, uz]
            sfmap (np.ndarray): safety map; shape (N, M, 3); each row is [x, y, safety].
            x0_next (np.ndarray): initial condition for the next leg; [rx, ry, rz, vx, vy, vz, m]
        """
        raise NotImplementedError

    
class HdaGreedy(HdaGuidance):
    """Greedy HDA guidance."""

    def __init__(self, lander: Lander, sfmap_model: SafetyMap, nn_reach: Module, border_sharpness: float = 10.0):
        """Initialize greedy HDA guidance.

        Args:
            lander (Lander): lander model
            sfmap_model (SafetyMap): safety map model
            nn_reach (Module): neural network model for reachability prediction
        """
        super().__init__(lander, sfmap_model)
        self.border_sharpness = border_sharpness

        nn_reach.eval()
        self.nn_reach = nn_reach
        self.target_list = []
        self.mean_safety_list = []
        self.reachmask_list = []

    def solve_single_leg(self, x0: np.ndarray, t0: float, tgo: float, T: float, dt: float):
        start = time.time()

        # get safety map
        sfmap = self.sfmap_list[-1]

        # get reachability set 
        mean_safety, soft_mask, safest_point = ic2mean_safety_npy(self.lander, x0, tgo, self.nn_reach, sfmap, self.border_sharpness, return_safest_point=True)
        self.mean_safety_list.append(mean_safety)
        self.reachmask_list.append(soft_mask)
        self.target_list.append(safest_point)
        cx = safest_point[0]
        cy = safest_point[1]
        
        # --------------------
        # solve minimum fuel trajectory targeting (cx, cy)
        # --------------------

        # solve minimum fuel trajectory
        N = int(tgo / dt)
        x0_log_mass = np.copy(x0)
        x0_log_mass[0] -= cx  # shift to the origin
        x0_log_mass[1] -= cy  #
        x0_log_mass[6] = np.log(x0[6])
        lcvx = LCvxMinFuel(
            lander=self.lander,
            N=N,
            parameterize_x0=False,
            parameterize_tf=False,
            fixed_target=False,
            close_approach=True
        )
        prob = lcvx.problem(x0=x0_log_mass, tf=tgo)
        prob.solve(solver=cp.ECOS, verbose=False)

        if prob.status != cp.OPTIMAL:
            print(f"Optimization failed: {prob.status}")
            # use solution from previous leg
            print("Use solution from previous leg")
            k0 = int(T / dt)
            X = np.copy(self.X_list[-1])[k0:, :]
            U = np.copy(self.U_list[-1])[k0:, :]

        else:
            sol = get_vars(prob, ["X", "U"])
            X_sol = sol["X"]
            U_sol = sol["U"]
            r, v, z, u, _ = lcvx.recover_variables(X_sol, U_sol)
            m = np.exp(z)
            X = np.hstack((r.T, v.T, m.reshape(-1, 1)))
            X[:, 0] += cx
            X[:, 1] += cy
            U = u.T * m[:-1].reshape(-1, 1)
            # --------------------

        t = np.linspace(t0, t0 + tgo, N + 1)
        k_next = int(T / dt)
        x0_next = np.copy(X[k_next, :])
        sfmap_next = self.sfmap_model.get_sfmap(x0_next[2])

        end = time.time()
        print(f"Greedy HDA optimized single leg: {end - start} sec")

        return t, X, U, sfmap_next, x0_next


class HdaReachSteering(HdaGreedy):
    """Reach-steering HDA guidance."""

    def __init__(self, lander: Lander, sfmap_model: SafetyMap, nn_reach: Module, border_sharpness: float = 10.0,
                 itr_max: int = 1000, ftol: float = 1e-8, ctol: float = 1e-6, verbosity: int = 100):
        """Initialize reach-steering HDA guidance.

        Args:
            lander (Lander): lander model
            sfmap_model (SafetyMap): safety map model
            nn_reach (Module): neural network model for reachability prediction
        """
        super().__init__(lander, sfmap_model, nn_reach, border_sharpness)
        self.mean_safety_pred_list = []
        self.itr_max = itr_max
        self.ftol = ftol
        self.ctol = ctol
        self.verbosity = verbosity

    def solve_single_leg(self, x0: np.ndarray, t0: float, tgo: float, T: float, dt: float):
        start = time.time()

        # get safety map
        sfmap = self.sfmap_list[-1]

        # get initial guess from greedy HDA
        t, _, U, _, _ = super().solve_single_leg(x0, t0, tgo, T, dt)
        
        # solve reach-steering problem
        N = int(tgo / dt)
        kmax = int(T / dt)
        udp = ReachSteeringCtrl(self.lander, N, x0, tgo, sfmap, self.nn_reach, kmax, self.border_sharpness)
        prob = pg.problem(udp)
        x0_udp = udp.construct_x(U)

        uda = snopt7(screen_output=False, library="C:/Users/ktomita3/libsnopt7/snopt7.dll", minor_version=7)
        uda.set_integer_option("Major Iteration Limit", self.itr_max)
        uda.set_numeric_option("Major optimality tolerance", self.ftol)
        uda.set_numeric_option("Major feasibility tolerance", self.ctol)
        uda.set_numeric_option('Minor feasibility tolerance', self.ctol)
        algo = pg.algorithm(uda)
        algo.set_verbosity(self.verbosity)

        pop = pg.population(prob, 0)
        pop.push_back(x0_udp)

        result = algo.evolve(pop)
        r, v, m, U = udp.construct_trajectory(result.champion_x)
        X = np.hstack((r, v, m.reshape(-1, 1)))
        
        x0_next = np.hstack((r[kmax, :].flatten(), v[kmax, :].flatten(), m[kmax]))
        sfmap_next = self.sfmap_model.get_sfmap(x0_next[2])

        mean_safety_pred, reach_mask_optimized = ic2mean_safety_npy(self.lander, x0_next, tgo-T, self.nn_reach, sfmap_next, self.border_sharpness)
        self.mean_safety_pred_list.append(mean_safety_pred)
        self.reachmask_list[-1] = reach_mask_optimized

        end = time.time()
        print(f"Reach-steering HDA optimized single leg: {end - start} sec")

        return t, X, U, sfmap_next, x0_next
