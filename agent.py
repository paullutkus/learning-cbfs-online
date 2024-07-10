import matplotlib.pyplot   as plt
import hj_reachability     as hj
import numpy               as np
from   utils           import kd_tree_detection, cylindrical_kd_tree_detection
from   tst_optim       import get_h_curr, get_Dphi
from   controls        import get_xd_mpc, get_safety_filter
from   scipy.integrate import solve_ivp, odeint
from   utils           import plot_cbf
from   data            import local_grid
from   integration     import dynamics_RK4



class Agent:

    def __init__(self, dynamics, pos, grid, obs_dict, width, sensor_radius=1,
                                                             bf      = 31,
                                                             b       =-1,
                                                             s       = 0.66,
                                                             t       = 0,
                                                             utype   ="ball",
                                                             umax    = 1,
                                                             spacing = None,
                                                             solver  ='MOSEK'):
        self.dynamics      = dynamics
        self.pos           = pos
        self.grid          = grid
        self.obs_dict      = obs_dict
        self.width         = width
        self.sensor_radius = sensor_radius
        self.bf            = bf # csrbf order
        self.b             = b  # csrbf offset
        self.s             = s  # csrbf zeroing distance
        self.t             = t
        self.utype         = utype
        self.umax          = umax
        self.spacing       = spacing
        self.solver        = solver
        self.thetas        = []
        self.centers       = []
        self.Nc_max        = None
        self.obstacles     = None


    def scan(self):
        scan_safe   = []
        scan_unsafe = []
        for x in self.grid:
            if np.linalg.norm(x[:2] - self.pos[:2]) <= self.sensor_radius:
                if self.obs_dict[tuple(np.round(x, 3))] == 0:
                    scan_safe.append(x)
                else:
                    scan_unsafe.append(x)
        scan_safe   = np.array(scan_safe)
        scan_unsafe = np.array(scan_unsafe)
        return (scan_safe, scan_unsafe)


    def scan_hjb(self, V, hjb_grid):
        states = []
        for x in self.grid:
            if np.linalg.norm(x[:2] - self.pos[:2]) <= self.sensor_radius:
                x = np.repeat(x.reshape(1, -1), hjb_grid.states.shape[2], axis=0)
                x = np.hstack((x, np.array(hjb_grid.coordinate_vectors[2]).reshape(-1, 1)))
                for pt in x:
                    states.append(pt)
                #states.append(x)
        scan_safe   = []
        scan_unsafe = []
        for x in states:
            Vx = hjb_grid.interpolate(V, x)
            if self.obs_dict[tuple(np.round(x[:2], 3))] == 0 and Vx >= 0:
                scan_safe.append(x)
            else:
                scan_unsafe.append(x)
        scan_safe   = np.array(scan_safe)
        scan_unsafe = np.array(scan_unsafe)
        return (scan_safe, scan_unsafe)


    def sample(self, outer_radius, grid=None, hjb_grid=None, hjb=False):
        samples = []
        if grid is None:
            grid = self.grid
        else:
            grid = grid
        is_obs = lambda x: (np.linalg.norm(x[:2] - self.pos[:2]) >= self.sensor_radius) and \
                           (np.linalg.norm(x[:2] - self.pos[:2]) <= outer_radius)
        for x in grid:
            #r = np.linalg.norm(x[:2] - self.pos[:2])
            #if (r >= self.sensor_radius) and (r <= outer_radius):
            if is_obs(x):
                if hjb:
                    ntheta = hjb_grid.states.shape[2]
                    x = np.repeat(x.reshape(1, -1), ntheta, axis=0)
                    x = np.hstack((x, np.array(hjb_grid.coordinate_vectors[2]).reshape(-1, 1)))
                    for pt in x:
                        samples.append(pt)
                else:
                    samples.append(x)
        samples = np.array(samples)
        #dx1 = abs(grid[0,1] - grid[1,1])
        #print("dx1:", dx1)
        #print("dx2:", dx)
        if hjb:
            dx = abs(hjb_grid.states[0, 0, 0, 0] - hjb_grid.states[1, 0, 0, 0])
            xmax  = np.max(samples[:, 0])
            xmin  = np.min(samples[:, 0])
            ymax  = np.max(samples[:, 1])
            ymin  = np.min(samples[:, 1])
            params = (xmax, xmin, ymax, ymin, dx)
            return samples, params, is_obs
        else:
            return samples


    def make_buffer(self, scan_safe, k, pct, hjb=False):
        # k  : number of neighbors to consider
        # pct: cutoff percentile

        if hjb:
            (x_buffer, x_safe, _) = cylindrical_kd_tree_detection(scan_safe, k, pct)
        else:
            (x_buffer, x_safe, _) = kd_tree_detection(scan_safe, k, pct=pct)
        return (x_buffer, x_safe)


    def get_local_V(self, gparams, obs_funcs, safe_pts, unsafe_pts, rx, thn, T=500, mult=1):

        local_hjb_grid, sdf, grid = local_grid(self.pos  , gparams, obs_funcs , safe_pts,
                                               unsafe_pts, rx     , thn       , mult    )
        solver_settings = hj.SolverSettings.with_accuracy("very_high",
                                                          hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
                                                          value_postprocessor      =hj.solver.static_obstacle(sdf))
        values = sdf
        time = 0.
        target_time = -T
        local_V = hj.step(solver_settings, self.dynamics, local_hjb_grid, time, values, target_time)

        plt.jet()
        plt.figure(figsize=(13, 8))
        plt.contourf(local_hjb_grid.coordinate_vectors[0], local_hjb_grid.coordinate_vectors[1], local_V[:, :, 0].T, levels=30)
        plt.colorbar()
        plt.contour(local_hjb_grid.coordinate_vectors[0],
                    local_hjb_grid.coordinate_vectors[1],
                    local_V[:, :, 0].T,
                    levels=0,
                    colors="black",
                    linewidths=3)

        return local_V, local_hjb_grid


    # ALWAYS ADD CENTERS AFTER NEW THETA IS OBTAINED!
    def rectify_c_and_theta(self):
        Cnew = self.centers[-1]
        print("centers shape before:")
        for c in self.centers:
            print(c.shape)
        if self.Nc_max is None:
            self.Nc_max = Cnew.shape[0]
            return

        numc, xdim = Cnew.shape
        print("numc", numc)
        print("cmax before", self.Nc_max)

        if numc <= self.Nc_max:
            print("numc <= self.cmax")
            print("Cnew shape before", Cnew.shape)
            cpad = np.repeat(np.zeros((1, xdim)), self.Nc_max - numc,axis=0)
            self.centers[-1] = np.vstack((Cnew, cpad))
            print("Cnew shape after", Cnew.shape)
            print("theta shape before", self.thetas[-1].shape)
            thetapad = np.zeros(self.Nc_max - numc)
            self.thetas[-1] = np.append(self.thetas[-1], thetapad)
            print("theta_shape after", self.thetas[-1].shape)

        else:
            print("numc > self.cmax")
            for i, C in enumerate(self.centers[:-1]):
                print("C[i] shape before", self.centers[i].shape)
                cpad = np.repeat(np.zeros((1, xdim)), numc - self.Nc_max,
                                                  axis=0)
                self.centers[i] = np.vstack((C, cpad))
                print("C[i] shape after", self.centers[i].shape)
                print("theta[i] shape before", self.thetas[i].shape)
                thetapad = np.zeros(numc - self.Nc_max)
                self.thetas[i] = np.append(self.thetas[i], thetapad)
                print("theta[i] shape after", self.thetas[i].shape)
            self.Nc_max = numc

        print("cmax after", self.Nc_max)
        print("centers shape after")
        for c in self.centers:
            print(c.shape)
        return


    def goto(self, target, T=0.5, tend=100, tol=0.05, use_cbf_mpc=False, angle=None, manual=True, DT=0.05, eps=0):
        f = self.dynamics.open_loop_dynamics
        g = self.dynamics.control_jacobian

        h             = get_h_curr(self)
        xd_mpc        = get_xd_mpc(self)
        safety_filter = get_safety_filter(self, eps=eps)

        print("position is", self.pos)
        print("h is", h(self.pos)[0])
        print("start time is", self.t)

        #if use_cbf_mpc:
        #    closed_loop = lambda t, y: f(y,0) + g(y,0) @ cbf_mpc(y, target, T, self.thetas, self.centers)
        #else:
        safe_controller = lambda y: safety_filter(y, xd_mpc(y, target, T=T))
        closed_loop     = lambda y, t: f(y,0) + g(y,0) @ safe_controller(y)
        ODE_RHS         = lambda y, u: f(y,0) + g(y,0) @ u

        boundary           = lambda t, y: h(y)[0] - tol
        boundary.terminal  = True
        boundary.direction = 0 #-1

        if manual:
            y = self.pos
            traj = []; traj.append(y)
            Dphi = get_Dphi(self)
            while h(y)[0] - tol > 0:
                #print("hmax", h(y)[0])
                #hy, i = h(y)
                #print("norm grad", np.linalg.norm(self.thetas[i].T @ Dphi(y, self.centers[i])))
                u = safety_filter(y, xd_mpc(y, target, T=T))
                y = dynamics_RK4(y, u, ODE_RHS, DT)
                traj.append(y)
            traj = np.array(traj)
        else: 
            x0 = self.pos
            tend=tend
            teval = np.linspace(0, tend, num=100000)
            sol = solve_ivp(closed_loop, (0, tend), x0, teval=teval, events=boundary) #teval=teval, dense_output=False)
            #sol = odeint(closed_loop, x0, teval)
            traj = sol.y.T

        if angle is not None:
            plot_cbf(self, np.array(self.centers), np.array(self.thetas), traj=traj, target=target, angle=angle, obstacles=self.obstacles)
        else:
            plot_cbf(self, np.array(self.centers), np.array(self.thetas), traj=traj, target=target, obstacles=self.obstacles)


        self.pos = traj[-1, :]
        self.t  += tend

        print("new position is" , self.pos)
        print("h is now", h(self.pos)[0])
        print("new time is", self.t)
        return traj
