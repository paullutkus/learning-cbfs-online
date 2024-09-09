import     numpy as np
import jax.numpy as jnp
import matplotlib.patches as pat
import matplotlib.pyplot  as plt
import hj_reachability    as hj
from jax import vmap



#######################
### Generate shapes ###
#######################


def make_rectangle(height=1, width=1, density=200, unsafe_margin=0.25, center=(0,0), return_params=False):

    # Create rectangular grid
    n_pts = int( (height+max(height,width)*unsafe_margin) * (width+max(height,width)*unsafe_margin) * density )
    ax_1, s_1 = np.linspace(  -(width + max(height,width) * unsafe_margin) / 2 + center[0],
                               (width + max(height,width) * unsafe_margin) / 2 + center[0],
                           int((width + max(height,width) * unsafe_margin) * np.sqrt(density)), retstep=True)

    ax_2, s_2 = np.linspace( -(height + max(height,width) * unsafe_margin) / 2 + center[1],
                              (height + max(height,width) * unsafe_margin) / 2 + center[1],
                          int((height + max(height,width) * unsafe_margin) * np.sqrt(density)), retstep=True)

    grid = np.einsum('ijk->kji', np.meshgrid(ax_1, ax_2)).reshape(ax_1.shape[0] * ax_2.shape[0], 2).astype(np.float32)

    # Create obstacle information dictionary
    in_width  = lambda x : True if (x <= width  / 2 + center[0]) and (x >= -width  / 2 + center[0]) else False
    in_height = lambda y : True if (y <= height / 2 + center[1]) and (y >= -height / 2 + center[1]) else False
    obs_dict  = {tuple(np.round(pt.astype(np.float32), 3)) : 1 if not (in_width(pt[0]) and in_height(pt[1])) else 0 for pt in grid}

    get_is_obs = lambda in_width, in_height: lambda x: not (in_width(x[0]) and in_height(x[1]))
    is_obs = get_is_obs(in_width, in_height)
    
    # Compute params
    lo = (ax_1[ 0], ax_2[ 0])
    hi = (ax_1[-1], ax_2[-1])
    n  = (len(ax_1), len(ax_2))
    params = (lo, hi, n, s_1)

    if return_params:
        return grid, params, obs_dict, is_obs
    else:
        return grid, obs_dict


def insert_shape(pos, grid, obs_dict, shape='triangle', scale=0.2, theta=0):
    if shape == 'triangle':
        # Create three hyperplanes, check to see whether it falls on the correct side of each
        # Alternative condition: angles w.r.t. vertices add up to 360 inside triangle, < 360 outside

        # Vertices are v1: top, v2: bottom left, v3 : bottom right
        # There are constraints on the angle where this is valid!
        assert(abs(theta) < np.pi/2)
        v1 = np.array((pos[0] - scale * np.sin(theta)            , pos[1] + scale * np.cos(theta)))
        v2 = np.array((pos[0] - scale * np.sin(theta + 2*np.pi/3), pos[1] + scale * np.cos(theta + 2*np.pi/3)))
        v3 = np.array((pos[0] - scale * np.sin(theta + 4*np.pi/3), pos[1] + scale * np.cos(theta + 4*np.pi/3)))
        v1v3 = lambda x : (v1[1]-v3[1])/(v1[0]-v3[0]) * x + (v1[1] - (v1[1]-v3[1])/(v1[0]-v3[0])*v1[0])
        v2v3 = lambda x : (v2[1]-v3[1])/(v2[0]-v3[0]) * x + (v2[1] - (v2[1]-v3[1])/(v2[0]-v3[0])*v2[0])
        v1v2 = lambda x : (v1[1]-v2[1])/(v1[0]-v2[0]) * x + (v1[1] - (v1[1]-v3[1])/(v1[0]-v2[0])*v1[0])
        triangle_pts = [tuple(np.round(np.array((x[0], x[1])).astype(np.float32), 3)) for x in grid if ((v1v3(x[0]) >= x[1]) and (v1v2(x[0]) >= x[1]) and (v2v3(x[0]) <= x[1]))]
        for pt in triangle_pts: obs_dict[pt] = 2

    elif shape == 'rhombus':
        for k, x in enumerate(grid):
            x_trans = x - np.array(pos)
            x_scale = 1/scale * x_trans
            rot = np.array([[np.cos(-theta),-np.sin(-theta)],
                            [np.sin(-theta), np.cos(-theta)]])
            x_derotated = rot @ x_scale
            if np.linalg.norm(x_derotated, ord=np.inf) <= 1:
                obs_dict[tuple(np.round(x.astype(np.float32), 3))] = 3

    elif shape == 'circle':
        for k, x in enumerate(grid):
            x_trans = x - np.array(pos)
            x_scale = 1/scale * x_trans
            if np.linalg.norm(x_scale) <= 1:
                obs_dict[tuple(np.round(x.astype(np.float32), 3))] = 4
        get_is_obs = lambda scale, pos: lambda x: np.linalg.norm(1/scale*(x-np.array(pos))) <= 1
        is_obs = get_is_obs(scale, pos)

    return obs_dict, is_obs


def check_obs(x, obs_f):
    is_obs = False
    if list is type(obs_f):
        for f in obs_f:
            is_obs = is_obs or f(x)
    else: 
        is_obs = obs_f(x)
    return is_obs



#########################
### Local Grid  & HJB ###
#########################


def get_gparams(data, hjb_grid):
    d = data.shape[-1] # system dimension
    dx = abs(hjb_grid.states[(0,)+d*(0,)] - hjb_grid.states[(1,)+d*(0,)])
    xmax  = np.max(data[:, 0])
    xmin  = np.min(data[:, 0])
    ymax  = np.max(data[:, 1])
    ymin  = np.min(data[:, 1])
    params = (xmax, xmin, ymax, ymin, dx)
    return params


def local_grid(pos, gparams, obs_funcs, thn=None, out_func=None, rx=None, mult=1,
               extension=0):
    xmax, xmin, ymax, ymin, dx = gparams
    ymax += extension
    ymin -= extension
    xmax += extension
    xmin -= extension
    xn  = int(np.round((xmax - xmin) / dx)) + 1
    yn  = int(np.round((ymax - ymin) / dx)) + 1
    if thn is not None:
        lo  = jnp.array((xmin, ymin, 0      ))
        hi  = jnp.array((xmax, ymax, 2*np.pi))
        n = jnp.array((mult*xn, mult*yn, thn))
        local_hjb_grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(lo, hi), n, periodic_dims=2)

    else:
        lo  = jnp.array((xmin, ymin))
        hi  = jnp.array((xmax, ymax))
        n = jnp.array((mult*xn, mult*yn))
        local_hjb_grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(lo, hi), n)

    unsafe_pts = []
    safe_pts   = []
    obs_dict   = {}
    if thn is not None:
        grid = local_hjb_grid.states[...,0,:2].reshape(-1, 2)
    else:
        grid = local_hjb_grid.states.reshape(-1, 2)
    if rx is not None and out_func is None:
        out_func = lambda x: np.linalg.norm(x - pos[:2], ord=2) > rx
    for x in grid:
        if check_obs(x, obs_funcs) or out_func(x):
            unsafe_pts.append(x)
            obs_dict[tuple(np.round(np.array(x).astype(np.float32), 3))] = 1
        else:
            safe_pts.append(x)
            obs_dict[tuple(np.round(np.array(x).astype(np.float32), 3))] = 0
    unsafe_pts = np.array(unsafe_pts)
    safe_pts   = np.array(safe_pts)

    sdf = np.empty(local_hjb_grid.states.shape[:-1])
    if thn is not None:
        for i in range(n[0]):
            for j in range(n[1]):
                if obs_dict[tuple(np.round(np.array(local_hjb_grid.states[i, j, 0, :2]), 3))] != 0:
                    sdf[i, j, :] = -np.min(np.linalg.norm(safe_pts   - local_hjb_grid.states[i, j, 0, :2], axis=1))
                else:
                    sdf[i, j, :] =  np.min(np.linalg.norm(unsafe_pts - local_hjb_grid.states[i, j, 0, :2], axis=1))
        plt.scatter(grid[:,0], grid[:,1], c=sdf[...,0].reshape(-1, 1))
    else:
        for i in range(n[0]):
            for j in range(n[1]):
                if obs_dict[tuple(np.round(np.array(local_hjb_grid.states[i, j, :2]), 3))] != 0:
                    sdf[i, j] = -np.min(np.linalg.norm(safe_pts   - local_hjb_grid.states[i, j, :2], axis=1))
                else:
                    sdf[i, j] =  np.min(np.linalg.norm(unsafe_pts - local_hjb_grid.states[i, j, :2], axis=1))
        plt.scatter(grid[:,0], grid[:,1], c=sdf.reshape(-1, 1))

    plt.show()
    return local_hjb_grid, sdf, grid



################
### Plotting ###
################


def plot_data(grid, obs_dict, extra=None, size=12):

    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.figure(figsize=(size, size))

    for x in grid:
        if   obs_dict[tuple(np.round(x, 3))] == 0:
            plt.plot(x[0], x[1], color="yellow", marker=".", linestyle="none")
        elif obs_dict[tuple(np.round(x, 3))] == 1:
            plt.plot(x[0], x[1], color="red"   , marker=".", linestyle="none")
        elif obs_dict[tuple(np.round(x, 3))] == 2:
            plt.plot(x[0], x[1], color="red"   , marker=".", linestyle="none")
        elif obs_dict[tuple(np.round(x, 3))] == 3:
            plt.plot(x[0], x[1], color="red"   , marker=".", linestyle="none")
        elif obs_dict[tuple(np.round(x, 3))] == 4:
            plt.plot(x[0], x[1], color="red"   , marker=".", linestyle="none")

    if extra is not None:
        for data, color, marker in extra:
            for x in data:
                plt.plot(x[0], x[1], color=color, marker=marker, linestyle="none")

    plt.show()
    return None


def plot_angle_data(centers, grid, obs_dict, s, safe=None, buffer=None, unsafe=None, samples=None, title=None): 

    fig, ax = plt.subplots(figsize=(12,12))
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)

    for x in grid:
        if obs_dict[tuple(np.round(x, 3))] == 0:
            plt.plot(x[0], x[1], color="greenyellow", marker="o", linestyle="none")
        else:
            plt.plot(x[0], x[1], color="red"   , marker="o", linestyle="none")
    if safe is not None:
        for x in safe:
            B = pat.Wedge(  x[:2], s/2, 360/(2*np.pi)*x[2] - 0.1, 360/(2*np.pi)*x[2] + 0.1, width=s/2, color='c')  
            ax.add_patch(B)
    if unsafe is not None:
        for x in unsafe:
            B = pat.Wedge(  x[:2], s/2, 360/(2*np.pi)*x[2] - 0.1, 360/(2*np.pi)*x[2] + 0.1, width=s/2, color='r')  
            ax.add_patch(B)
    if buffer is not None:
        for x in buffer:
            B = pat.Wedge(  x[:2], s/2, 360/(2*np.pi)*x[2] - 0.1, 360/(2*np.pi)*x[2] + 0.1, width=s/2, color='b')  
            ax.add_patch(B)
    if samples is not None:
        for x in samples:
            B = pat.Wedge(  x[:2], s/2, 360/(2*np.pi)*x[2] - 0.1, 360/(2*np.pi)*x[2] + 0.1, width=s/2, color='r')  
            ax.add_patch(B)
    for x in centers:
        ax.plot(x[0], x[1], color="black", marker=".", linestyle="none")

    if title is not None:
        ax.set_title(title)
    plt.show()
    return



###############################
### Trajectory-sampled data ###
###############################


def generate_trajecotries(a, V, grid):
    f = a.dynamics.open_loop_dyanmics
    g = a.dynamics.control_jacobian 

    fv = vmap(f, in_axes=(0, None))
    gv = vmap(g, in_axes=(0, None))

    ODE_RHS = lambda y, u: fv(y,0) + np.linalg.norm(grid.grad_values)
    pass

