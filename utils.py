import matplotlib.patches as pat
import matplotlib.pyplot  as plt
import numpy as np
from scipy.stats       import rankdata
from sklearn.neighbors import KDTree, BallTree
from tst_optim import get_h, get_h_curr



################################
### Boundary point detection ###
################################


def idx_to_coords(lo, hi, n):
    dx = (np.array(hi) - np.array(lo)) / np.array(n)
    return lambda x: tuple([round(y) for y in ((np.array(x) - np.array(lo)) / dx)])



################################
### Boundary point detection ###
################################


def kd_tree_detection(data, k, eta=None, pct=None):
    Z_safe      = data
    tree        = KDTree(Z_safe)
    _, knn_inds = tree.query(Z_safe, k=k)
    flat_inds   = knn_inds.flatten()
    counts      = np.bincount(flat_inds)

    if pct is not None:
        pct_unsafe = pct
        nbr_thresh = np.quantile(counts, pct_unsafe)
        Z_N  = Z_safe[counts < nbr_thresh]
        Z_Nc = Z_safe[counts >= nbr_thresh]
    elif eta is not None:
        Z_N  = Z_safe[counts < eta]
        Z_Nc = Z_safe[counts >= eta]
    else:
        pct_unsafe = 0.2
        nbr_thresh = np.quantile(counts, pct_unsafe)
        Z_N  = Z_safe[counts < nbr_thresh]
        Z_Nc = Z_safe[counts >= nbr_thresh]

    return Z_N, Z_Nc, counts


def cylindrical_metric(x, y):
    d = y - x
    return np.sqrt(d[0]**2 + d[1]**2 + np.minimum(abs(d[2]), 2*np.pi-abs(d[2]))**2)


def cylindrical_kd_tree_detection(data, k, pct):
    ball        = BallTree(data, metric='pyfunc', func=cylindrical_metric)
    _, knn_inds = ball.query(data, k=k)
    flat_inds   = knn_inds.flatten()
    counts      = np.bincount(flat_inds)
    nb_thresh   = np.quantile(counts, pct)
    bd          = data[counts <  nb_thresh]
    _int        = data[counts >= nb_thresh]
    return bd, _int, counts



################
### Plotting ###
################


def plot_cbf(a, centers, thetas, traj=None, target=None, obstacles=None, angle=None):

    #######################
    # agent-specific info #
    #######################
    width   = a.width
    s = a.s
    h = get_h(a)
    #######################

    n  = 500
    r  = np.ceil(width/2)
    x1 = np.linspace(-r, r, num=n)
    x2 = np.linspace(-r, r, num=n) 
    hvals = np.empty((n, n))
    for i, x in enumerate(x1):
        for j, y in enumerate(x2):
            if angle is not None:
                hvals[i,j], _ = h(np.array([x, y, angle]), centers, thetas)
            else:
                hvals[i,j], _ = h(np.array([x, y]), centers, thetas)

    #hvals = vmap(lambda s1: vmap(lambda s2: h_joint(jnp.array([s1, s2]), thetas, centers))(x2))(x1)

    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.figure(figsize=(12, 12))
    contour_plot = plt.contour(np.linspace(-r, r, num=n), np.linspace(-r, r, num=n), hvals.T)
    plt.clabel(contour_plot, inline=1, fontsize=10)

    # These add the dashed lines through (1,1)
    #plt.plot(np.linspace(-2, 1, num=20), np.ones((20,)), 'k:', linewidth=2)
    #plt.plot(np.array([1, 1]), np.array([-2, 1]), 'k:', linewidth=1.5)

    if traj is not None:
        plt.plot(traj[:,0], traj[:,1], 'ro')
    
    if target is not None:
        plt.plot(target[0], target[1], "m*")

    if obstacles is not None:
        plt.plot(obstacles[:,0], obstacles[:,1], "r.")

    if angle is not None:
        plt.title("theta="+str(angle))

    plt.xlabel('$x_1$', fontsize=18)
    plt.ylabel('$x_2$', fontsize=18)
    x1 = np.linspace(-r, r, num=n)
    x2 = np.linspace(-r, r, num=n)
    xx, yy = np.meshgrid(x1, x2)
    zz = np.empty((n, n))
    '''
    for i, x in enumerate(x1):
        for j, y in enumerate(x2):
            if angle is not None:
                zz[i,j], _ = h(np.array([x, y, angle]), centers, thetas)
            else:
                zz[i,j], _ = h(np.array([x, y]), centers, thetas)
    '''
    #zz = vmap(lambda arg1, arg2: vmap(lambda s1, s2: h_joint(jnp.array([s1, s2]), thetas, centers), in_axes=(0, 0))(arg1, arg2), in_axes=(0,0))(xx, yy)

    fig = plt.figure(figsize=(12, 12))#plt.figure(figsize=(16, 10))
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=60, azim=-30)
    #ax.view_init(elev=35, azim=-30)
    ax.plot_surface(xx, yy, hvals)
    plt.show()
    print("max h", np.max(hvals))
    return np.max(hvals)


def plot_angles(a, grid, hjb_grid, obs_dict, pos=None):
    h = get_h_curr(a)
    s = a.spacing

    fig, ax = plt.subplots(figsize=(12,12))

    plt.rc('xtick', labelsize=14)                                                
    plt.rc('ytick', labelsize=14)                                                

    for x in grid:                                                               
        if obs_dict[tuple(np.round(x, 3))] == 0:                               
            ax.plot(x[0], x[1], color="black", marker=".", linestyle="none")   
            A = pat.Annulus(x, s/2, s/2-0.01)
            ax.add_patch(A)
            pt = np.array([x[0], x[1], 0])
            idx = hjb_grid.nearest_index(pt)[:2]
            for theta in hjb_grid.states[idx[0], idx[1], :, -1]:
                #theta = hjb_grid.states[idx[0], idx[1], i][-1]
                hx, _ = h(np.array([x[0], x[1], theta]))
                if hx <= 0:
                    B = pat.Wedge(  x, s/2, 360/(2*np.pi)*theta - 0.1, 360/(2*np.pi)*theta + 0.1, width=s/2, color='r')  
                    ax.add_patch(B)
                #else:
                #    B = pat.Wedge(  x, s/2, 360/(2*np.pi)*theta - 1, 360/(2*np.pi)*theta + 1, width=s/2, color='b')  
                #    ax.add_patch(B)
        else:
            ax.plot(x[0], x[1], color="red"   , marker="*", linestyle="none") 
    if pos is not None:
        B = pat.Wedge(  pos[:2], s/2, 360/(2*np.pi)*pos[-1] - 1, 360/(2*np.pi)*pos[-1] + 1, width=s/2, color='limegreen')
        ax.add_patch(B)

    plt.show()
    return


def quad_plot(a, ax, pos, centers, thetas, curr_data, traj, grid, obs_dict):
    h = get_h(a)
    s = a.spacing

    #print("len curr_data is", len(curr_data))
    for x in curr_data:
        hx, _ = h(x, centers, thetas)
        if hx <= 0:
            B = pat.Wedge(x[:2], s/2, 360/(2*np.pi)*x[2] - 0.1, 360/(2*np.pi)*x[2] + 0.1, width=s/2, color='r')
            ax.add_patch(B)
        else:
            B = pat.Wedge(x[:2], s/2, 360/(2*np.pi)*x[2] - 0.1, 360/(2*np.pi)*x[2] + 0.1, width=s/2, color='c')
            ax.add_patch(B)
    for x in grid:
        if obs_dict[tuple(np.round(x, 3))] != 0:
            ax.plot(x[0], x[1], color="red"  , marker="*", linestyle="none")
        else:
            ax.plot(x[0], x[1], color="black", marker=".", linestyle="none")

    ax.plot(traj[:,0], traj[:,1], color='magenta', marker=".")
    #ax.plot( init_pos[0],  init_pos[1], color="yellow", marker="*")
    #ax.plot(final_pos[0], final_pos[1], color="orange", marker="*")

    return
