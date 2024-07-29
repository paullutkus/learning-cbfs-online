import matplotlib.patches as pat
import matplotlib.pyplot  as plt
import numpy as np
from scipy.stats       import rankdata
from sklearn.neighbors import KDTree, BallTree
from optim import get_h, get_h_curr
from distances import cylindrical_metric



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


def cylindrical_kd_tree_detection(data, k, pct):
    ball        = BallTree(data, metric='pyfunc', func=cylindrical_metric)
    _, knn_inds = ball.query(data, k=k)
    flat_inds   = knn_inds.flatten()
    counts      = np.bincount(flat_inds)
    print("counts", counts)
    print("pct", pct)
    nb_thresh   = np.quantile(counts, pct)
    bd          = data[counts <  nb_thresh]
    _int        = data[counts >= nb_thresh]
    return bd, _int, counts



################
### Plotting ###
################


def plot_cbf(a, centers, thetas, traj=None, nom_traj=None, target=None, obstacles=None, angle=None, line=None):

    ### agent-specific info ###
    width   = a.width
    #s = a.s
    h = get_h(a)

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

    if line is not None:
        plt.plot(np.linspace(-r, r, num=50), line*np.ones((50,)), 'k:', linewidth=2)
        #plt.plot(np.array([1, 1]), np.array([-2, 1]), 'k:', linewidth=1.5)
    if traj is not None:
        plt.plot(traj[:,0], traj[:,1], 'ro')
    if nom_traj is not None:
        plt.plot(nom_traj[:,0], nom_traj[:,1], 'go')
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
    #zz = vmap(lambda arg1, arg2: vmap(lambda s1, s2: h_joint(jnp.array([s1, s2]), thetas, centers), in_axes=(0, 0))(arg1, arg2), in_axes=(0,0))(xx, yy)
    '''

    fig = plt.figure(figsize=(12, 12))
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=60, azim=-30) # prev (35, -30)
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


def _plot_angles(a, centers, thetas, grid, hjb_grid, obs_dict, pos=None):
    h = get_h(a)
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
                hx, _ = h(np.array([x[0], x[1], theta]), centers, thetas)
                if hx <= 0:
                    B = pat.Wedge(  x, s/2, 360/(2*np.pi)*theta - 0.1, 360/(2*np.pi)*theta + 0.1, width=s/2, color='r')  
                    ax.add_patch(B)
        else:
            ax.plot(x[0], x[1], color="red"   , marker="*", linestyle="none") 

    if pos is not None:
        B = pat.Wedge(  pos[:2], s/2, 360/(2*np.pi)*pos[-1] - 1, 360/(2*np.pi)*pos[-1] + 1, width=s/2, color='limegreen')
        ax.add_patch(B)

    plt.show()
    return


def quad_plot(a, ax, pos, centers, thetas, curr_data, traj, grid, obs_dict, obstacles=None):
    h = get_h(a)
    s = a.spacing
    
    # system dimension
    d = curr_data.shape[-1]

    if d == 3:
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

    elif d == 2:
        width = a.width
        n  = 500
        r  = np.ceil(width/2)
        x1 = np.linspace(-r, r, num=n)
        x2 = np.linspace(-r, r, num=n) 
        hvals = np.empty((n, n))
        for i, x in enumerate(x1):
            for j, y in enumerate(x2):
                hvals[i,j], _ = h(np.array([x, y]), centers, thetas)
        #hvals = vmap(lambda s1: vmap(lambda s2: h_joint(jnp.array([s1, s2]), thetas, centers))(x2))(x1)

        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        contour_plot  = ax.contour(np.linspace(-r, r, num=n), np.linspace(-r, r, num=n), hvals.T)
        ax.plot(obstacles[:,0], obstacles[:,1], "r.")
        ax.plot(traj[:,0], traj[:,1], color='magenta', marker=".")
        plt.clabel(contour_plot, inline=1, fontsize=8)

    return



######################
### Set Operations ###
######################


def union(X, Y):
    return np.unique(np.vstack((X,Y)), axis=0)


def intersection(X, Y):
    I = []
    for x in X:
        for y in Y:
            if np.linalg.norm(x-y) <= 1e-3:
                I.append(x)
    return np.unique(np.array(I), axis=0)


def difference(X, Y):
    C = []
    for x in X:
        skip=False
        for y in Y:
            if np.linalg.norm(x-y) <= 1e-3:
                skip=True
        if not skip:
            C.append(x)
    return np.unique(np.array(C), axis=0)
        

