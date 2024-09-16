import matplotlib
import matplotlib.pyplot      as plt
import matplotlib.patches     as pat
import matplotlib.patheffects as pe
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
})
import numpy as np
from scipy.stats       import rankdata
from sklearn.neighbors import KDTree, BallTree
from rbf       import get_h, get_h_curr
from distances import cylindrical_metric
from tqdm      import tqdm



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


def plot_cbf(a, centers, thetas, traj=None, nom_traj=None, slack_traj=None, target=None, obstacles=None, angle=None, line=None,
             N_part=25, plot_max_cbf=False, show_plots=True):
    print("### PLOTTING CBF AND TRAJECTORY ###")
    # GET AGENT-SPECIFIC INFO #
    width = a.width
    h = get_h(a)#, jax_f=True)

    # BUILD GRID FOR CONTOUR PLOT #
    n  = 300
    r  = np.ceil(width/2)
    x1 = np.linspace(-r, r, num=n)
    x2 = np.linspace(-r, r, num=n) 
    x3 = np.linspace( 0, 2*np.pi, num=60)
    #hvals = np.empty((n, n)) #(for manual grid-building)
    hvals=None
    pts = []
    for i, x in enumerate(x1):
        for j, y in enumerate(x2):
            if plot_max_cbf:
                for k, z in enumerate(x3):
                    p = np.array([x, y, z])
                    #p = np.array([x, y, angle])
                    pts.append(p)
            else:
                if angle is not None:
                    p = np.array([x, y, angle])
                else:
                    p = np.array([x, y])
                pts.append(p)
            # manually insert h at gridpoints:
            '''
            if angle is not None:
                hvals[i,j], _ = h(np.array([x, y, angle])[np.newaxis,...], centers, thetas)
            else:
                hvals[i,j], _ = h(np.array([x, y]), centers, thetas)
            '''

    # PARTITIONED, VECTORIZED COMPUTATION OF `h` AT GRIDPOINTS #
    pts = np.array(pts)
    N = pts.shape[0]
    print("num data", N)
    partition = int(N / N_part)
    print("partition", partition)
    remainder = N - (N_part * partition)
    print("remainder", N % partition)
    for i in tqdm(range(N_part+1)):
        if i == N_part and remainder != 0:
            idx_start = i * partition
            idx_end = idx_start + remainder
        elif i == N_part and remainder == 0:
            break
        else:
            idx_start = i * partition
            idx_end = (i+1) * partition 
        X = pts[idx_start:idx_end]
        #print(X.shape)
        if hvals is None:
            # FIX THIS
            hvals = h(X, centers, thetas)[0]
        else:
            hvals = np.concatenate((hvals, h(X, centers, thetas)[0]))

    if plot_max_cbf:
        hvals = hvals.reshape(n, n, 60)
        hvals = np.max(hvals, axis=-1)
    else:
        hvals = hvals.reshape(n, n)

    if show_plots:
        # MAKE CONTOUR PLOT FROM GRID #
        # OLD: hvals = vmap(lambda s1: vmap(lambda s2: h_joint(jnp.array([s1, s2]), thetas, centers))(x2))(x1)
        plt.figure(figsize=(12, 12))
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        contour_plot = plt.contour(np.linspace(-r, r, num=n), np.linspace(-r, r, num=n), hvals.T)
        plt.clabel(contour_plot, inline=1, fontsize=10)

        # PLOT OPTIONAL QUANTITIES #
        if line is not None:
            plt.plot(np.linspace(-r, r, num=50), line*np.ones((50,)), 'k:', linewidth=2)
            #plt.plot(np.array([1, 1]), np.array([-2, 1]), 'k:', linewidth=1.5)
        if traj is not None:
            plt.plot(traj[:,0], traj[:,1], 'ro')
        if nom_traj is not None:
            plt.plot(nom_traj[:,0], nom_traj[:,1], 'go')
        if slack_traj is not None:
            plt.plot(slack_traj[:,0], slack_traj[:,1], 'bo')
        if target is not None:
            plt.plot(target[0], target[1], "m*")
        if obstacles is not None:
            plt.plot(obstacles[:,0], obstacles[:,1], "r.")
        if angle is not None:
            plt.title("theta="+str(angle))

        # MAKE GRID FOR SURFACE PLOT #
        plt.xlabel('$x_1$', fontsize=18)
        plt.ylabel('$x_2$', fontsize=18)
        x1 = np.linspace(-r, r, num=n)
        x2 = np.linspace(-r, r, num=n)
        xx, yy = np.meshgrid(x1, x2)
        zz = np.empty((n, n))

        # UNUSED -- WE RE-USE `hvals` FROM CONTOUR PLOT # 
        '''
        for i, x in enumerate(x1):
            for j, y in enumerate(x2):
                if angle is not None:
                    zz[i,j], _ = h(np.array([x, y, angle]), centers, thetas)
                else:
                    zz[i,j], _ = h(np.array([x, y]), centers, thetas)
        #zz = vmap(lambda arg1, arg2: vmap(lambda s1, s2: h_joint(jnp.array([s1, s2]), thetas, centers), in_axes=(0, 0))(arg1, arg2), in_axes=(0,0))(xx, yy)
        '''
        # SURFACE PLOT #
        fig = plt.figure(figsize=(12, 12))
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=60, azim=-30) # prev (35, -30)
        ax.plot_surface(xx, yy, hvals)
        plt.show()

        # RETURN MAXIMUM `h` IN CASE NORMALIZATION IS PREFERRED #
        print("max h", np.max(hvals))
        return np.max(hvals)
    else:
        return x1, x2, hvals



def plot_angles(a, grid, hjb_grid, obs_dict, pos=None, N_part=25):
    # AGENT-SPECIFIC INFO AND INITIALIZE PLOT #
    h = get_h_curr(a)
    s = a.spacing
    fig, ax = plt.subplots(figsize=(12,12))
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)

    # ATTACH ANGLES TO EACH GRID POINT #
    pts = None
    Nth = hjb_grid.states.shape[2]
    for x in grid:
        p = np.repeat(x[np.newaxis,...], Nth, axis=0)
        p = np.hstack((p, hjb_grid.states[0,0,:,-1][...,np.newaxis]))
        if pts is None:
            pts = p
        else:
            pts = np.vstack((pts, p))

    # PARTITIONED, VECTORIZED COMPUTATION OF `h` AT GRIDPOINTS #
    #pts = grid.reshape(-1, 3)
    N = pts.shape[0]
    print("num data", N)
    partition = int(N / N_part)
    print("partition", partition)
    remainder = N - (N_part * partition)
    print("remainder", N % partition)
    hvals = None
    for i in range(N_part+1):
        if i == N_part and remainder != 0:
            idx_start = i * partition
            idx_end = idx_start + remainder
        elif i == N_part and remainder == 0:
            break
        else:
            idx_start = i * partition
            idx_end = (i+1) * partition 
        X = pts[idx_start:idx_end]
        print(X.shape)
        if hvals is None:
            hvals = h(X)[0]
        else:
            hvals = np.concatenate((hvals, h(X)[0]))

    # AT EACH GRIDPOINT, PLOT UNSAFE ANGLES AS RED, SAFE AS BLUE #
    for i, x in enumerate(grid):
        if obs_dict[tuple(np.round(x, 3))] == 0:
            ax.plot(x[0], x[1], color="black", marker=".", linestyle="none")
            A = pat.Annulus(x, s/2, s/2-0.01)
            ax.add_patch(A)
            pt = np.array([x[0], x[1], 0])
            idx = hjb_grid.nearest_index(pt)[:2]
            for j, theta in enumerate(hjb_grid.states[idx[0], idx[1], :, -1]):
                # [UNUSED] compute h at each gridpoint
                #theta = hjb_grid.states[idx[0], idx[1], i][-1]
                #hx, _ = h(np.array([x[0], x[1], theta])[np.newaxis,...])
                hx = hvals[i*Nth+j]
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



def _plot_angles(a, centers, thetas, grid, hjb_grid, obs_dict, pos=None, N_part=25):
    # AGENT-SPECIFIC INFO AND INITIALIZE PLOT #
    h = get_h(a)
    s = a.spacing
    fig, ax = plt.subplots(figsize=(12,12))
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)

    # ATTACH ANGLES TO EACH GRID POINT #
    pts = None
    Nth = hjb_grid.states.shape[2]
    for x in grid:
        p = np.repeat(x[np.newaxis,...], Nth, axis=0)
        p = np.hstack((p, hjb_grid.states[0,0,:,-1][...,np.newaxis]))
        if pts is None:
            pts = p
        else:
            pts = np.vstack((pts, p))

    # PARTITIONED, VECTORIZED COMPUTATION OF `h` AT GRIDPOINTS #
    #pts = grid.reshape(-1, 3)
    N = pts.shape[0]
    print("num data", N)
    partition = int(N / N_part)
    print("partition", partition)
    remainder = N - (N_part * partition)
    print("remainder", N % partition)
    hvals = None
    for i in range(N_part+1):
        if i == N_part and remainder != 0:
            idx_start = i * partition
            idx_end = idx_start + remainder
        elif i == N_part and remainder == 0:
            break
        else:
            idx_start = i * partition
            idx_end = (i+1) * partition 
        X = pts[idx_start:idx_end]
        print(X.shape)
        if hvals is None:
            hvals = h(X, centers, thetas)[0]
        else:
            hvals = np.concatenate((hvals, h(X, centers, thetas)[0]))

    # AT EACH GRIDPOINT, PLOT UNSAFE ANGLES AS RED, SAFE AS BLUE #
    for i, x in enumerate(grid):
        if obs_dict[tuple(np.round(x, 3))] == 0:
            ax.plot(x[0], x[1], color="black", marker=".", linestyle="none")
            A = pat.Annulus(x, s/2, s/2-0.01)
            ax.add_patch(A)
            pt = np.array([x[0], x[1], 0])
            idx = hjb_grid.nearest_index(pt)[:2]
            for j, theta in enumerate(hjb_grid.states[idx[0], idx[1], :, -1]):
                hx = hvals[i*Nth+j]
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



def quad_plot(a, i, ax, pos, centers, thetas, curr_data, traj, grid, obs_dict, final_traj, colors, traj_idx, obstacles=None):

    #matplotlib.use('ps')
    #from matplotlib import rc

    #rc('text',usetex=True)
    #rc('text.latex', preamble=r'\usepackage{color}')
    centers_last = np.expand_dims(np.array(centers[-1]), 0)
    thetas_last = np.expand_dims(np.array(thetas[-1]), 0)

    centers = np.array(centers)
    thetas = np.array(thetas)

    h = get_h(a)
    s = a.spacing
    
    # system dimension
    d = curr_data.shape[-1]

    ax.set_ylim(-2.5, 1.0)
    ax.set_xlim(-2.5, 2.5)
    ax.autoscale(False)


    #ax.set_title(r'x(t) under h', fontsize=20, y=0.9, color=colors[i])
    if i < 3:
        ax.text(-0.4, 0.70, r'x(t)', fontsize=25, color=colors[i], ha ='right',
                path_effects=[pe.withStroke(linewidth=1, foreground="black")])
        ax.text(0.4, 0.70, r'$h_{}(x)$'.format(i+1), fontsize=25, color='grey', ha ='left',
                path_effects=[pe.withStroke(linewidth=1, foreground="black")])
        ax.text(0.0, 0.70, ' under ', fontsize=25, color='black', ha ='center')
    elif i == 3:
        ax.text(-0.8, 0.70, r'x(t)', fontsize=25, color='magenta', ha ='right',
                path_effects=[pe.withStroke(linewidth=1, foreground="black")])
        ax.text(0.0, 0.70, r'$\max_i\{h_i(x)\}$', fontsize=25, color='black', ha ='left')
        ax.text(-0.4, 0.70, ' under ', fontsize=25, color='black', ha ='center')

    #if i >= 0:
    #    ax.set_axis_off()

    obs_C = pat.Circle((0,0), 0.5, facecolor='darkgrey', alpha=1.00, hatch='////', linewidth=0, zorder=1, edgecolor='dimgrey')
    obs_C_outline = pat.Circle((0.015,-0.015), 0.502, color='black', alpha=0.75, linewidth=4, fill=None, zorder=0)
    ax.add_patch(obs_C_outline)
    ax.add_patch(obs_C)

    obs_L = pat.Rectangle((-2.275, -2.275), 0.525, 0.9 + 2.275, facecolor='darkgrey', alpha=1.00, hatch='////', linewidth=0, edgecolor='dimgrey')
    #obs_L_outline = pat.Rectangle((-2.31, -2.31), 0.035, 0.9 + 2.32, color='black', alpha=0.80, linewidth=0)
    obs_L_outline = pat.Rectangle((-2.31+0.035+0.525, -2.31+0.035), 0.035, 0.9 + 2.232-0.035, color='black', alpha=0.75, linewidth=0)
    ax.add_patch(obs_L)
    ax.add_patch(obs_L_outline)

    obs_R = pat.Rectangle((2.275, -2.275), -0.525, 0.9 + 2.275, facecolor='darkgrey', alpha=1.00, hatch='////', linewidth=0, edgecolor='dimgrey')
    obs_R_outline = pat.Rectangle((2.31, -2.31), -0.035, 0.9 + 2.232, color='black', alpha=0.75, linewidth=0)
    ax.add_patch(obs_R)
    ax.add_patch(obs_R_outline)

    obs_C = pat.Rectangle((-1.75, -2.275), 3.5, 0.525, facecolor='darkgrey', alpha=1.00, hatch='////', linewidth=0, edgecolor='dimgrey')
    obs_C_outline = pat.Rectangle((-2.31+0.035+0.088, -2.31), 3.5+2*0.525-0.088, 0.035, color='black', alpha=0.75, linewidth=0)
    ax.add_patch(obs_C)
    ax.add_patch(obs_C_outline)

    if d == 3:
        for x in curr_data:
            hx, _ = h(x[np.newaxis,...], centers, thetas)
            if hx <= 0:
                B = pat.Wedge(x[:2], s/2, 360/(2*np.pi)*x[2] - 0.1, 360/(2*np.pi)*x[2] + 0.1, width=s/2, color='r')
                ax.add_patch(B)
            else:
                B = pat.Wedge(x[:2], s/2, 360/(2*np.pi)*x[2] - 0.1, 360/(2*np.pi)*x[2] + 0.1, width=s/2, color='c')
                ax.add_patch(B)
            ax.plot(x[0], x[1], color="black", marker=".", linestyle="none", markersize="0.5")
        '''
        for x in grid:
            if x[1] <= 0.9:
                if obs_dict[tuple(np.round(x, 3))] != 0:
                    ax.plot(x[0], x[1], color="black" , marker="s", linestyle="none")
            #else:
            #    ax.plot(x[0], x[1], color="black", marker=".", linestyle="none")
        '''
        if i < 3:
            x_1, x_2, hvals = plot_cbf(a, centers_last, thetas_last, show_plots=False, angle=0, plot_max_cbf=True)
            ax.contourf(x_1, x_2, hvals.T, levels=[0, 10], colors=['black', 'grey'], alpha=0.3, extend='max')#, extent=(-2, 2, -2, 0.5))
        elif i==3:
            x_1, x_2, hvals = plot_cbf(a, centers_last, thetas_last, show_plots=False, angle=0, plot_max_cbf=True)
            ax.contourf(x_1, x_2, hvals.T, levels=[0, 10], colors=['black', 'grey'], alpha=0.0, extend='max')#, extent=(-2, 2, -2, 0.5))

        if i < 3:
            for j in range(i+1):
                ax.plot(traj[traj_idx[j]:traj_idx[j+1],0], traj[traj_idx[j]:traj_idx[j+1],1], color=colors[j], marker=".")
        elif i == 3:
            for j in range(3):
                ax.plot(traj[traj_idx[j]:traj_idx[j+1],0], traj[traj_idx[j]:traj_idx[j+1],1], color=colors[j], marker=".")
            ax.plot(final_traj[:,0], final_traj[:,1], color='magenta', linestyle="--", linewidth=5)

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
        ax.plot(obstacles[:,0], obstacles[:,1], color="r", marker="s")
        ax.plot(traj[:,0], traj[:,1], color='magenta', marker=".")
        plt.clabel(contour_plot, inline=1, fontsize=8)

    return


def cbf_sdf_plot(a, i, ax, cbf_traj, sdf_traj, ht, ht_sdf, centers, thetas):

    #matplotlib.use('ps')
    #from matplotlib import rc

    #rc('text',usetex=True)
    #rc('text.latex', preamble=r'\usepackage{color}')
    centers = np.array(centers)
    thetas = np.array(thetas)

    h = get_h(a)
    s = a.spacing 

    #ax.set_title(r'x(t) under h', fontsize=20, y=0.9, color=colors[i])
    '''
    if i < 3:
        ax.text(-0.4, 0.70, r'x(t)', fontsize=25, color=colors[i], ha ='right',
                path_effects=[pe.withStroke(linewidth=1, foreground="black")])
        ax.text(0.4, 0.70, r'$h_{}(x)$'.format(i+1), fontsize=25, color='grey', ha ='left',
                path_effects=[pe.withStroke(linewidth=1, foreground="black")])
        ax.text(0.0, 0.70, ' under ', fontsize=25, color='black', ha ='center')
    elif i == 3:
        ax.text(-0.8, 0.70, r'x(t)', fontsize=25, color='magenta', ha ='right',
                path_effects=[pe.withStroke(linewidth=1, foreground="black")])
        ax.text(0.0, 0.70, r'$\max_i\{h_i(x)\}$', fontsize=25, color='black', ha ='left')
        ax.text(-0.4, 0.70, ' under ', fontsize=25, color='black', ha ='center')
    '''

    #if i >= 0:
    #    ax.set_axis_off()

    if i == 0:
        ax.set_ylim(-2.31, 0.9)
        ax.set_xlim(-2.31, 1.0)
        ax.autoscale(False)


        obs_C = pat.Circle((0,0), 0.5, facecolor='darkgrey', alpha=1.00, hatch='////', linewidth=0, zorder=1, edgecolor='dimgrey')
        obs_C_outline = pat.Circle((0.015,-0.015), 0.50, color='black', alpha=0.75, linewidth=6, fill=None, zorder=0)
        ax.add_patch(obs_C_outline)
        ax.add_patch(obs_C)

        obs_L = pat.Rectangle((-2.275, -2.275), 0.525, 0.9 + 2.275, facecolor='darkgrey', alpha=1.00, hatch='////', linewidth=0, edgecolor='dimgrey')
        #obs_L_outline = pat.Rectangle((-2.31, -2.31), 0.035, 0.9 + 2.32, color='black', alpha=0.80, linewidth=0)
        obs_L_outline = pat.Rectangle((-2.31+0.035+0.525, -2.31+0.035), 0.035, 0.9 + 2.232-0.035, color='black', alpha=0.75, linewidth=0)
        ax.add_patch(obs_L)
        ax.add_patch(obs_L_outline)

        '''
        obs_R = pat.Rectangle((2.275, -2.275), -0.525, 0.9 + 2.275, facecolor='darkgrey', alpha=1.00, hatch='////', linewidth=0, edgecolor='dimgrey')
        obs_R_outline = pat.Rectangle((2.31, -2.31), -0.035, 0.9 + 2.232, color='black', alpha=0.75, linewidth=0)
        ax.add_patch(obs_R)
        ax.add_patch(obs_R_outline)
        '''

        obs_C = pat.Rectangle((-1.75, -2.275), 0.9+2.275-0.525, 0.525, facecolor='darkgrey', alpha=1.00, hatch='////', linewidth=0, edgecolor='dimgrey')
        obs_C_outline = pat.Rectangle((-2.31+2*0.035+0.035, -2.31), 0.9+2.275-0.035, 0.035, color='black', alpha=0.75, linewidth=0)
        obs_C_outline2= pat.Rectangle((-2.31+2*0.035+0.035+0.9+2.275-0.035, -2.275),-0.035, 0.525-2*0.035, color='black', alpha=0.75, linewidth=0)
        ax.add_patch(obs_C)
        ax.add_patch(obs_C_outline)
        ax.add_patch(obs_C_outline2)

        scan = pat.Circle((-1, -1), 1, edgecolor='red', linewidth=5, fill=False, linestyle=':', capstyle='round', alpha=0.66)
        ax.add_patch(scan)

        x_1, x_2, hvals = plot_cbf(a, centers, thetas, show_plots=False, angle=0, plot_max_cbf=True)
        contour_plot = ax.contourf(x_1, x_2, hvals.T, cmap='winter_r',
                                  levels=np.linspace(0, np.max(hvals), 5))
        contour_line = ax.contour(x_1, x_2, hvals.T, colors='black', linestyles='solid',linewidths=0.5,
                           levels=np.linspace(0, np.max(hvals), 10))

 
        ax.plot(cbf_traj[:,0], cbf_traj[:,1], color='magenta', linestyle='solid', linewidth=5, path_effects=[pe.withStroke(linewidth=6,foreground='black')], 
                solid_capstyle='round', label="CBF")
        ax.plot(sdf_traj[:,0], sdf_traj[:,1], color='gold',  linestyle='solid', linewidth=5, path_effects=[pe.withStroke(linewidth=6,foreground='black')], 
                solid_capstyle='round', label="SDF")
        ax.legend(fontsize=26, loc=(0.80, 0.86))

    elif i == 1:
        ax.set_ylabel(r'$h(t)$', fontsize=36)
        ax.set_xlabel(r'$t$', fontsize=36)
        #ax.legend(fontsize=24)
        ax.tick_params(axis='both', direction='in', labelsize=16)
        ax.grid()
        zeroline = np.zeros(len(ht))
        ax.plot(ht_sdf, color='gold',path_effects=[pe.withStroke(linewidth=3, foreground='black')])
        ax.plot(ht, color='magenta')
        ax.plot(zeroline, linestyle='dashed', alpha=1.0, linewidth=3, color='k')

    return


def performance_comparison(a, fig, a2, axs, centers, thetas, trajecs, vs, usigs, N_part=20):
    width = a.width
    h1 = get_h(a)#, jax_f=True)
    h2 = get_h(a2)

    traj1 = trajecs[0]
    traj2 = trajecs[1]

    v1 = vs[0]
    v2 = vs[1]

    usig = usigs[0]
    usig2 = usigs[1]

    #plt.figure(figsize=(12, 12))
    #plt.rc('xtick', labelsize=14)
    #plt.rc('ytick', labelsize=14)


    # BUILD GRID FOR CONTOUR PLOT #
    for k, h in enumerate([h1, h2]):
        ax = axs[k]
        n  = 300
        r  = np.ceil(width/2)
        x1 = np.linspace(-r, r, num=n)
        x2 = np.linspace(-r, r, num=n) 
        hvals=None
        pts = []
        for i, x in enumerate(x1):
            for j, y in enumerate(x2):
                p = np.array([x, y])
                pts.append(p)

        # PARTITIONED, VECTORIZED COMPUTATION OF `h` AT GRIDPOINTS #
        pts = np.array(pts)
        N = pts.shape[0]
        print("num data", N)
        partition = int(N / N_part)
        print("partition", partition)
        remainder = N - (N_part * partition)
        print("remainder", N % partition)
        for i in tqdm(range(N_part+1)):
            if i == N_part and remainder != 0:
                idx_start = i * partition
                idx_end = idx_start + remainder
            elif i == N_part and remainder == 0:
                break
            else:
                idx_start = i * partition
                idx_end = (i+1) * partition 
            X = pts[idx_start:idx_end]
            #print(X.shape)
            if hvals is None:
                # FIX THIS
                hvals = h(X, centers[k], thetas[k])[0]
            else:
                hvals = np.concatenate((hvals, h(X, centers[k], thetas[k])[0]))

        hvals = hvals.reshape(n, n)
        # MAKE CONTOUR PLOT FROM GRID #
        # OLD: hvals = vmap(lambda s1: vmap(lambda s2: h_joint(jnp.array([s1, s2]), thetas, centers))(x2))(x1)
        #color_map = plt.get_cmap('RdPu')
        hmax = np.max(hvals)
        #color_map.

        if k == 0:
            ax.set_ylim(-1.2, 2.5)
            ax.set_xlim(-2.2+0.95, 2.2)
            ax.autoscale(False)

            
            #ax.text(0.9, 2.2, r'x(t)', fontsize=36, color='#351463', ha ='right',
            #path_effects=[pe.withStroke(linewidth=1, foreground="black")])
            #ax.text(0.3, 2.2, r'$\max_i\{h_i|X_i\}$', fontsize=36, color='black', ha ='left')
            #ax.text(0.6, 2.2, ' under ', fontsize=36, color='black', ha ='center')
            ax.text(-0.3, 2.2, r'x(t)', fontsize=36, color='#351463', ha ='right',
            path_effects=[pe.withStroke(linewidth=1, foreground="black")])
            ax.text(0.3, 2.2, r'$\max_i\{h_i|X_i\}$', fontsize=36, color='black', ha ='left')
            ax.text(0.0, 2.2, ' under ', fontsize=36, color='black', ha ='center')


            #obs_C = pat.Circle((0.02,0.73), 0.490, facecolor='darkgrey', alpha=1.00, hatch='////', linewidth=0, zorder=1, edgecolor='dimgrey')
            #obs_C_outline = pat.Circle((0.010,0.72-0.015+0.020), 0.495, color='black', alpha=0.75, linewidth=4, fill=None, zorder=0)
            obs_C = pat.Circle((0.00,0.74), 0.490, facecolor='darkgrey', alpha=1.00, hatch='////', linewidth=0, zorder=1, edgecolor='dimgrey')
            obs_C_outline = pat.Circle((0.010,0.74-0.010), 0.490, color='black', alpha=0.75, linewidth=5, fill=None, zorder=0)


            ax.add_patch(obs_C_outline)
            ax.add_patch(obs_C)

            '''
            obs_R = pat.Rectangle((2.0, -2.0+0.1+0.8), -0.525, 4-0.8, facecolor='darkgrey', alpha=1.00, hatch='////', linewidth=0, edgecolor='dimgrey')
            obs_R_outline = pat.Rectangle((2.0-0.525-0.020, -2.025+0.1+0.8+0.002+0.023), 0.020, 4-0.8-0.002-0.525-0.020+0.022, color='black', alpha=0.75, linewidth=0)
            obs_R_outline2 = pat.Rectangle((2.0-0.525-0.020, -2.025+0.1+0.8+0.002), 0.525-0.044+0.020, 0.023, color='black', alpha=0.75, linewidth=0)


            ax.add_patch(obs_R)
            ax.add_patch(obs_R_outline)
            ax.add_patch(obs_R_outline2)

            obs_C = pat.Rectangle((-2.0+0.80, 2+0.1), 3.5-0.80, -0.525, facecolor='darkgrey', alpha=1.00, hatch='////', linewidth=0, edgecolor='dimgrey')
            obs_C_outline = pat.Rectangle((-2.0+0.80-0.044+0.022, 2-0.525+0.1), 3.5-0.80-0.065+0.044-0.002, -0.020, color='black', alpha=0.75, linewidth=0)
            obs_C_outline2 = pat.Rectangle((-2.0+0.80-0.044+0.022, 2-0.525+0.1), 0.023, 0.525-0.044+0.020 , color='black', alpha=0.75, linewidth=0)

            ax.add_patch(obs_C)
            ax.add_patch(obs_C_outline)
            ax.add_patch(obs_C_outline2)
            '''
            obs_R = pat.Rectangle((2.0, -2.0+0.1+0.8), -0.525, 4-0.8, facecolor='darkgrey', alpha=1.00, hatch='////', linewidth=0, edgecolor='dimgrey')
            obs_R_outline = pat.Rectangle((2.027-0.010, -2.025+0.1+0.8+0.002), -0.020, 4-0.8-0.002, color='black', alpha=0.75, linewidth=0)
            obs_R_outline2 = pat.Rectangle((2.027-0.010-0.020, -2.025+0.1+0.8+0.002), -0.525+0.044, 0.023, color='black', alpha=0.75, linewidth=0)

            ax.add_patch(obs_R)
            ax.add_patch(obs_R_outline)
            ax.add_patch(obs_R_outline2)

            obs_C = pat.Rectangle((-2.0+0.80, 2+0.1), 3.5-0.80, -0.525, facecolor='darkgrey', alpha=1.00, hatch='////', linewidth=0, edgecolor='dimgrey')
            obs_C_outline = pat.Rectangle((-2.0+0.80+0.044, 2-0.525+0.1), 3.5-0.80-0.065, -0.020, color='black', alpha=0.75, linewidth=0)
            ax.add_patch(obs_C)
            ax.add_patch(obs_C_outline)


            #ax.invert_xaxis()
            contour_plot = ax.contourf(np.linspace(-r, r, num=n), np.linspace(-r, r, num=n), hvals.T, cmap='winter_r',
                            levels=np.linspace(0, hmax, 10))
            #ax.colorbar()
            #contour_plot = ax.contour(np.linspace(-r, r, num=n), np.linspace(-r, r, num=n), hvals.T, colors=['red', 'green', 'blue'], linestyles='solid', linewidths=2,
            #                           levels=[0.50, 1, 1.5] )
            contour_line = ax.contour(np.linspace(-r, r, num=n), np.linspace(-r, r, num=n), hvals.T, colors='black', linestyles='solid',linewidths=0.5,
                                       levels=np.linspace(0, hmax, 15))
            ax.plot(traj2[:,0], traj2[:,1], color='#fcd69a', linestyle='dashed', linewidth=3, path_effects=[pe.withStroke(linewidth=5,foreground='black')]) #f6ff00
            ax.plot(traj1[:,0], traj1[:,1], color='#351463', linestyle='solid', linewidth=3, path_effects=[pe.withStroke(linewidth=5,foreground='black')]) #cb0e40
            ax.plot(traj1[-1,0], traj1[-1,1], color='red', marker='*', ms=20, markeredgecolor='black')

        else:
            ax.set_ylim(-1.2, 2.5)
            ax.set_xlim(-2.2+0.95, 2.2)
            ax.autoscale(False)

            ax.text(-0.3, 2.2, r'x(t)', fontsize=36, color='#fcd69a', ha ='right',
            path_effects=[pe.withStroke(linewidth=1, foreground="black")])
            ax.text(0.3, 2.2, r'$h|\bigcup_i X_i$', fontsize=36, color='black', ha ='left')
            ax.text(0.0, 2.2, ' under ', fontsize=36, color='black', ha ='center')


            #obs_C = pat.Circle((0.00,0.75), 0.490, facecolor='darkgrey', alpha=1.00, hatch='////', linewidth=0, zorder=1, edgecolor='dimgrey')
            #obs_C_outline = pat.Circle((0.035-0.005,0.72-0.015+0.010), 0.495, color='black', alpha=0.75, linewidth=4, fill=None, zorder=0)
            obs_C = pat.Circle((0.00,0.74), 0.490, facecolor='darkgrey', alpha=1.00, hatch='////', linewidth=0, zorder=1, edgecolor='dimgrey')
            obs_C_outline = pat.Circle((0.010,0.74-0.010), 0.490, color='black', alpha=0.75, linewidth=5, fill=None, zorder=0)
            ax.add_patch(obs_C_outline)
            ax.add_patch(obs_C)

            obs_R = pat.Rectangle((2.0, -2.0+0.1+0.8), -0.525, 4-0.8, facecolor='darkgrey', alpha=1.00, hatch='////', linewidth=0, edgecolor='dimgrey')
            obs_R_outline = pat.Rectangle((2.027-0.010, -2.025+0.1+0.8+0.002), -0.020, 4-0.8-0.002, color='black', alpha=0.75, linewidth=0)
            obs_R_outline2 = pat.Rectangle((2.027-0.010-0.020, -2.025+0.1+0.8+0.002), -0.525+0.044, 0.023, color='black', alpha=0.75, linewidth=0)

            ax.add_patch(obs_R)
            ax.add_patch(obs_R_outline)
            ax.add_patch(obs_R_outline2)

            obs_C = pat.Rectangle((-2.0+0.80, 2+0.1), 3.5-0.80, -0.525, facecolor='darkgrey', alpha=1.00, hatch='////', linewidth=0, edgecolor='dimgrey')
            obs_C_outline = pat.Rectangle((-2.0+0.80+0.044, 2-0.525+0.1), 3.5-0.80-0.065, -0.020, color='black', alpha=0.75, linewidth=0)
            ax.add_patch(obs_C)
            ax.add_patch(obs_C_outline)

            contour_plot = ax.contourf(np.linspace(-r, r, num=n), np.linspace(-r, r, num=n), hvals.T, cmap='winter_r',
                            levels=np.linspace(0, hmax, 10))
            contour_line = ax.contour(np.linspace(-r, r, num=n), np.linspace(-r, r, num=n), hvals.T, colors='black', linestyles='solid',linewidths=0.5,
                                       levels=np.linspace(0, hmax, 15))
            ax.plot(traj1[:,0], traj1[:,1], color='#351463', linestyle='dashed', linewidth=3, path_effects=[pe.withStroke(linewidth=5,foreground='black')])
            ax.plot(traj2[:,0], traj2[:,1], color='#fcd69a', linestyle='solid', linewidth=3, path_effects=[pe.withStroke(linewidth=5,foreground='black')])
            ax.plot(traj1[-1,0], traj1[-1,1], color='red', marker='*', ms=20, markeredgecolor='black')


            #contour_plot = ax.contour(np.linspace(-r, r, num=n), np.linspace(-r, r, num=n), hvals.T, colors=['red', 'green', 'blue'], linestyles='dashed', linewidths=2,
            #                           levels=[0.50, 1, 1.5] )
            #outline_effect = pe.withStroke(linewidth=6, foreground='black')
            #contour_plot.set_path_effects([outline_effect])
            #lbl = ax.clabel(contour_plot, inline=1, fontsize=24, colors=['red', 'green', 'blue'])
            #plt.setp(lbl, path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    
    #plt.clabel(contour_plot, inline=1, fontsize=10, colors='black')
    #for k, ax in enumerate(axs[2]):
    print(k)
    ax = axs[2]
    #if k == 0 :
    ax.set_title('2-Norm of Control and Speed vs. Time', fontsize=36)
    ax.plot(v1[:-1], color="#351463", linewidth=4, label=r'$\mathrm{Max CBF:\ } v(t)$',linestyle='dashed', alpha=0.25)#path_effects=[pe.withStroke(linewidth=5, foreground='black')], 
    ax.plot(v2[:-1], color="#fcd69a", linewidth=4, label=r'$\textrm{CBF (all data):\ } v(t)$',linestyle='dashed', alpha=0.75) #path_effects=[pe.withStroke(linewidth=6, foreground='black')], linestyle='dashed', alpha=0.5)

    ax.tick_params(axis='both', direction='in', labelsize=24)
    ax.grid()
    #ax.set_xticks(fontsize=24)
    ax.set_ylabel(r'$v(t)$', fontsize=24)
    ax.set_xlabel(r'$t$', fontsize=24)
    ax.legend(fontsize=24)
    #if k == 1:
    ax.plot(np.linalg.norm(usig, axis=1), color="#351463", label=r'$\mathrm{Max CBF:\ }\|u(t)\|$', linewidth=4, path_effects=[pe.withStroke(linewidth=5, foreground='black')])
    ax.plot(len(usig), np.linalg.norm(usig, axis=1)[-1], color='red', marker='*', ms=23, markeredgecolor='black')

    ax.plot(np.linalg.norm(usig2, axis=1), color="#fcd69a", label=r'$\textrm{CBF (all data):\ }\|u(t)\|$', linewidth=4, path_effects=[pe.withStroke(linewidth=5, foreground='black')])
    ax.plot(len(usig2), np.linalg.norm(usig2, axis=1)[-1], color='red', marker='*', ms=23, markeredgecolor='black')

    ax.set_ylabel(r'$\|u(t)\|, v(t)$', fontsize=36)
    ax.set_xlabel(r'$t$', fontsize=36)
    ax.legend(fontsize=24)

    #fig.tight_layout()
    '''
    box = axs[0].get_position()
    box.x0 = box.x0-5
    box.x1 = box.x1+5
    axs[0].set_position(box)
    '''
    fig.subplots_adjust(left=0.00, right=1.00,bottom=0.00, top=1.00, wspace=-0.0, hspace=-0.2) #-0.3
    plt.show()



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


