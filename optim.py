import numpy     as np
import cvxpy     as cp
import casadi    as ca
import clarabel
from   jax       import vmap
from   scipy     import sparse



def get_h(a):
    b  = a.b
    bf = a.bf
    s  = a.s 
    def h(x, C, theta):
        h_vec = np.einsum('ijk,jk->j', phiT(x, C, bf, s), theta)
        i = np.argmax(h_vec)
        return h_vec[i] + b, i
    return h


def get_h_curr(a):
    b     = a.b
    bf    = a.bf
    s     = a.s 
    C     = np.array(a.centers)
    theta = np.array(a.thetas)
    def h(x):
        h_vec = np.einsum('ijk,jk->j', phiT(x, C, bf, s), theta)
        i = np.argmax(h_vec)
        return h_vec[i] + b, i
    return h


def phiT(x, C, bf, s): # is a column vector
    r = np.linalg.norm(x[...,np.newaxis,:] - C[np.newaxis,...], ord=2, axis=-1)
    if bf == 0:
        phi = np.e**(-s*r**2)
        return phi
    if bf == 2:
        phi = np.sqrt(s + r**2)
        return phi
    if bf == 31:
        phi = np.maximum(0, s - r)**4 * (1 + 4*r)/20
        return phi
    if bf == 32: 
        phi = np.maximum(0, s - r)**6 * (3 + 18*r + 35*r**2)/1680
        return phi


def get_phiT(a):
    bf = a.bf
    s  = a.s
    def phiT(x, C): # is a column vector
        r = np.linalg.norm(x[...,np.newaxis,:] - C[np.newaxis,...], ord=2, axis=-1)
        if bf == 0:
            phi =  np.e**(-s*r**2)
            return phi
        if bf == 2:
            phi = np.sqrt(s + r**2)
            return phi
        if bf == 31:
            phi = np.maximum(0, s - r)**4 * (1 + 4*r)/20
            return phi
        if bf == 32: 
            phi = np.maximum(0, s - r)**6 * (3 + 18*r + 35*r**2)/1680
            return phi
    return phiT


def get_phiT_curr(a):
    bf = a.bf
    s  = a.s
    C  = a.centers[-1]
    def phi(x): # is a column vector
        r = np.linalg.norm(x[...,np.newaxis,:] - C[np.newaxis,...], ord=2, axis=-1)
        if bf == 0:
            phi = np.e**(-s*r**2)
            return phi
        if bf == 2:
            phi = np.sqrt(s + r**2)
            return phi
        if bf == 31:
            phi = np.maximum(0, s - r)**4 * (1 + 4*r)/20
            return phi
        if bf == 32:    
            phi = np.maximum(0, s - r)**6 * (3 + 18*r + 35*r**2)/1680
            return phi
    return phi


def get_Dphi(a):
    bf = a.bf
    s  = a.s
    def Dphi(x, C): # is a row vector 
        r   = np.linalg.norm(x[...,np.newaxis,:] - C[np.newaxis,...], ord=2, axis=-1)
        msk = np.sign(np.maximum(0, s - r))
        if bf == 0:
            dwdr = -4*s*r*np.e**(-s*r**2)
            Dphi = dwdr[...,np.newaxis] * (x[...,np.newaxis,:] - C[np.newaxis,...])
            return Dphi
        if bf == 2:
            Dphi = (x[...,np.newaxis,:] - C[np.newaxis,...]) / np.sqrt(s + r**2)[...,np.newaxis]
            return Dphi
        if bf == 31:
            dwdr = (msk*(-4*(s - r)**3 * (1 + 4*r)    / 20 +\
                          4*np.maximum(0, s -   r)**4 / 20) )
            Dphi = dwdr[...,np.newaxis] * (x[...,np.newaxis,:] - C[np.newaxis,...]) / (1e-5+r)[...,np.newaxis]
            return Dphi
        if bf == 32:
            dwdr = (msk*(-6*(s-r)**5 * (1 + 18*r + 35*r**2)  /1680 + \
                                       (    18   + 70*r   )  /1680) )
            Dphi = dwdr[...,np.newaxis] * (x[...,np.newaxis,:] - C[np.newaxis,...]) / (1e-5+r)[...,np.newaxis]
            return Dphi
    return Dphi


def get_Dphi_curr(a):
    bf = a.bf
    s  = a.s
    C  = a.centers[-1]
    def Dphi(x): # is a row vector 
        r   = np.linalg.norm(x[...,np.newaxis,:] - C[np.newaxis,...], ord=2, axis=-1)
        msk = np.sign(np.maximum(0, s - r))
        if bf == 0:
            dwdr = -4*s*r*np.e**(-s*r**2)
            Dphi = dwdr[...,np.newaxis] * (x[...,np.newaxis,:] - C[np.newaxis,...])
            return Dphi
        if bf == 2:
            Dphi = (x[...,np.newaxis,:] - C[np.newaxis,...]) / np.sqrt(s + r**2)[...,np.newaxis]
            return Dphi
        if bf == 31:
            dwdr = (msk*(-4*(s - r)**3 * (1 + 4*r)    / 20 +\
                          4*np.maximum(0, s -   r)**4 / 20) )
            Dphi = dwdr[...,np.newaxis] * (x[...,np.newaxis,:] - C[np.newaxis,...]) / (1e-5+r)[...,np.newaxis]
            return Dphi
        if bf == 32:
            dwdr = (msk*(-6*(s-r)**5 * (1 + 18*r + 35*r**2)  /1680 + \
                                       (    18   + 70*r   )  /1680) )
            Dphi = dwdr[...,np.newaxis] * (x[...,np.newaxis,:] - C[np.newaxis,...]) / (1e-5+r)[...,np.newaxis]
            return Dphi
    return Dphi


def cas_train_cbf(a, x_safe  , 
                     x_buffer, 
                     x_unsafe, gamma_safe  ,
                               gamma_dyn   ,
                               gamma_unsafe, lam_dh=1,
                                            verbose=False):

    # get agent-spefic objects
    C        = np.array(a.centers[-1])
    dynamics = a.dynamics
    f        = dynamics.open_loop_dynamics
    g        = dynamics.control_jacobian
    phi      = get_phi(a)
    Dphi     = get_Dphi(a)
    b        = a.b

    opti        = ca.Opti()
    cons        = []
    n           = C.shape[0]
    theta       = opti.variable(n, 1)
    theta_cost  = ca.sumsqr(theta) #+ 0.5 * ca.sum1(ca.fabs(theta))
    dh_cost     = 0

    for x in x_safe:
        cons.append(ca.mtimes(theta.T, phi(x, C)) + b >= gamma_safe)
        cons.append(ca.mtimes(ca.mtimes(theta.T, Dphi(x, C)), f(x,0)) + ca.mtimes(theta.T, phi(x, C)) +\
                    ca.norm_2(ca.mtimes(ca.mtimes(np.array(g(x,0)).T, Dphi(x, C).T), theta)) >= gamma_dyn)
        dh_cost += ca.sumsqr(ca.mtimes(theta.T, phi(x, C)))

    for x in x_buffer:
        cons.append(ca.mtimes(ca.mtimes(theta.T, Dphi(x, C)), f(x,0)) + ca.mtimes(theta.T, phi(x, C)) +\
                    ca.norm_2(ca.mtimes(ca.mtimes(np.array(g(x,0)).T, Dphi(x, C).T), theta)) >= gamma_dyn)
        dh_cost += ca.sumsqr(ca.mtimes(theta.T, phi(x, C)))

    for x in x_unsafe:
        cons.append(ca.mtimes(theta.T, phi(x, C)) + b <= gamma_unsafe)

    # setup and solve problem
    opti.minimize(theta_cost + lam_dh * dh_cost)
    opti.subject_to(cons)
    opti.set_initial(theta, ca.DM.rand(C.shape[0]))
    p_opts = {"expand" : True, "verbose" : verbose, "print_time" : verbose}
    s_opts = {"max_iter" : 1000, "print_level" : 5*int(verbose)}
    opti.solver('ipopt', p_opts, s_opts)
    sol = opti.solve()
    return sol.value(theta)


def cvx_train_cbf(a, x_safe  , u_safe  ,
                     x_buffer, u_buffer,
                     x_unsafe,          gamma_safe  ,
                                        gamma_dyn   ,
                                        gamma_unsafe, lam_dh=0         ,
                                                      lam_sp=0         ,
                                                      verbose=False):

    C         = np.array(a.centers[-1])
    dynamics  = a.dynamics
    solver    = a.solver
    f         = dynamics.open_loop_dynamics
    g         = dynamics.control_jacobian
    phi       = get_phi(a)
    Dphi      = get_Dphi(a)
    b         = a.b
    theta_dim = C.shape[0]
    theta     = cp.Variable((theta_dim, 1))
    cons      = []
    dh_cost   = 0

    for x, u in zip(x_safe, u_safe):
        cons.append(  theta.T @  phi(x,C) + b >= gamma_safe )
        cons.append(  theta.T @ Dphi(x,C) @ (f(x,0) + g(x,0) @ u) \
                    + theta.T @  phi(x,C) + b >= gamma_dyn 
                   )
        dh_cost += cp.sum_squares(theta.T @ Dphi(x,C))

    for x in x_unsafe:
        cons.append(theta.T @ phi(x,C) + b <= gamma_unsafe)

    for x, u in zip(x_buffer, u_buffer):
        cons.append(  theta.T @ Dphi(x,C) @ (f(x,0) + g(x,0) @ u) \
                    + theta.T @  phi(x,C) + b >= gamma_dyn 
                   )
        dh_cost += cp.sum_squares(theta.T @ Dphi(x,C))

    obj  = cp.Minimize(cp.sum_squares(theta) + lam_dh*dh_cost + lam_sp*cp.norm(theta,1))
    prob = cp.Problem(obj, cons)
    prob.solve(verbose=True, solver=solver)
    return theta.value.flatten()


def clarabel_fit_cbf(a, x_safe  , u_safe  ,
                          x_buffer, u_buffer,
                          x_unsafe, gamma_safe, gamma_dyn, gamma_unsafe):
    fv   = vmap(a.dynamics.open_loop_dynamics, in_axes=(0, None))
    gv   = vmap(a.dynamics.control_jacobian  , in_axes=(0, None))
    phiT = get_phiT(a)
    Dphi = get_Dphi(a)
    C    = a.centers[-1]
    o    = a.b

    ns = x_safe.shape[0]
    nb = x_buffer.shape[0]
    nu = x_unsafe.shape[0]


    P = sparse.csc_array(np.eye(C.shape[0]))
    q = np.zeros(C.shape[0])
    b = np.concatenate([-gamma_safe  *np.ones(ns) + o*np.ones(ns),
                         gamma_unsafe*np.ones(nu) - o*np.ones(nu),
                        -gamma_dyn*np.ones(ns+nb) + o*np.ones(ns+nb)])

    #A = sparse.csc_array(np.vstack( phiT(x,C),
    #    DphiT(x,C),)


    if u_buffer.shape[0] != 0:
        A = sparse.csc_array(np.vstack(( -phiT(x_safe  ,C),
                                          phiT(x_unsafe,C),
                                      -((Dphi(x_safe  ,C) @ (fv(x_safe  ,0) + (gv(x_safe  ,0) @   u_safe[:,np.newaxis,:]).squeeze())[...,np.newaxis]).squeeze() + phiT(x_safe  ,C)),
                                      -((Dphi(x_buffer,C) @ (fv(x_buffer,0) + (gv(x_buffer,0) @ u_buffer[:,np.newaxis,:]).squeeze())[...,np.newaxis]).squeeze() + phiT(x_buffer,C))
                                      ))
                            )
    else:
        A = sparse.csc_array(np.vstack(( -phiT(x_safe  ,C),
                                          phiT(x_unsafe,C),
                                      -((Dphi(x_safe,C) @ (fv(x_safe,0) + (gv(x_safe,0) @ u_safe[:,np.newaxis,:]).squeeze())[...,np.newaxis]).squeeze() + phiT(x_safe,C))
                                      ))
                            )
    cone = [clarabel.NonnegativeConeT(A.shape[0])]

    print("problem done building")
    settings = clarabel.DefaultSettings()
    solver   = clarabel.DefaultSolver(P, q, A, b, cone, settings)
    solution = solver.solve()
    return solution.x


def clarabel_solve_cbf(a, x_safe  , u_safe  ,
                          x_buffer, u_buffer,
                          x_unsafe, gamma_safe, gamma_dyn, gamma_unsafe, centers=None):
    fv   = vmap(a.dynamics.open_loop_dynamics, in_axes=(0, None))
    gv   = vmap(a.dynamics.control_jacobian  , in_axes=(0, None))
    phiT = get_phiT(a)
    Dphi = get_Dphi(a)
    if centers is None:
        C = a.centers[-1]
    else:
        C = centers
    o    = a.b

    ns = x_safe.shape[0]
    nb = x_buffer.shape[0]
    nu = x_unsafe.shape[0]


    P = sparse.csc_array(np.eye(C.shape[0]))
    q = np.zeros(C.shape[0])
    b = np.concatenate([-gamma_safe  *np.ones(ns) + o*np.ones(ns),
                         gamma_unsafe*np.ones(nu) - o*np.ones(nu),
                        -gamma_dyn*np.ones(ns+nb) + o*np.ones(ns+nb)])
    if u_buffer.shape[0] != 0:
        A = sparse.csc_array(np.vstack(( -phiT(x_safe  ,C),
                                          phiT(x_unsafe,C),
                                      -((Dphi(x_safe  ,C) @ (fv(x_safe  ,0) + (gv(x_safe  ,0) @   u_safe[...,np.newaxis]).squeeze())[...,np.newaxis]).squeeze() + phiT(x_safe  ,C)),
                                      -((Dphi(x_buffer,C) @ (fv(x_buffer,0) + (gv(x_buffer,0) @ u_buffer[...,np.newaxis]).squeeze())[...,np.newaxis]).squeeze() + phiT(x_buffer,C))
                                      ))
                            )
    else:
        A = sparse.csc_array(np.vstack(( -phiT(x_safe  ,C),
                                          phiT(x_unsafe,C),
                                      -((Dphi(x_safe,C) @ (fv(x_safe,0) + (gv(x_safe,0) @ u_safe[...,np.newaxis]).squeeze())[...,np.newaxis]).squeeze() + phiT(x_safe,C))
                                      ))
                            )
    cone = [clarabel.NonnegativeConeT(A.shape[0])]

    print("problem done building")
    settings = clarabel.DefaultSettings()
    solver   = clarabel.DefaultSolver(P, q, A, b, cone, settings)
    solution = solver.solve()
    return solution.x


def get_learning_cbfs_lagrangian(a, x_safe  ,
                                    x_buffer, 
                                    x_unsafe, lam_safe  ,
                                              lam_dyn   ,
                                              lam_unsafe,
                                              lam_dh    , gamma_safe,
                                                          gamma_dyn ,
                                                          gamma_unsafe):
    phi      = get_phiT_curr(a)
    Dphi     = get_Dphi_curr(a)
    dynamics = a.dynamics
    b        = a.b
    tst_Dphi = get_Dphi(a)

    f        = dynamics.open_loop_dynamics
    g        = dynamics.control_jacobian
    fv   = vmap(a.dynamics.open_loop_dynamics, in_axes=(0, None))
    gv   = vmap(a.dynamics.control_jacobian  , in_axes=(0, None))

    def L(theta):
        S  = gamma_safe - phi(x_safe) @ theta[:,np.newaxis] - b
        Ds = gamma_dyn  - (theta.T @ Dphi(x_safe  )[:,np.newaxis,:] @ fv(x_safe  ,0)[...,np.newaxis]).squeeze() -\
                np.linalg.norm(np.einsum("ijk->ikj", gv(x_safe  ,0)) @ np.einsum("ijk->ikj", Dphi(x_safe  )) @ theta, ord=2, axis=1).squeeze() -\
                (phi(x_safe  ) @ theta[:,np.newaxis]).squeeze() - b
        Db = gamma_dyn  - (theta.T @ Dphi(x_buffer)[:,np.newaxis,:] @ fv(x_buffer,0)[...,np.newaxis]).squeeze() -\
                np.linalg.norm(np.einsum("ijk->ikj", gv(x_buffer,0)) @ np.einsum("ijk->ikj", Dphi(x_buffer)) @ theta, ord=2, axis=1).squeeze() -\
                (phi(x_buffer) @ theta[:,np.newaxis]).squeeze() - b
        U = phi(x_unsafe) @ theta[:,np.newaxis] + b + gamma_unsafe
        dhDs = np.linalg.norm(np.einsum("ijk->ikj", Dphi(x_safe  )) @ theta[:,np.newaxis], ord=2, axis=1)
        dhDb = np.linalg.norm(np.einsum("ijk->ikj", Dphi(x_buffer)) @ theta[:,np.newaxis], ord=2, axis=1)

        dS_dT  = -phi(x_safe)
        dDs_dT = -phi(x_safe) - (Dphi(x_safe) @ fv(x_safe,0)[...,np.newaxis]).squeeze() - \
                 Dphi(x_safe) @ gv(x_safe,0) @ np.einsum("ijk->ikj", gv(x_safe,0)) @ np.einsum("ijk->ikj", Dphi(x_safe)) @ theta /\
                 np.linalg.norm(np.einsum("ijk->ikj", gv(x_safe,0)) @ np.einsum("ijk->ikj", Dphi(x_safe)) @ theta, ord=2, axis=1, keepdims=True)
        dDb_dT = -phi(x_buffer) - (Dphi(x_buffer) @ fv(x_buffer,0)[...,np.newaxis]).squeeze() - \
                 Dphi(x_buffer) @ gv(x_buffer,0) @ np.einsum("ijk->ikj", gv(x_buffer,0)) @ np.einsum("ijk->ikj", Dphi(x_buffer)) @ theta /\
                 np.linalg.norm(np.einsum("ijk->ikj", gv(x_buffer,0)) @ np.einsum("ijk->ikj", Dphi(x_buffer)) @ theta, ord=2, axis=1, keepdims=True)
        dU_dT  = phi(x_unsafe)
        ddhDs_dT = Dphi(x_safe  ) @ np.einsum("ijk->ikj", Dphi(x_safe  )) @ theta / np.linalg.norm(np.einsum("ijk->ikj", Dphi(x_safe  )) @ theta, ord=2, axis=1, keepdims=True)
        ddhDb_dT = Dphi(x_buffer) @ np.einsum("ijk->ikj", Dphi(x_buffer)) @ theta / np.linalg.norm(np.einsum("ijk->ikj", Dphi(x_buffer)) @ theta, ord=2, axis=1, keepdims=True)

        return ((theta**2).sum() +
                 lam_safe  * np.maximum(0, S ).sum() +\
                 lam_dyn   * np.maximum(0, Ds).sum() +\
                 lam_dyn   * np.maximum(0, Db).sum() +\
                 lam_unsafe* np.maximum(0, U ).sum() +\
                 lam_dh    * dhDs.sum() +\
                 lam_dh    * dhDs.sum()
               , 2*theta +\
                 lam_safe * (np.sign(np.maximum(0, S )) * dS_dT ).sum(axis=0) +\
                 lam_dyn  * (np.sign(np.maximum(0, Ds))[:,np.newaxis] * dDs_dT).sum(axis=0) +\
                 lam_dyn  * (np.sign(np.maximum(0, Db))[:,np.newaxis] * dDb_dT).sum(axis=0) +\
                 lam_unsafe* (np.sign(np.maximum(0, U )) * dU_dT ).sum(axis=0) +\
                 lam_dh    * (ddhDs_dT).sum(axis=0) + \
                 lam_dh    * (ddhDb_dT).sum(axis=0))

    return L

'''
, 2*theta + lam_safe  *sum([np.sign(np.maximum(0, gamma_safe - theta.T @ phi(x) - b)) * -phi(x) \
                            for x in x_safe  ]) + \
            lam_dyn   *sum([np.sign(np.maximum(0, gamma_dyn - theta.T @ Dphi(x) @ f(x,0) - np.linalg.norm(g(x,0).T @ Dphi(x).T @ theta, ord=2) - theta.T @ phi(x) - b)) * \
                          (-phi(x) - Dphi(x) @ f(x,0) - Dphi(x) @ g(x,0) @ g(x,0).T @ Dphi(x).T @ theta / np.linalg.norm(g(x,0).T @ Dphi(x).T @ theta, ord=2)) \
                            for x in x_safe  ]) + \
            lam_dyn   *sum([np.sign(np.maximum(0, gamma_dyn - theta.T @ Dphi(x) @ f(x,0) - np.linalg.norm(g(x,0).T @ Dphi(x).T @ theta, ord=2) - theta.T @ phi(x) - b)) * \
                          (-phi(x) - Dphi(x) @ f(x,0) - Dphi(x) @ g(x,0) @ g(x,0).T @ Dphi(x).T @ theta / np.linalg.norm(g(x,0).T @ Dphi(x).T @ theta, ord=2)) \
                            for x in x_buffer]) + \
            lam_unsafe*sum([np.sign(np.maximum(0, theta.T @ phi(x) + b + gamma_unsafe)) * phi(x) \
                            for x in x_unsafe]) + \
            lam_dh    *sum([Dphi(x) @ Dphi(x).T @ theta / np.linalg.norm(theta.T @ Dphi(x), ord=2) \
                            for x in x_safe  ]) + \
            lam_dh    *sum([Dphi(x) @ Dphi(x).T @ theta / np.linalg.norm(theta.T @ Dphi(x), ord=2) \
                            for x in x_buffer])
)

'''


###################
#### DEPRECATED ###
###################
def get_learning_cbfs_lagrangian_hj(a, x_safe  , u_safe  ,
                                       x_buffer, u_buffer, 
                                       x_unsafe,           lam_safe  ,
                                                           lam_dyn   ,
                                                           lam_unsafe,
                                                           lam_dh    , gamma_safe,
                                                                       gamma_dyn ,
                                                                       gamma_unsafe):
    phi      = get_phi_curr(a)
    Dphi     = get_Dphi_curr(a)
    dynamics = a.dynamics
    f        = dynamics.open_loop_dynamics
    g        = dynamics.control_jacobian
    b        = a.b
    def L(theta):
        return ( (theta**2).sum() +\
                  lam_safe  *sum([np.maximum(0, gamma_safe - theta.T @ phi(x) - b) \
                                  for x in x_safe  ]) + \
                  lam_dyn   *sum([np.maximum(0, gamma_dyn - theta.T @ Dphi(x) @ ( f(x,0) + g(x,0)@u ) - theta.T @ phi(x) - b) \
                                  for x, u in zip(x_safe, u_safe)]) + \
                  lam_dyn   *sum([np.maximum(0, gamma_dyn - theta.T @ Dphi(x) @ ( f(x,0) + g(x,0)@u ) - theta.T @ phi(x) - b) \
                                  for x, u in zip(x_buffer, u_buffer)]) + \
                  lam_unsafe*sum([np.maximum(0, theta.T @ phi(x) + b + gamma_unsafe) \
                                  for x in x_unsafe]) + \
                  lam_dh    *sum([np.linalg.norm(Dphi(x).T @ theta, ord=2) \
                                  for x in x_safe  ]) + \
                  lam_dh    *sum([np.linalg.norm(Dphi(x).T @ theta, ord=2) \
                                  for x in x_buffer])
                , 2*theta + lam_safe  *sum([np.sign(np.maximum(0, gamma_safe - theta.T @ phi(x) - b)) * -phi(x) \
                                            for x in x_safe  ]) + \
                            lam_dyn   *sum([np.sign(np.maximum(0, gamma_dyn - theta.T @ Dphi(x) @ ( f(x,0) + g(x,0)@u ) - theta.T @ phi(x) - b)) * \
                                          (-phi(x) - Dphi(x) @ ( f(x,0) + g(x,0)@u )) \
                                            for x, u in zip(x_safe, u_safe)]) + \
                            lam_dyn   *sum([np.sign(np.maximum(0, gamma_dyn - theta.T @ Dphi(x) @ ( f(x,0) + g(x,0)@u ) - theta.T @ phi(x) - b)) * \
                                          (-phi(x) - Dphi(x) @ ( f(x,0) + g(x,0)@u )) \
                                            for x, u in zip(x_buffer, u_buffer)]) + \
                            lam_unsafe*sum([np.sign(np.maximum(0, theta.T @ phi(x) + b + gamma_unsafe)) * phi(x) \
                                            for x in x_unsafe]) + \
                            lam_dh    *sum([Dphi(x) @ Dphi(x).T @ theta / np.linalg.norm(theta.T @ Dphi(x), ord=2) \
                                            for x in x_safe  ]) + \
                            lam_dh    *sum([Dphi(x) @ Dphi(x).T @ theta / np.linalg.norm(theta.T @ Dphi(x), ord=2) \
                                            for x in x_buffer])
               )
    return L

##################
### DEPRECATED ###
##################
def get_learning_cbfs_lagrangian_hj_optim(a, x_safe  , u_safe  ,
                                             x_buffer, u_buffer, 
                                             x_unsafe,           lam_safe  ,
                                                                 lam_dyn   ,
                                                                 lam_unsafe,
                                                                 lam_dh    , gamma_safe,
                                                                             gamma_dyn ,
                                                                             gamma_unsafe):
    phi      = get_phi_curr(a)
    Dphi     = get_Dphi_curr(a)
    dynamics = a.dynamics
    f        = dynamics.open_loop_dynamics
    g        = dynamics.control_jacobian
    b        = a.b
    f_safe   = lambda theta: sum([lam_safe * np.maximum(0, gamma_safe - theta.T @ phi(x) - b) +\
                                  lam_dyn  * np.maximum(0, gamma_dyn - theta.T @ Dphi(x) @ ( f(x,0) + g(x,0)@u ) - theta.T @ phi(x) - b) +\
                                  lam_dh   * np.linalg.norm(Dphi(x).T @ theta, ord=2) \
                                  for x, u in zip(x_safe, u_safe)])

    f_buffer = lambda theta: sum([lam_dyn * np.maximum(0, gamma_dyn - theta.T @ Dphi(x) @ ( f(x,0) + g(x,0)@u ) - theta.T @ phi(x) - b) +\
                                  lam_dh  * np.linalg.norm(Dphi(x).T @ theta, ord=2) \
                                  for x, u in zip(x_buffer, u_buffer)])

    f_unsafe = lambda theta: lam_unsafe*sum([np.maximum(0, theta.T @ phi(x) + b + gamma_unsafe) \
                                             for x in x_unsafe])

    Df_safe   = lambda theta: sum([lam_safe* np.sign(np.maximum(0, gamma_safe - theta.T @ phi(x) - b)) * -phi(x)  +\
                                  lam_dyn  * np.sign(np.maximum(0, gamma_dyn - theta.T @ Dphi(x) @ ( f(x,0) + g(x,0)@u ) - theta.T @ phi(x) - b)) *\
                                  (-phi(x) - Dphi(x) @ ( f(x,0) + g(x,0)@u )) +\
                                  lam_dh * Dphi(x) @ Dphi(x).T @ theta / np.linalg.norm(theta.T @ Dphi(x), ord=2) \
                                  for x, u in zip(x_safe, u_safe)])

    Df_buffer   = lambda theta: sum([lam_dyn  * np.sign(np.maximum(0, gamma_dyn - theta.T @ Dphi(x) @ ( f(x,0) + g(x,0)@u ) - theta.T @ phi(x) - b)) *\
                                    (-phi(x) - Dphi(x) @ ( f(x,0) + g(x,0)@u )) +\
                                    lam_dh * Dphi(x) @ Dphi(x).T @ theta / np.linalg.norm(theta.T @ Dphi(x), ord=2) \
                                    for x, u in zip(x_buffer, u_buffer)])

    Df_unsafe = lambda theta: lam_unsafe*sum([np.sign(np.maximum(0, theta.T @ phi(x) + b + gamma_unsafe)) * phi(x) \
                                             for x in x_unsafe])

    def L(theta):
        return ( (theta**2).sum() + f_safe(theta) + f_buffer(theta) + f_unsafe(theta)
                , 2*theta +        Df_safe(theta) +Df_buffer(theta) +Df_unsafe(theta) )
    return L
