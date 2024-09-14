import numpy     as np
import cvxpy     as cp
import casadi    as ca
import clarabel
import jax.numpy as jnp
import jax
from   jax       import vmap
from   rbf       import phiT, get_phiT, get_phiT_curr, get_Dphi, get_Dphi_curr, jax_get_phiT, jax_get_phiT_curr, jax_phiT
from   jax_optim import jax_get_learning_cbfs_lagrangian
from   scipy     import sparse
from   distances import cylindrical_metric
import matplotlib.pyplot as plt



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


# fit cbf or cbvf data by inverting system of equations
def fit_cbf_w_inv(a, X, rhs):
    phiT = get_phiT(a)
    A = phiT(X, X)
    #print(A.shape)
    theta = np.linalg.inv(A) @ (rhs - a.b)
    return theta


# gradient descent to tune fitted weights
def gradient_descent(a, theta, it, eps, mu, tol, x_safe, u_safe, x_buffer, u_buffer, x_unsafe, u_unsafe, x_0, x_2pi,\
                     gamma_safe, gamma_dyn, gamma_unsafe, lam_safe, lam_dyn, lam_unsafe, lam_dh, lam_theta, lam_bd, hmax,\
                     centers=None, use_jax=False, ckpt=None):
    if ckpt is not None:
        ckpt.reverse()
    if use_jax:
        theta = jnp.array(theta)
        x_safe = jnp.array(x_safe); u_safe = jnp.array(u_safe)
        x_buffer = jnp.array(x_buffer); u_buffer = jnp.array(u_buffer)
        x_unsafe = jnp.array(x_unsafe)
        L = jax.jit(jax_get_learning_cbfs_lagrangian(a, x_safe, u_safe, x_buffer, u_buffer, x_unsafe, u_unsafe, x_0, x_2pi,\
                    gamma_safe, gamma_dyn, gamma_unsafe, lam_safe, lam_dyn, lam_unsafe, lam_dh, lam_theta, lam_bd, hmax,\
                    centers=centers))
    else:
        L = get_learning_cbfs_lagrangian(a, x_safe, u_safe, x_buffer, u_buffer, x_unsafe, u_unsafe,
                                         lam_safe, lam_dyn, lam_unsafe, lam_dh,
                                         gamma_safe, gamma_dyn, gamma_unsafe, centers=centers)
    loss = []
    loss_Lsafe = []
    loss_Ldyn = []
    loss_Lunsafe = []
    loss_Ldh = []
    loss_Lhm = []
    loss_Lbd = []
    weights = []
    mom = 0
    v, grad, _ = L(theta)
    v_prev = v + 10

    for i in range(it+1):
        #theta_ip1 = theta - eps * (1 + jnp.cos(i*jnp.pi/it)) * grad
        v, grad, losses = L(theta)
        L_safe_v, L_safe_dyn, L_buffer, L_unsafe_v, L_unsafe_dyn, L_dh, L_hm, L_bd = losses
        loss_Lsafe.append(L_safe_v)
        loss_Ldyn.append(L_safe_dyn + L_buffer + L_unsafe_dyn)
        loss_Lunsafe.append(L_unsafe_v)
        loss_Ldh.append(L_dh)
        loss_Lhm.append(L_hm)
        loss_Lbd.append(L_bd)
        mom = mu * mom + grad
        theta = theta - eps * mom
        if i % 10 == 0:
            print("it", i, "loss:", v)
        if v == 0 or v == v_prev or abs(v_prev - v) <= tol:
            print(v_prev - v)
            print(v)
            print(v_prev)
            print("triggered")
            break
        if ckpt is not None:
            if i == ckpt[-1]:
                weights.append(theta)
                ckpt.pop()
        loss.append(v)
        v_prev = v

    plt.figure(figsize=(14,12))
    #plt.ylim((0, 200))
    plt.plot(loss_Lsafe, label="safe")
    plt.plot(loss_Ldyn, label="dyn")
    plt.plot(loss_Lunsafe, label="unsafe")
    plt.plot(loss_Ldh, label="grad norm")
    plt.plot(loss_Lhm, label="hmax")
    plt.plot(loss_Lbd, label="boundary")
    plt.figtext(.78, .75, "gamma_safe="+str(gamma_safe))
    plt.figtext(.78, .73, "gamma_dyn="+str(gamma_dyn))
    plt.figtext(.78, .71, "gamma_unsafe="+str(gamma_unsafe))
    plt.legend()
    plt.show()
    return theta, loss, loss_Lsafe, loss_Ldyn, loss_Lunsafe, weights


def clarabel_solve_cbf(a, x_safe  , u_safe  ,
                          x_buffer, u_buffer,
                          x_unsafe, gamma_safe, gamma_dyn, gamma_unsafe, centers=None, x2pi=None, x0=None):
    fv   = vmap(a.dynamics.open_loop_dynamics, in_axes=(0, None))
    gv   = vmap(a.dynamics.control_jacobian  , in_axes=(0, None))
    phiT = get_phiT(a)
    Dphi = get_Dphi(a)
    if centers is None:
        C = a.centers[-1]
    else:
        C = centers
    o    = a.b
    gamma= a.gamma

    ns = x_safe.shape[0]
    nb = x_buffer.shape[0]
    nu = x_unsafe.shape[0]


    P = sparse.csc_array(np.eye(C.shape[0]))
    q = np.zeros(C.shape[0])
    b = np.concatenate([-gamma_safe  *np.ones(ns) + o*np.ones(ns),
                         gamma_unsafe*np.ones(nu) - o*np.ones(nu),
                        -gamma_dyn*np.ones(ns+nb) + o*np.ones(ns+nb)])

    if x2pi is not None:
        b2 = np.zeros(x2pi.shape[0])
        b = np.concatenate([b, b2])
    print("b shape", b.shape)

    if u_buffer.shape[0] != 0:
        A1 = sparse.csc_array(np.vstack(( -phiT(x_safe  ,C),
                                           phiT(x_unsafe,C),
                                       -((Dphi(x_safe  ,C) @ (fv(x_safe  ,0) + (gv(x_safe  ,0) @   u_safe[...,np.newaxis]).squeeze())[...,np.newaxis]).squeeze() + gamma*phiT(x_safe  ,C)),
                                       -((Dphi(x_buffer,C) @ (fv(x_buffer,0) + (gv(x_buffer,0) @ u_buffer[...,np.newaxis]).squeeze())[...,np.newaxis]).squeeze() + gamma*phiT(x_buffer,C))
                                       ))
                             )
    else:
        A1 = sparse.csc_array(np.vstack(( -phiT(x_safe  ,C),
                                           phiT(x_unsafe,C),
                                       -((Dphi(x_safe,C) @ (fv(x_safe,0) + (gv(x_safe,0) @ u_safe[...,np.newaxis]).squeeze())[...,np.newaxis]).squeeze() + gamma*phiT(x_safe,C))
                                       ))
                             )

    if x2pi is not None:
        A2 = sparse.csc_array(phiT(x2pi,C) - phiT(x0,C))
        #                    Dphi(x2pi,C) - Dphi(x0,C)))
        A = sparse.csc_array(sparse.vstack((A1, A2)))
    else:
        A = A1

    n0 = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] == 0:
                n0 += 1
    print("num zero elems in A:", n0)

    if x2pi is not None:
        cones = [clarabel.NonnegativeConeT(A1.shape[0]), clarabel.ZeroConeT(A2.shape[0])]
    else:
        cones = [clarabel.NonnegativeConeT(A1.shape[0])]

    print("problem done building")
    settings = clarabel.DefaultSettings()
    solver   = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solution = solver.solve()
    return solution.x


def get_learning_cbfs_lagrangian(a, x_safe  , u_safe  ,
                                    x_buffer, u_buffer,
                                    x_unsafe, lam_safe,
                                              lam_dyn   ,
                                              lam_unsafe,
                                              lam_dh    , gamma_safe,
                                                          gamma_dyn ,
                                                          gamma_unsafe,
                                                          centers=None):
    if centers is None:
        C = a.centers[-1]
    else:
        C = centers
    phi      = get_phiT(a)
    Dphi     = get_Dphi(a)
    dynamics = a.dynamics
    b        = a.b

    f        = dynamics.open_loop_dynamics
    g        = dynamics.control_jacobian
    fv   = vmap(a.dynamics.open_loop_dynamics, in_axes=(0, None))
    gv   = vmap(a.dynamics.control_jacobian  , in_axes=(0, None))


    # preformat u shapes for vectorized matrix multiplication (ijk),(ikl)->(ijl)
    u_safe = u_safe.reshape(u_safe.shape[0], u_safe.shape[1], 1)
    if u_buffer.shape[0] != 0:
        u_safe = u_buffer.reshape(u_buffer.shape[0], u_buffer.shape[1], 1)

    # M = dim(theta), N = dim(x)
    Dphi_xsafe = Dphi(x_safe,C) 
    safe_f = phi(x_safe,C)
    safe_dyn_f = Dphi_xsafe @ (fv(x_safe,C)[...,np.newaxis] + gv(x_safe,0) @ u_safe)
    if u_buffer.shape[0] != 0:
        Dphi_xbuffer = Dphi(x_buffer,C)
        buffer_f = phi(x_buffer,C)
        buffer_dyn_f = Dphi_xbuffer @ (f(x_buffer,C)[...,np.newaxis] + gv(x_buffer,0) @ u_buffer)
    unsafe_f = phi(x_unsafe,C)


    def L(theta): 
        safe_L = gamma_safe - (safe_f @ theta[:,np.newaxis]).squeeze() - b # (M x 1)
        safe_dyn_L = gamma_dyn  - (theta[np.newaxis,np.newaxis,:] @ safe_dyn_f).squeeze() -\
                         (safe_f @ theta[:,np.newaxis]).squeeze() - b
        unsafe_L = (unsafe_f @ theta[:,np.newaxis]).squeeze() + b + gamma_unsafe
        # norm of gradient
        #gnorm_safe_L = np.linalg.norm(theta[np.newaxis,np.newaxis,:] @ Dphi_xsafe, ord=2, axis=2).squeeze()
        Dsafe_L = -safe_f
        Dsafe_dyn_L = -safe_f - (safe_dyn_f).squeeze()
        Dunsafe_L = unsafe_f
        #Dgnorm_safe_L = (Dphi_xsafe @ (np.einsum("ijk->ikj", Dphi_xsafe) @ theta)[...,np.newaxis] /\
        #                 np.linalg.norm(theta[np.newaxis,np.newaxis,:] @ Dphi_xsafe, ord=2, axis=2, keepdims=True)).squeeze()
        if u_buffer.shape[0] != 0:
            buffer_dyn_L = gamma_dyn - (theta[np.newaxis,np.newaxis,:] @ buffer_dyn_f).squeeze() -\
                            (buffer_f @ theta[:,np.newaxis]).squeeze() - b
            #gnorm_buffer_L = np.linalg.norm(theta[np.newaxis,np.newaxis,:] @ Dphi_xbuffer, ord=2, axis=2).squeeze()
            Dbuffer_dyn_L = -buffer_f - (buffer_dyn_f).squeeze()
            #Dgnorm_buffer_L = (Dphi_xbuffer @ (np.einsum("ijk->ikj", Dphi_xbuffer) @ theta)[...,np.newaxis] /\
            #np.linalg.norm(theta[np.newaxis,np.newaxis,:] @ Dphi_xbuffer, ord=2, axis=2, keepdims=True)).squeeze()
        else:
            buffer_dyn_L = np.array([0])
            #gnorm_buffer_L = np.array([0])
            Dbuffer_dyn_L = np.array([0])
            #Dgnorm_buffer_L = np.array([0])

        return ((#theta**2).sum() +
                 lam_safe  * np.maximum(0, safe_L).sum() +\
                 lam_dyn   * np.maximum(0, safe_dyn_L).sum() +\
                 lam_dyn   * np.maximum(0, buffer_dyn_L).sum() +\
                 lam_unsafe* np.maximum(0, unsafe_L).sum())# +\
                 #lam_dh    * gnorm_safe_L.sum() +\
                 #lam_dh    * gnorm_buffer_L.sum()
               , 2*theta +\
                    lam_safe * (np.sign(np.maximum(0, safe_L))[:,np.newaxis] * Dsafe_L).sum(axis=0) +\
                 lam_dyn  * (np.sign(np.maximum(0, safe_dyn_L))[:,np.newaxis] * Dsafe_dyn_L).sum(axis=0) +\
                 lam_dyn  * (np.sign(np.maximum(0, buffer_dyn_L))[:,np.newaxis] * Dbuffer_dyn_L).sum(axis=0) +\
                 lam_unsafe* (np.sign(np.maximum(0, unsafe_L))[:,np.newaxis] * Dunsafe_L ).sum(axis=0))# +\
                 #lam_dh    * (Dgnorm_safe_L).sum(axis=0) + \
                 #lam_dh    * (Dgnorm_buffer_L).sum(axis=0))

    return L


def get_learning_cbfs_lagrangian_dualnorm(a, x_safe  ,
                                             x_buffer, 
                                             x_unsafe, lam_safe  ,
                                                       lam_dyn   ,
                                                       lam_unsafe,
                                                       lam_dh    , gamma_safe,
                                                                   gamma_dyn ,
                                                                   gamma_unsafe,
                                                                   centers=None):
    if centers is None:
        C = a.centers[-1]
    else:
        C = centers
    phi      = get_phiT(a)
    Dphi     = get_Dphi(a)
    dynamics = a.dynamics
    b        = a.b

    f        = dynamics.open_loop_dynamics
    g        = dynamics.control_jacobian
    fv   = vmap(a.dynamics.open_loop_dynamics, in_axes=(0, None))
    gv   = vmap(a.dynamics.control_jacobian  , in_axes=(0, None))

    def L(theta):
        S  = gamma_safe - phi(x_safe,C) @ theta[:,np.newaxis] - b
        Ds = gamma_dyn  - (theta.T @ Dphi(x_safe,C)[:,np.newaxis,:] @ fv(x_safe  ,0)[...,np.newaxis]).squeeze() -\
                np.linalg.norm(np.einsum("ijk->ikj", gv(x_safe  ,0)) @ np.einsum("ijk->ikj", Dphi(x_safe,C)) @ theta, ord=2, axis=1).squeeze() -\
                (phi(x_safe,C) @ theta[:,np.newaxis]).squeeze() - b
        Db = gamma_dyn  - (theta.T @ Dphi(x_buffer,C)[:,np.newaxis,:] @ fv(x_buffer,0)[...,np.newaxis]).squeeze() -\
                np.linalg.norm(np.einsum("ijk->ikj", gv(x_buffer,0)) @ np.einsum("ijk->ikj", Dphi(x_buffer,C)) @ theta, ord=2, axis=1).squeeze() -\
                (phi(x_buffer,C) @ theta[:,np.newaxis]).squeeze() - b
        U = phi(x_unsafe,C) @ theta[:,np.newaxis] + b + gamma_unsafe
        dhDs = np.linalg.norm(np.einsum("ijk->ikj", Dphi(x_safe,C)) @ theta[:,np.newaxis], ord=2, axis=1)
        dhDb = np.linalg.norm(np.einsum("ijk->ikj", Dphi(x_buffer,C)) @ theta[:,np.newaxis], ord=2, axis=1)

        dS_dT  = -phi(x_safe,C)
        dDs_dT = -phi(x_safe,C) - (Dphi(x_safe,C) @ fv(x_safe,0)[...,np.newaxis]).squeeze() - \
                 Dphi(x_safe,C) @ gv(x_safe,0) @ np.einsum("ijk->ikj", gv(x_safe,0)) @ np.einsum("ijk->ikj", Dphi(x_safe,C)) @ theta /\
                 np.linalg.norm(np.einsum("ijk->ikj", gv(x_safe,0)) @ np.einsum("ijk->ikj", Dphi(x_safe,C)) @ theta, ord=2, axis=1, keepdims=True)
        dDb_dT = -phi(x_buffer,C) - (Dphi(x_buffer,C) @ fv(x_buffer,0)[...,np.newaxis]).squeeze() - \
                 Dphi(x_buffer,C) @ gv(x_buffer,0) @ np.einsum("ijk->ikj", gv(x_buffer,0)) @ np.einsum("ijk->ikj", Dphi(x_buffer,C)) @ theta /\
                 np.linalg.norm(np.einsum("ijk->ikj", gv(x_buffer,0)) @ np.einsum("ijk->ikj", Dphi(x_buffer,C)) @ theta, ord=2, axis=1, keepdims=True)
        dU_dT  = phi(x_unsafe,C)
        ddhDs_dT = Dphi(x_safe,C) @ np.einsum("ijk->ikj", Dphi(x_safe,C)) @ theta / np.linalg.norm(np.einsum("ijk->ikj", Dphi(x_safe,C)) @ theta, ord=2, axis=1, keepdims=True)
        ddhDb_dT = Dphi(x_buffer,C) @ np.einsum("ijk->ikj", Dphi(x_buffer,C)) @ theta / np.linalg.norm(np.einsum("ijk->ikj", Dphi(x_buffer,C)) @ theta, ord=2, axis=1, keepdims=True)

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
''':::DERIVATIVE REFERENCE:::
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
