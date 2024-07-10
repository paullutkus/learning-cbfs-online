import numpy     as np
import cvxpy     as cp
import casadi    as ca

#import jax.numpy as jnp
#import jax

#from dynamics import f, g -- [use hjr dynamics instead]
a=1e-4

def phi(x, C, bf=31, s=1): # is a column vector

    r = np.linalg.norm(x - C, ord=2, axis=1)

    if bf == 31:
        ########## 3, 1 #############################
        return np.maximum(0, s - r)**4 * (1 + 4*r)/20
        #############################################

    if bf == 32: 
        ########### 3, 2 ##########################################
        # C has shape (n, d)
        # want x to have shape (d, 1)
        return np.maximum(0, s - r)**6 * (3 + 18*r + 35*r**2)/1680
        ##########################################################
'''
    x = np.array(x)
    x = x.reshape(1, -1)

    # output shape should be (1, n)
    return  np.sqrt(a + np.sum(np.square(x - C), axis=1, keepdims=True))
'''


def Dphi(x, C, bf=31, s=1): # is a row vector 
    r   = np.linalg.norm(x - C, ord=2, axis=1)
    msk = np.sign(np.maximum(0, s - r))
    x = np.array(x)
    x = x.reshape(1, -1)

    if bf == 31:
        ########### 3, 2 ############################################################
        dwdr = (msk*(-4*(s - r)**3 * (1 + 4*r)    / 20 +\
                      4*np.maximum(0, s -   r)**4 / 20) ).reshape(-1, 1) 
        Dphi = dwdr * (x - C) / (a+r.reshape(-1, 1))
        return Dphi
        #############################################################################

    if bf == 32:
        ########### 3, 1 ############################################################

        Dphi = ( msk*(-6*(s - r)**5 * (1 + 18*r + 35*r**2)/1680 + \
                                      (    18   + 70*r   )/1680) ).reshape(-1, 1) * \
               (x - C) / np.sqrt(a + np.sum(np.square(x - C), axis=1, keepdims=True))
        return Dphi
        #############################################################################
'''
    # output shape should be (n, d)
    x = np.array(x)

    return (x - C) / phi(x, C)
'''


def hjoint(x, thetas, centers, argmax=False, s=1):
    x = np.array(x)
    hs = []
    N  = len(thetas)
    for i in range(N):
        hs.append(phi(x, centers[i], s=s).T @ thetas[i])
    if argmax == False:
        return max(hs)
    else: 
        return max(hs), np.argmax(hs)


def cas_train_cbf(x_safe, x_buffer, x_unsafe,
                  C, theta_max, max_elem, rx, rc, gamma_safe,
                                          gamma_unsafe,
                                          gamma_dyn,
                                          s,
                                          dynamics,
                                          verbose=False):

    f = dynamics.open_loop_dynamics
    g = dynamics.control_jacobian

    opti = ca.Opti()
    cons = []

    n = C.shape[0]
    theta       = opti.variable(n, 1)
    theta_cost  = ca.sumsqr(theta)
    

    for x in x_safe:
        cons.append(ca.mtimes(theta.T, phi(x, C,s=s)) >= gamma_safe)
        cons.append(ca.mtimes(ca.mtimes(theta.T, Dphi(x, C, s=s)), f(x,0)) + ca.mtimes(theta.T, phi(x, C, s=s)) +\
                    ca.norm_2(ca.mtimes(ca.mtimes(np.array(g(x,0).T), Dphi(x, C, s=s).T), theta)) >= gamma_dyn)

    for x in x_buffer:
        cons.append(ca.mtimes(ca.mtimes(theta.T, Dphi(x, C, s=s)), f(x,0)) + ca.mtimes(theta.T, phi(x, C, s=s)) +\
                    ca.norm_2(ca.mtimes(ca.mtimes(np.array(g(x,0).T), Dphi(x, C, s=s).T), theta)) >= gamma_dyn)

    for x in x_unsafe:
        cons.append(ca.mtimes(theta.T, phi(x, C,s=s)) <= gamma_unsafe)

    # ca.sum2 enforces each elment less than theta_max?
    #cons.append(ca.sum1(theta) < -theta_max)

    #K = (rx-rc) / np.sqrt(a+(rx-rc)**2)
    #print(K)
    #R = 1 / (K*np.sqrt(1-(rc/rx)**2))
    #print(R)
    #print((R-1) * theta_max / (n*(R+1)))

    
    #for i in range(n):
    #    cons.append(theta[i] <=  max_elem)
    #    cons.append(theta[i] >= -max_elem)
        #cons.append(theta[i] <=      (R-1) * theta_max / (n*(R+1)))
        #cons.append(theta[i] >= -1 * (R-1) * theta_max / (n*(R+1)))

    #cons.append( ca.norm_inf(theta) <= (R-1) * theta_max / (n*(R+1)) )

    opti.minimize(theta_cost)
    opti.subject_to(cons)
    opti.set_initial(theta, ca.DM.rand(C.shape[0]))
    p_opts = {"expand" : True, "verbose" : verbose, "print_time" : verbose}
    s_opts = {"max_iter" : 1000, "print_level" : 5*int(verbose)}
    opti.solver('ipopt', p_opts, s_opts)
    sol = opti.solve()

    return sol.value(theta)




def cvx_train_cbf(c, x_safe, u_safe, x_unsafe, x_buffer, u_buffer, params):
    theta_dim = c.shape[0]
    theta     = cp.Variable((theta_dim, 1))

    # constraints
    cons = []

    gamma_dyn    = params["gamma_dyn"   ]
    gamma_safe   = params["gamma_safe"  ]
    gamma_unsafe = params["gamma_unsafe"]
    sum_theta    = params["sum_theta"   ]
    max_theta    = params["max_theta"   ]
    dynamics     = params["dynamics"    ]
    solver       = params["solver"      ]
    s            = params["s"           ]

    f = dynamics.open_loop_dynamics
    g = dynamics.control_jacobian

    for x, u in zip(x_safe, u_safe):
        cons.append(  phi(x,c,s=s).T @ theta >= gamma_safe)
        cons.append((Dphi(x,c,s=s).T @ theta).T @ (f(x,0) + g(x,0) @ u) \
                    + phi(x,c,s=s).T @ theta >= gamma_dyn )

    for x in x_unsafe:
        cons.append(phi(x,c,s=s).T @ theta <= gamma_unsafe)

    for x, u in zip(x_buffer, u_buffer):
        cons.append((Dphi(x,c,s=s).T @ theta).T @ (f(x,0) + g(x,0) @ u) \
                    + phi(x,c,s=s).T @ theta >= gamma_dyn )

    #cons.append(cp.sum(theta) <= sum_theta)
    #cons.append(cp.norm(theta, "inf") <= max_theta)

    # objective
    obj  = cp.Minimize(cp.sum_squares(theta))

    prob = cp.Problem(obj, cons)
    prob.solve(verbose=True, max_iter=1000000, solver=solver)

    return theta.value

