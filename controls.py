import numpy as np
import cvxpy as cp
import hj_reachability as hj
from scipy.integrate import solve_ivp
from optim           import get_Dphi, get_h
from jax             import vmap


def get_xd_mpc(a):
    dynamics = a.dynamics
    g        = dynamics.control_jacobian
    umax     = a.umax
    Df       = dynamics.Df
    solver   = a.solver

    def xd_mpc(xi, xd, T=0.5):
        dt   = 0.05
        k    = int(T / dt)
        A    = Df(xi)
        B    =  g(xi, 0)

        xdim = xd.shape[0]
        udim =  B.shape[1]
        x    = cp.Variable((k, xdim))
        xe   = x - xd.reshape(1, -1)
        u    = cp.Variable((k, udim))    

        cons = []
        cons.append(x[0] == xi)
        for t in range(k-1):
            cons.append(dt * (A@x[t] + B@u[t]) + x[t] == x[t+1])
            cons.append(u[t] <=  umax)
            cons.append(u[t] >= -umax)

        obj  = cp.Minimize( cp.quad_form(cp.vec(xe), np.eye(k*xdim)) + 0*cp.quad_form(cp.vec(u), np.eye(k*udim)) )
        prob = cp.Problem(obj, cons)
        prob.solve(solver=solver, verbose=False)
        return u.value[0,:]

    return xd_mpc


def get_safety_filter(a, eps=0):
    dynamics = a.dynamics
    f        = dynamics.open_loop_dynamics
    g        = dynamics.control_jacobian
    umax     = a.umax
    udim     = dynamics.disturbance_space.lo.shape[0]
    thetas   = np.array(a.thetas)
    centers  = np.array(a.centers)
    s        = a.s
    h        = get_h(a)
    Dphi     = get_Dphi(a)
    solver  = a.solver

    def safety_filter(x, ud):
        #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        #########################################
        ### NEED TO ADD THE ALMOST-ACTIVE SET ###
        #########################################
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #print("state:", x)
        #print("desired control:", ud)
        cons = []
        u    = cp.Variable(udim)
        ue   = u - ud
        obj  = cp.Minimize( cp.quad_form(ue, np.eye(udim)) )
        hmax, i = h(x, centers, thetas)
        #print("hmax", hmax) 
        cons.append(thetas[i].T @ Dphi(x, centers[i]) @ (f(x,0) + g(x,0) @ u) + \
                    hmax >= eps)
        cons.append(u <=  umax)
        cons.append(u >= -umax)

        prob = cp.Problem(obj, cons)
        prob.solve(solver=solver, verbose=False)
        if u.value is not None:
            if (u.value > umax).any():
                print("val", u.value, "max", umax)
        #print("safe u", u.value)
        return u.value
    
    return safety_filter


def hjb_controls(a, x, grid, V, verbose=False):
    dynamics = a.dynamics
    utype    = a.utype
    umax     = a.umax

    g  = dynamics.control_jacobian
    gx = g(x, 0)
    dV  = grid.interpolate(grid.grad_values(V), x)

    u   = cp.Variable(gx.shape[1])
    c   = dV.T @ gx @ u

    cns = [] 
    if   utype == "ball":
        cns.append(cp.norm(u) <= umax)
    elif utype == "box":
        for i in range(gx.shape[1]):
            cns.append(u[i] <= umax)
            cns.append(u[i] >=-umax)

    prb = cp.Problem(cp.Maximize(c), cns)
    prb.solve(verbose=False, solver='CLARABEL')
    
    if verbose:
        print("umax:", umax)
        print("utype:", utype)
        print("cvx result:", dV.T @ gx @ u.value )
        if   utype == "ball":
            print("dual norm result:", umax * np.linalg.norm(gx.T @ dV, ord=2))
        elif utype == "box":
            print("dual norm result:", umax * np.linalg.norm(gx.T @ dV, ord=1))
        
    return u.value


def hjb_controls_parallel(a, X, grid, V, verbose=False):
    dynamics = a.dynamics
    utype    = a.utype
    umax     = a.umax
    
    g  = dynamics.control_jacobian
    gv = vmap(g, in_axes=(0,None))
    gX = gv(X,0)

    interpv = vmap(grid.interpolate, in_axes=(None, 0))
    dVX  = interpv(grid.grad_values(V), X)

    u   = cp.Variable(X.shape[0] * gX.shape[-1])
    c   = np.einsum('ijk,ikl->ijl', dVX[:,np.newaxis,:], gX).squeeze().reshape(-1)
    cTu = c.T @ u 

    if utype == "box":
        P = np.diag(X.shape[0]*gX.shape[-1]*( 1,))
        N = np.diag(X.shape[0]*gX.shape[-1]*(-1,))
        A = np.vstack([np.vstack((P[i,:], N[i,:],)) for i in range(X.shape[0]*gX.shape[-1])])
        b   = np.tile((*gX.shape[-1]*(umax,),*gX.shape[-1]*(umax,)), X.shape[0])
        cns = [A@u <= b]

    prb = cp.Problem(cp.Maximize(cTu), cns)
    prb.solve(verbose=verbose, solver='CLARABEL')
    return u.value.reshape(-1, gX.shape[-1])


