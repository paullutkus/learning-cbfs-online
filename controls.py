import jax.numpy as jnp
import numpy as np
import cvxpy as cp
import hj_reachability as hj
import clarabel
from scipy.integrate import solve_ivp
from scipy           import sparse
from rbf             import get_Dphi, get_Dphi_curr, get_h, get_h_curr
from jax             import vmap



def get_xd_mpc(a, dt=0.01, bicycle=False):
    dynamics = a.dynamics
    g        = dynamics.control_jacobian
    umax     = a.umax
    Df       = dynamics.Df
    solver   = a.solver

    def xd_mpc(xi, xd, T=0.5):
        #dt   = 0.01
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

        if bicycle:
            #_diag = np.ones(k*xdim)
            _diag = np.array(k*(1, 1, 0))
            P = np.diag(_diag)
        else:
            P = np.eye(k*xdim)
        obj  = cp.Minimize(cp.quad_form(cp.vec(xe), P) + 0*cp.quad_form(cp.vec(u), np.eye(k*udim)) )
        prob = cp.Problem(obj, cons)
        prob.solve(solver=solver, verbose=False)
        return u.value[0,:]

    return xd_mpc



def get_slack_safety_filter(a, eps=0):
    dynamics = a.dynamics
    f        = dynamics.open_loop_dynamics
    g        = dynamics.control_jacobian
    umax     = a.umax
    udim     = dynamics.disturbance_space.lo.shape[0]
    thetas   = np.array(a.thetas)
    centers  = np.array(a.centers)
    s        = a.s
    gamma    = a.gamma
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
        u    = cp.Variable(udim + 1) # slack variable
        #ue   = u - np.concatenate((ud, np.array([0])))
        ue = u[:-1] - ud
        obj  = cp.Minimize( cp.quad_form(ue, np.eye(udim)) + 1e7*u[-1])
        #print("x", x)A
        hmax, i = h(x[np.newaxis,...], centers, thetas)
        hmax = hmax.item()
        i = i.item()
        #grad = thetas[i].T @ Dphi(x[np.newaxis,...], centers[i])
        #Lf = grad @ f(x,0)
        #Lg = np.hstack((grad @ g(x,0), np.array([[1]]))) @ u
        #cons.append(Lf + Lg + gamma*hmax >= eps)
        cons.append(thetas[i].T @ Dphi(x[np.newaxis,...], centers[i]) @ (f(x,0) + g(x,0) @ u[:-1]) +\
                    gamma*hmax + u[-1] >= eps)
        cons.append(u[:-1] <=  umax)
        cons.append(u[:-1] >= -umax)
        cons.append(u[-1] >= 0)
        #cons.append(u[-1] == 5)

        prob = cp.Problem(obj, cons)
        prob.solve(solver=solver, verbose=False)
        if u.value is not None:
            if (u.value[:-1] > umax).any():
                print("val", u.value, "max", umax)
        #print("safe u", u.value)
        #print("slack variable", u.value[-1])
        return u.value[:-1], u.value[-1]
    
    return safety_filter



def get_safety_filter(a, eps=0):
    dynamics = a.dynamics
    f        = dynamics.open_loop_dynamics
    g        = dynamics.control_jacobian
    umax     = a.umax
    udim     = dynamics.disturbance_space.lo.shape[0]
    thetas   = np.array(a.thetas)
    centers  = np.array(a.centers)
    s        = a.s
    gamma    = a.gamma
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
        u    = cp.Variable(udim) # slack variable
        ue = u - ud
        obj  = cp.Minimize( cp.quad_form(ue, np.eye(udim)) )
        #print("x", x)
        hmax, i = h(x[np.newaxis,...], centers, thetas)
        hmax = hmax.item()
        i = i.item()
        grad = thetas[i].T @ Dphi(x[np.newaxis,...], centers[i])
        #Lf = grad @ f(x,0)
        #Lg = np.hstack((grad @ g(x,0), np.array([[1]]))) @ u
        #cons.append(Lf + Lg + gamma*hmax >= eps)
        cons.append(thetas[i].T @ Dphi(x[np.newaxis,...], centers[i]) @ (f(x,0) + g(x,0) @ u) + \
                    gamma*hmax >= eps)
        cons.append(u <=  umax)
        cons.append(u >= -umax)
        #cons.append(u[-1] >= 0)

        prob = cp.Problem(obj, cons)
        prob.solve(solver=solver, verbose=False)
        if u.value is not None:
            if (u.value > umax).any():
                print("val", u.value, "max", umax)
        #print("safe u", u.value)
        #print("slack variable", u.value[-1])
        return u.value#[:-1]
    
    return safety_filter



def get_cbvf_safety_filter(a, grid, V):
    dynamics = a.dynamics
    f        = dynamics.open_loop_dynamics
    g        = dynamics.control_jacobian
    umax     = a.umax
    udim     = dynamics.disturbance_space.lo.shape[0]
    s        = a.s
    gamma    = a.gamma
    solver  = a.solver

    def cbvf_safety_filter(x, ud):
        cons = []
        u    = cp.Variable(udim + 1)
        ue   = u[:-1] - ud
        obj  = cp.Minimize( cp.quad_form(ue, np.eye(udim)) + 1e7*u[-1])
        #hmax, i = h(x, centers, thetas)
        #print("hmax", hmax) 
        #print("x", x)
        cons.append(grid.interpolate(grid.grad_values(V), x) @ (f(x,0) + g(x,0) @ u[:-1]) + \
                    gamma*grid.interpolate(V, x) + u[-1] >= 0)
        cons.append(u <=  umax)
        cons.append(u >= -umax)
        cons.append(u[-1] >= 0)

        prob = cp.Problem(obj, cons)
        prob.solve(solver=solver, verbose=False)
        if u.value[-1] >= 0.01:
            print("slack:", u.value[-1])
        #print("safe u", u.value)
        return u.value[:-1], u.value[-1]
    
    return cbvf_safety_filter



def cbf_controls(a, x):
    dynamics = a.dynamics
    utype    = a.utype
    umax     = a.umax

    g  = dynamics.control_jacobian
    gx = g(x, 0)

    h = get_h_curr(a)
    Dphi = get_Dphi_curr(a)
    _, argmax = h(x)

    u   = cp.Variable(gx.shape[1])
    #print((a.thetas[argmax] @ Dphi(x)).shape)
    c = a.thetas[argmax] @ Dphi(x) @ gx @ u

    cns = [] 
    if   utype == "ball":
        cns.append(cp.norm(u) <= umax)
    elif utype == "box":
        for i in range(gx.shape[1]):
            cns.append(u[i] <= umax)
            cns.append(u[i] >=-umax)

    prb = cp.Problem(cp.Maximize(c), cns)
    prb.solve(verbose=False, solver='CLARABEL')
    
    return u.value



def cbf_controls_parallel(a, data, N_part=10, verbose=False):
    dynamics = a.dynamics
    utype    = a.utype
    umax     = a.umax
    g  = dynamics.control_jacobian
    gv = vmap(g, in_axes=(0,None))
    N = data.shape[0]
    print("num data", N)
    partition = int(N / N_part)
    print("partition", partition)
    remainder = N - (N_part * partition)
    print("remainder", N % partition)

    
    u = []
    for i in range(N_part + 1):
        if i == N_part and remainder != 0:
            idx_start = i * partition
            idx_end = idx_start + remainder
        elif i == N_part and remainder == 0:
            break
        else:
            idx_start = i * partition
            idx_end = (i+1) * partition 
        X = data[idx_start:idx_end,:]
        gX = gv(X,0)
        h = get_h_curr(a)
        Dphi = get_Dphi_curr(a)
        dhX = []
        '''
        for x in X: 
            x = np.expand_dims(x, 0)
            _, argmax = h(x); argmax = argmax.item()
            dh = a.thetas[argmax] @ Dphi(x)
            dhX.append(dh)
        '''
        _, argmax = h(X)
        thetas_max = np.array(a.thetas)[argmax, :]
        dhX = np.einsum('ij,ijk->ik', thetas_max, Dphi(X))
        P = sparse.csc_array(np.diag(np.zeros(X.shape[0] * gX.shape[-1])))
        print("P shape", P.shape)
        q = np.einsum('ij,ij->i', dhX, gX.squeeze())
        print("q shape", q.shape)
        #cTu = c.T @ u 

        if utype == "box":
            pos = np.diag(X.shape[0]*gX.shape[-1]*( 1,))
            neg = np.diag(X.shape[0]*gX.shape[-1]*(-1,))
            A = sparse.csc_array(np.vstack([np.vstack((pos[i,:], neg[i,:],)) for i in range(X.shape[0]*gX.shape[-1])]))
            print("A shape", A.shape)
            b = np.tile((*gX.shape[-1]*(umax,),*gX.shape[-1]*(umax,)), X.shape[0])
            print("b shape", b.shape)
        cones = [clarabel.NonnegativeConeT(A.shape[0])]

        settings = clarabel.DefaultSettings()
        solver   = clarabel.DefaultSolver(P, q, A, b, cones, settings)
        solution = solver.solve()

        if i == 0:
            U = np.array(solution.x).reshape(-1, gX.shape[-1])
        else:
            U = np.vstack((U, np.array(solution.x).reshape(-1, gX.shape[-1])))

    return U



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


def hjb_controls_parallel_partitioned(a, data, grid, V, N_part=10, verbose=False):
    dynamics = a.dynamics
    utype    = a.utype
    umax     = a.umax
    g  = dynamics.control_jacobian
    gv = vmap(g, in_axes=(0,None))
    N = data.shape[0]
    print("num data", N)
    partition = int(N / N_part)
    print("partition", partition)
    remainder = N - (N_part * partition)
    print("remainder", N % partition)

    for i in range(N_part+1):
        if i == N_part and remainder != 0:
            idx_start = i * partition
            idx_end = idx_start + remainder
        elif i == N_part and remainder == 0:
            break
        else:
            idx_start = i * partition
            idx_end = (i+1) * partition 
        X = data[idx_start:idx_end,:]
        gX = gv(X,0)
        '''
        h = get_h_curr(a)
        Dphi = get_Dphi_curr(a)
        dhX = []
        for x in X: 
            _, argmax = h(x)
            dh = a.thetas[argmax] @ Dphi(x)
            dhX.append(dh)
        dhX = np.array(dhX)
        '''
        interpv = vmap(grid.interpolate, in_axes=(None, 0))
        dVX  = interpv(grid.grad_values(V), X)

        print("dVX shape", dVX[:,np.newaxis,:].shape)
        print("gX shape", gX.shape)
        #u   = cp.Variable(X.shape[0] * gX.shape[-1])
        P = sparse.csc_array(np.diag(np.zeros(X.shape[0] * gX.shape[-1])))
        print("P shape", P.shape)
        q = np.einsum('ijk,ikl->ijl', dVX[:,np.newaxis,:], gX).squeeze().reshape(-1)
        print("q shape", q.shape)
        #cTu = c.T @ u 

        if utype == "box":
            pos = np.diag(X.shape[0]*gX.shape[-1]*( 1,))
            neg = np.diag(X.shape[0]*gX.shape[-1]*(-1,))
            A = sparse.csc_array(np.vstack([np.vstack((pos[i,:], neg[i,:],)) for i in range(X.shape[0]*gX.shape[-1])]))
            print("A shape", A.shape)
            b = np.tile((*gX.shape[-1]*(umax,),*gX.shape[-1]*(umax,)), X.shape[0])
            print("b shape", b.shape)
        cones = [clarabel.NonnegativeConeT(A.shape[0])]

        settings = clarabel.DefaultSettings()
        solver   = clarabel.DefaultSolver(P, q, A, b, cones, settings)
        solution = solver.solve()

        if i == 0:
            U = np.array(solution.x).reshape(-1, gX.shape[-1])
        else:
            U = np.vstack((U, np.array(solution.x).reshape(-1, gX.shape[-1])))

    return U

