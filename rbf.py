import jax
import jax.numpy as jnp
import     numpy as  np
from jax.tree_util import Partial
from distances import cylindrical_metric


w = 1e-4


def get_h(a, jax_f=False, almost_active=False):
    b  = a.b
    bf = a.bf
    s  = a.s 
    eps = a.eps
    thscl = a.theta_scale
    if jax_f or a.jax_f:
        if a.almost_active:
            def h(x, C, theta):
                h_vec = jnp.einsum('ijk,jk->j', phiT(x[jnp.newaxis,:], C, bf, s, thscl=thscl, jax_f=True), theta)
                idx = jnp.argsort(h_vec, descending=True)
                almost_active_set = (h_vec[idx][0] - h_vec[idx] <= eps)
                # instead of i = idx[almost_active_set]
                i = jnp.where(almost_active_set, x=idx, y=-jnp.ones_like(idx))
                return h_vec[i[0]] + b, i[0]
        else:
            def h(x, C, theta):
                h_vec = jnp.einsum('ijk,jk->j', phiT(x[jnp.newaxis,:], C, bf, s, thscl=thscl, jax_f=True), theta)
                i = jnp.argmax(h_vec)
                return h_vec[i] + b, i
        return jax.vmap(h, in_axes=(0, None, None))
    else:
        def h(x, C, theta):
            h_vec = np.einsum('ijk,jk->ij', phiT(x, C, bf, s, thscl=thscl, jax_f=False), theta)
            i = np.argmax(h_vec, axis=1)
            return h_vec[np.arange(h_vec.shape[0]), i] + b, i
        return h


def get_h_curr(a, jax_f=False):
    b     = a.b
    bf    = a.bf
    s     = a.s 
    eps   = a.eps
    thscl = a.theta_scale
    C     = np.array(a.centers)
    theta = np.array(a.thetas)
    if jax_f or a.jax_f:
        if a.almost_active:
            def h(x):
                h_vec = jnp.einsum('ijk,jk->j', phiT(x[jnp.newaxis,:], C, bf, s, thscl=thscl, jax_f=True), theta)
                idx = jnp.argsort(h_vec, descending=True)
                almost_active_set = (h_vec[idx][0] - h_vec[idx] <= eps)
                # instead of i = idx[almost_active_set]
                i = jnp.where(almost_active_set, x=idx, y=-jnp.ones_like(idx))
                return h_vec[i[0]] + b, i[0]
        else: 
            def h(x):
                h_vec = jnp.einsum('ijk,jk->j', phiT(x[jnp.newaxis,:], C, bf, s, thscl=thscl, jax_f=True), theta)
                i = jnp.argmax(h_vec)
                return h_vec[i] + b, i
        return jax.vmap(h, in_axes=(0))
    else:
        def h(x):
            h_vec = np.einsum('ijk,jk->ij', phiT(x, C, bf, s, thscl=thscl, jax_f=False), theta)
            i = np.argmax(h_vec, axis=1)
            return h_vec[np.arange(h_vec.shape[0]), i] + b, i
        return h


def phiT(x, C, bf, s, thscl=None, jax_f=False):
    if jax_f:
        if bf == 31: 
            phiTv = jax.vmap(jax_phiT, in_axes=(0, None, None, None))
            return phiTv(x, C, s, thscl)
        if bf == 2:
            phiTv = jax.vmap(jax_multiquadric, in_axes=(0, None, None, None))
            return phiTv(x, C, s, thscl)
    if thscl is not None:
        r = cylindrical_metric(x[...,np.newaxis,:], C, a=thscl, phi_eval=True)[:,np.newaxis,:]
    else:
        r = np.linalg.norm(x[...,np.newaxis,:] - C, ord=2, axis=-1)[:,np.newaxis,:]
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
    if bf == 51:
        phi = np.maximum(0, s - r)**5 * (1 + 5*r)/30
        return phi


def get_phiT(a):
    bf = a.bf
    s  = a.s
    thscl = a.theta_scale
    jax_f = a.jax_f
    if jax_f:
        if bf == 31:
            phiT = jax_get_phiT(a)
            phiTv = jax.vmap(phiT, in_axes=(0, None))
            return phiTv # only supports 31
        if bf == 2:
            phiT = jax_get_multiquadric(a)
            phiTv = jax.vmap(phiT, in_axes=(0, None))
            return phiTv 
    def phiT(x, C): # is a column vector
        if thscl is not None:
            r = cylindrical_metric(x[...,np.newaxis,:], C, a=thscl, phi_eval=True)[:,np.newaxis,:]
        else:
            r = np.linalg.norm(x[...,np.newaxis,:] - C, ord=2, axis=-1)[:,np.newaxis,:]
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
        if bf == 51:
            phi = np.maximum(0, s - r)**5 * (1 + 5*r)/30
            return phi
    return phiT


def get_phiT_curr(a):
    bf = a.bf
    s  = a.s
    C  = a.centers[-1]
    xdim = C.shape[1]
    thscl = a.theta_scale
    jax_f = a.jax_f
    if jax_f:
        if bf == 31:
            phiT = jax_get_phiT_curr(a)
            phiTv = jax.vmap(phiT, in_axes=(0, None))
            return phiTv 
        if bf == 2:
            phiT = jax_get_multiquadric_curr(a)
            phiTv = jax.vmap(phiT, in_axes=(0, None))
            return phiTv
    def phiT(x): # is a column vector
        if thscl is not None:
            r = cylindrical_metric(x[...,np.newaxis,:], C, a=thscl, phi_eval=True)[:,np.newaxis,:]
        else:
            r = np.linalg.norm(x[...,np.newaxis,:] - C, ord=2, axis=-1)[:,np.newaxis,:]
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
        if bf == 51:
            phi = np.maximum(0, s - r)**5 * (1 + 5*r)/30
    return phiT


def get_Dphi(a):
    bf = a.bf
    s  = a.s
    thscl = a.theta_scale
    jax_f = a.jax_f
    if jax_f:
        if bf == 31:
            phiT = jax_get_phiT(a)
            Dphi = jax.jacfwd(phiT, 0)
            Dphiv = jax.vmap(Dphi, in_axes=(0,None))
            return Dphiv
        if bf == 2:
            phiT = jax_get_multiquadric(a)
            Dphi = jax.jacfwd(phiT, 0)
            Dphiv = jax.vmap(Dphi, in_axes=(0,None))
            return Dphiv
    def Dphi(x, C): # is a row vector 
        if thscl is not None:
            r = cylindrical_metric(x[...,np.newaxis,:], C[np.newaxis,...], phi_eval=True)
        else:
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
            dwdr = (msk*(-6*(s-r)**5 * (1 + 18*r + 35*r**2)  /1680 +\
                                       (    18   + 70*r   )  /1680) )
            Dphi = dwdr[...,np.newaxis] * (x[...,np.newaxis,:] - C[np.newaxis,...]) / (1e-5+r)[...,np.newaxis]
            return Dphi
        if bf == 51:
            dwdr = msk*(-5*(s-r)**4 * (1+5*r)     / 30 +\
                         4*np.maximum(0, s-r)**5 / 30) 
            Dphi = dwdr[...,np.newaxis] * (x[...,np.newaxis,:] - C[np.newaxis,...]) / (1e-5+r)[...,np.newaxis]
            return Dphi
    return Dphi


def get_Dphi_curr(a):
    bf = a.bf
    s  = a.s
    C  = a.centers[-1]
    thscl = a.theta_scale
    jax_f = a.jax_f
    if jax_f:
        if bf == 31:
            phiT = jax_get_phiT_curr(a)
            Dphi = jax.jacfwd(phiT, 0)
            Dphiv = jax.vmap(Dphi, in_axes=(0,))
            return Dphiv
        if bf == 2:
            phiT = jax_get_multiquadric_curr(a)
            Dphi = jax.jacfwd(phiT, 0)
            Dphiv = jax.vmap(Dphi, in_axes=(0,))
            return Dphiv
    def Dphi(x): # is a row vector 
        if thscl is not None:
            r = cylindrical_metric(x[...,np.newaxis,:], C[np.newaxis,...], phi_eval=True)
        else:
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
        if bf == 51:
            dwdr = msk*(-5*(s-r)**4 * (1+5*r)     / 30 +\
                         4*np.maximum(0, s-r)**5 / 30)
            Dphi = dwdr[...,np.newaxis] * (x[...,np.newaxis,:] - C[np.newaxis,...]) / (1e-5+r)[...,np.newaxis]
            return Dphi
    return Dphi


def jax_phiT(x, C, s, thscl=None): # is a column vector
    if thscl is not None:
        D = x[...,jnp.newaxis,:] - C
        r = jnp.sqrt(w + D[...,0]**2 + D[...,1]**2 + jnp.minimum(thscl*D[...,2], thscl*(2*jnp.pi-D[...,2]))**2)
    else:
        D = x[...,jnp.newaxis,:] - C
        r = jnp.sqrt(w + D[...,0]**2 + D[...,1]**2)
    phi = (jnp.maximum(0, s - r)**4 * (1 + 4*r)/20)
    return phi


def jax_get_phiT(a):
    s  = a.s
    thscl = a.theta_scale
    phiT = Partial(jax_phiT, s=s, thscl=thscl)
    return phiT


def jax_get_phiT_curr(a):
    s  = a.s
    C  = a.centers[-1]
    thscl = a.theta_scale
    phiT = Partial(jax_phiT, C=C, s=s, thscl=thscl)
    return phiT


def jax_multiquadric(x, C, s, thscl):
    D = x[jnp.newaxis,:] - C
    r = jnp.sqrt(w + D[...,0]**2 + D[...,1]**2 + jnp.minimum(thscl*D[...,2], thscl*(2*jnp.pi-D[...,2]))**2)
    phi = r
    return phi


def jax_get_multiquadric(a):
    s = a.s
    thscl = a.theta_scale
    phiT = Partial(jax_multiquadric, s=s, thscl=thscl)
    return phiT


def jax_get_multiquadric_curr(a):
    s = a.s
    C = a.centers[-1]
    thscl = a.theta_scale
    phiT = Partial(jax_multiquadric, C=C, s=s, thscl=thscl)
    return phiT

