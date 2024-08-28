import jax
#jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from rbf import get_phiT, get_Dphi



def get_L_safe(a, gamma_safe, gamma_dyn, lam_safe, lam_dyn, centers=None):
    if centers is not None:
        C = centers
    else:
        C = a.centers[-1]
    dynamics = a.dynamics
    f = dynamics.open_loop_dynamics
    g = dynamics.control_jacobian
    phi = get_phiT(a)
    Dphi= get_Dphi(a)
    b = a.b
    def L_safe(theta, x_safe, u_safe):
        x_safe = jnp.expand_dims(x_safe,0)
        #print("phi shape", phi(x_safe,C).shape)
        #print("theta shape", theta.shape)
        #print((phi(x_safe, C) @ theta).shape)
        #print("Dphi(x_safe,C) shape", Dphi(x_safe, C).shape)
        #print("theta @ Dphi(x_safe,C) shape", (theta @ Dphi(x_safe,C)).shape)
        #print("(f(x_safe[0],0) + g(x_safe[0],0) @ u_safe)", (f(x_safe[0],0) + g(x_safe[0],0) @ u_safe).shape)
        Lv = lam_safe * jnp.maximum(0, gamma_safe - phi(x_safe,C) @ theta - b)**2
        Ldyn = lam_dyn * jnp.maximum(0, gamma_dyn  - theta @ Dphi(x_safe,C) @ (f(x_safe[0],0) + g(x_safe[0],0) @ u_safe) - phi(x_safe,C) @ theta - b)**2
        L = Lv + Ldyn
        return L[0], (Lv[0], Ldyn[0])
    return jax.vmap(L_safe, in_axes=(None, 0, 0)), jax.vmap(jax.grad(L_safe, argnums=0, has_aux=True), in_axes=(None, 0, 0))


def get_L_unsafe(a, gamma_unsafe, lam_unsafe, centers=None):
    if centers is not None:
        C = centers
    else:
        C = a.centers[-1]
    dynamics = a.dynamics
    f = dynamics.open_loop_dynamics
    g = dynamics.control_jacobian
    phi = get_phiT(a)
    Dphi= get_Dphi(a)
    b = a.b
    def L_unsafe(theta, x_unsafe):
        x_unsafe = jnp.expand_dims(x_unsafe,0)
        L = lam_unsafe * jnp.maximum(0, phi(x_unsafe,C) @ theta + b - gamma_unsafe)**2
        return L[0]
    return jax.vmap(L_unsafe, in_axes=(None, 0)), jax.vmap(jax.grad(L_unsafe, argnums=0), in_axes=(None,0))


def get_L_buffer(a, gamma_dyn, lam_dyn, centers=None):
    if centers is not None:
        C = centers
    else:
        C = a.centers[-1]
    dynamics = a.dynamics
    f = dynamics.open_loop_dynamics
    g = dynamics.control_jacobian
    phi = get_phiT(a)
    Dphi= get_Dphi(a)
    b = a.b
    def L_buffer(theta, x_buffer, u_buffer):
        x_buffer = jnp.expand_dims(x_buffer,0)
        L = lam_dyn * jnp.maximum(0, gamma_dyn  - theta @ Dphi(x_buffer,C) @ (f(x_buffer[0],0) + g(x_buffer[0],0) @ u_buffer) - phi(x_buffer,C) @ theta - b)**2
        return L[0]
    return jax.vmap(L_buffer, in_axes=(None, 0, 0)), jax.vmap(jax.grad(L_buffer, argnums=0), in_axes=(None, 0, 0))


def get_L_dh(a, lam_dh, centers=None):
    if centers is not None:
        C = centers
    else: 
        C = a.centers[-1]
    Dphi = get_Dphi(a)
    def L_dh(theta, x):
        x = jnp.expand_dims(x, 0)
        L = lam_dh * jnp.linalg.norm(theta @ Dphi(x,C))**2
        return L
    return jax.vmap(L_dh, in_axes=(None,0)), jax.vmap(jax.grad(L_dh, argnums=0), in_axes=(None,0))


def get_L_hmax(a, hmax, centers=None):
    if centers is not None:
        C = centers
    else: 
        C = a.centers[-1]
    phi = get_phiT(a)
    b = a.b
    def L_hmax(theta, x_safe):
        x = jnp.expand_dims(x_safe, 0)    
        L = jnp.maximum(0, phi(x,C) @ theta + b - hmax)**2
        return L[0]
    return jax.vmap(L_hmax, in_axes=(None,0)), jax.vmap(jax.grad(L_hmax, argnums=0), in_axes=(None,0))


def jax_get_learning_cbfs_lagrangian(a, x_safe  , u_safe  ,
                                        x_buffer, u_buffer,
                                        x_unsafe, gamma_safe  ,
                                                  gamma_dyn   ,
                                                  gamma_unsafe, lam_safe  ,
                                                                lam_dyn   ,
                                                                lam_unsafe,
                                                                lam_dh    ,
                                                                lam_th    ,
                                                                hmax      ,
                                                                centers=None):

    # JIT these
    print("gamma_safe:", gamma_safe)
    print("gamma_dyn:", gamma_dyn)
    print("gamma_unsafe", gamma_unsafe)
    print("lam safe", lam_safe)
    print("lam_dyn", lam_dyn)
    print("lam_unsafe", lam_unsafe)
    print("lam_dh", lam_dh)
    print("lam_th", lam_th)
    print("hmax", hmax)
    Ls, DLs = get_L_safe(a, gamma_safe, gamma_dyn, lam_safe, lam_dyn, centers=centers)
    Lb, DLb = get_L_buffer(a, gamma_dyn, lam_dyn, centers=centers)
    Lu, DLu = get_L_unsafe(a, gamma_unsafe, lam_unsafe, centers=centers)
    Ldh, DLdh = get_L_dh(a, lam_dh, centers=centers)
    Lhm, DLhm = get_L_hmax(a, hmax, centers=centers)

    def L(theta):
        L_safe, components = Ls(theta,x_safe,u_safe)
        L_safe_v, L_safe_dyn = components
        L_safe=L_safe.sum(); L_safe_v=L_safe_v.sum();L_safe_dyn=L_safe_dyn.sum()
        L_unsafe = Lu(theta,x_unsafe).sum()

        if x_buffer.shape[0] != 0:
            L_buffer = Lb(theta,x_buffer,u_buffer).sum()
            DL_buffer = DLb(theta,x_buffer,u_buffer).sum(axis=0)
        else:
            L_buffer = 0
            DL_buffer = 0

        DL_safe, _ = DLs(theta,x_safe,u_safe)
        DL_safe = DL_safe.sum(axis=0)
        DL_unsafe = DLu(theta,x_unsafe).sum(axis=0)

        L_dh = Ldh(theta, jnp.vstack((x_safe, x_buffer, x_unsafe))).sum()
        DL_dh = DLdh(theta, jnp.vstack((x_safe, x_buffer, x_unsafe))).sum(axis=0)

        L_hm = Lhm(theta, x_safe).sum()
        DL_hm = DLhm(theta, x_safe).sum(axis=0)

        return lam_th * jnp.linalg.norm(theta)**2 + L_safe + L_buffer + L_unsafe + L_dh + L_hm,\
               lam_th * 2 * theta + DL_safe + DL_buffer + DL_unsafe + DL_dh + DL_hm,\
               (L_safe_v, L_safe_dyn, L_buffer, L_unsafe, L_dh, L_hm)

    return L


