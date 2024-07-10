import numpy as np

def dynamics_RK4(X, U, ODE_RHS, DT):
    # RK4 integrator
    k1 = ODE_RHS(X, U)
    k2 = ODE_RHS(X + DT/2*k1, U)
    k3 = ODE_RHS(X + DT/2*k2, U)
    k4 = ODE_RHS(X + DT*k3, U)
    X = X + DT / 6 * (k1 + 2*k2 + 2*k3 + k4)
    return X
