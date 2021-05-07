import numpy as np
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

def collision_check(c_out_tch, G_out, obstacle):
    """ Collision Check 

    Compute the (differentiable) collision check value between an output zonotope 
    "obstacle" zonotope by solving a convex program. 
    A value less than 1 implies collision.

    Args:
        c_out_tch: torch tensor
        G_out: np array
        obstacle: Zonotope object
    """
    n = obstacle.dim # dimension

    c_obs = obstacle.c
    G_obs = obstacle.G

    A = np.hstack((G_out, -G_obs))
    m = A.shape[1] # combined order of 2 zonotopes

    z = cp.Variable((m, 1))
    v = cp.Variable()
    c_out = cp.Parameter((2, 1))

    b = c_obs - c_out

    constraints = [cp.pnorm(z, p='inf') <= v,
                   A @ z == b]
    objective = cp.Minimize(v)
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    cvxpylayer = CvxpyLayer(problem, parameters=[c_out], variables=[z, v])

    # solve the problem
    z_opt, v_opt = cvxpylayer(c_out_tch)

    return v_opt

def collision_loss(c_out_tch, G_out, obstacle):
    """ Collision Loss

    A loss function to use to drive 2 zonotopes out of collision
        (1 - v)^2
    where v is the value of the collision check. Thus, minimizing this loss 
    drives v to 1, pushing the 2 zonotopes barely out of collision.

    TODO: (1 + eps - v)^2 so that zonotopes are actually out of collision

    Args:
        c_out_tch: torch tensor
        G_out: np array
        obstacle: Zonotope object
    """
    v = collision_check(c_out_tch, G_out, obstacle)
    loss = torch.square(1 - v)

    return loss