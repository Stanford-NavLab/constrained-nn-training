import numpy as np
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from .NN_con_zono import forward_pass_NN_con_zono_torch

def center_param_collision_check(c_out_tch, G_out, obstacle):
    """ Center-parameterized Collision Check 

    Compute the (differentiable) collision check value between an output zonotope 
    parameterized by its center and "obstacle" zonotope by solving a convex program. 
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

def center_param_collision_loss(c_out_tch, G_out, obstacle):
    """ Center-parameterized Collision Loss

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
    v = center_param_collision_check(c_out_tch, G_out, obstacle)
    loss = torch.square(1 - v)

    return loss


def torch_collision_check(Z_out, Z_obs):
    """ Torch Collision Check 

    Compute the (differentiable) collision check value between an output zonotope 
    and "obstacle" zonotope by solving a convex program. 
    A value less than 1 implies collision.

    Args:
        Z_out: TorchConstrainedZonotope object
        Z_obs: TorchConstrainedZonotope object
    """
    # Compute intersection constrained zonotope
    Z_int = Z_out.intersect(Z_obs)

    # Extract A and b constraints from intersection zonotope
    A_con = Z_int.A
    b_con = Z_int.b

    # Form CVX problem
    m = A_con.shape[1] 

    z = cp.Variable((m, 1))
    v = cp.Variable()
    A = cp.Parameter(A_con.shape)
    b = cp.Parameter(b_con.shape)

    constraints = [cp.pnorm(z, p='inf') <= v,
                   A @ z == b]
    objective = cp.Minimize(v)
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    cvxpylayer = CvxpyLayer(problem, parameters=[A,b], variables=[z,v])

    # solve the problem
    z_opt, v_opt = cvxpylayer(A_con,b_con)

    return v_opt


def NN_constraint_loss(Z_in, Z_out_con, net):
    # extract weights and biases from network
    NN_weights = []
    NN_biases = []

    idx = 0
    for param in net.parameters():
        if idx % 2 == 0: # "even" parameters are weights
            NN_weights.append(param)
        else: # "odd" parameters are biases
            NN_biases.append(param[:,None])
        idx += 1
    
    Z_out = forward_pass_NN_con_zono_torch(Z_in, NN_weights, NN_biases)

    for z in Z_out:
        v = torch_collision_check(z, Z_out_con)
        if v <= 1: # in collision
            v.backward() # backprop

