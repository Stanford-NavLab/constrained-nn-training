import numpy as np
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from .NN_con_zono import forward_pass_NN_torch

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


def NN_constraint_step(Z_in, Z_obs, net, con_opt):
    """ NN Constraint Step

    Compute the forward pass of an input zonotope thru a network, then evaluate the 
    collision check for all of the output zonotopes, determine which are in collision,
    then run gradients updates for those.

    Args:
        Z_in: TorchZonotope or TorchConstrainedZonotope object
        Z_out_con: TorchZonotope or TorchConstrainedZonotope object
        net: torch network (nn.Module)
        con_opt: torch optimizer
    """
    con_opt.zero_grad()

    Z_out = forward_pass_NN_torch(Z_in, net)

    losses = []

    for z in Z_out:
        v = torch_collision_check(z, Z_obs)
        if v <= 1: # in collision
            print("in collision")
            loss = torch.square(1 - v)
            losses.append(loss)

    if losses:
        total_loss = sum(losses)          
        total_loss.backward() # backprop
        con_opt.step()

    # v1 = torch_collision_check(Z_out[0], Z_obs)
    # print("in collision")
    # con_opt.zero_grad()
    # loss1 = torch.square(1 - v1)
    # loss1.backward(retain_graph=True) # backprop
    # con_opt.step()

    # v2 = torch_collision_check(Z_out[1], Z_obs)
    # print("in collision")
    # con_opt.zero_grad()
    # loss2 = torch.square(1 - v2)
    # loss2.backward(retain_graph=True) # backprop
    # con_opt.step()


