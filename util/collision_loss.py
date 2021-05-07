import numpy as np

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

# Compute the (differentiable) collision check loss between an output zonotope 
# "obstacle" zonotope
#
# Output zonotope is parameterized by its center
# Obstacle zonotope is a zonotope class object
def collision_loss(output_cen, output_gen, obstacle):
    n = obstacle.dim # dimension
    m = 4 # combined order of 2 zonotopes

    c_obs = obstacle.c
    G_obs = obstacle.G
    G2 = np.array([[0.5, 0.5],[-0.5, 0.5]])

    A = np.hstack((G1, -G2))

    z = cp.Variable((m, 1))
    v = cp.Variable()
    c2 = cp.Parameter((2, 1))

    b = c2 - c1

    constraints = [cp.pnorm(z, p=1) <= v,
                   A @ z == b]
    objective = cp.Minimize(v)
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    cvxpylayer = CvxpyLayer(problem, parameters=[c2], variables=[z, v])

    # solve the problem
    z_opt, loss = cvxpylayer(c2_tch)

    return loss