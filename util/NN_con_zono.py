import numpy as np
import torch
from util.zonotope import Zonotope
from util.constrained_zonotope import ConstrainedZonotope


def hpint_perm(n):
    """
    This function outputs lists of matrices for calculating all permutations of half-plane intersections in the
    ReLU_con_zono_single function.
    """
    c_new = []
    D_new = []
    H_new = []
    for i in range(2 ** n - 1):
        c_new_i = np.zeros((n, 1))
        binStr = bin(i + 1)[2:]
        for j in range(len(binStr)):
            c_new_i[n - 1 - j][0] = int(binStr[len(binStr) - 1 - j])
        c_new.append(c_new_i)
        D_new_i = np.diag(np.transpose(c_new_i)[0])
        D_new.append(D_new_i)
        H_new_i = np.diag(np.transpose(c_new_i * (-2) + 1)[0])
        H_new.append(H_new_i)

    return c_new, D_new, H_new


def ReLU_con_zono_single(Z_in):
    """
    INPUT:
    Z_in: A single constrained zonotope of class ConstrainedZonotope from constrained_zonotope.py.

    OUTPUT:
    Z_out: A list of constrained zonotopes of class ConstrainedZonotope from constrained_zonotope.py as a result of
    ReLU-activating Z_in.
    """
    # SETUP
    # Get the zonotope parameters
    c = Z_in.c
    G = Z_in.G
    A = Z_in.A
    b = Z_in.b

    # Get dimension of things
    n = c.shape[0]
    n_gen = G.shape[1]
    if type(A) == np.ndarray:
        n_con = A.shape[0]
    else:
        n_con = 0
    n_out_max = (2 ** n) - 1

    # Create list of output zonotopes
    Z_out = []

    # CREATE THE ZONOTOPES
    c_new, D_new, H_new = hpint_perm(n)

    for i in range(n_out_max):
        # Get new center and generator matrices
        c_i = D_new[i] @ c
        G_i = D_new[i] @ G
        G_i = np.concatenate((G_i, np.zeros((n, n))), axis=1)

        # Get new constraint arrays
        HG = H_new[i] @ G
        d_i = np.absolute(HG) @ np.ones((n_gen, 1))
        Hc = H_new[i] @ c
        d_i = 0.5 * (d_i - Hc)

        b_i = -Hc - d_i
        if type(b) == np.ndarray:
            b_i = np.concatenate((b, b_i), axis=0)

        A_i = np.concatenate((HG, np.diag(np.transpose(d_i)[0])), axis=1)
        if n_con > 0:
            A_i = np.concatenate((np.concatenate((A, np.zeros((n_con, n))), axis=1), A_i), axis=0)

        # Create output zonotope
        Z_out.append(ConstrainedZonotope(c_i, G_i, A_i, b_i))

    return Z_out

def ReLU_con_zono(Z_in):
    """
    INPUT:
    Z_in: A list of constrained zonotopes of class ConstrainedZonotope from constrained_zonotope.py.

    OUTPUT:
    Z_out: A list of constrained zonotopes of class ConstrainedZonotope from constrained_zonotope.py as a result of
    ReLU-activating each element in Z_in.
    """
    # Get the number of input zonotopes
    n_in = len(Z_in)

    # Iterate through input zonotopes and generate output
    Z_out = []

    for i in range(n_in):
        Z_i = Z_in[i]
        Z_out_i = ReLU_con_zono_single(Z_i)
        for j in range(len(Z_out_i)):
            Z_out.append(Z_out_i[j])

    return Z_out


def linear_layer_con_zono(Z_in, W, b):
    """
    INPUTS:
    Z_in: A list of constrained zonotopes of class ConstrainedZonotope from constrained_zonotope.py.
    W: A single weight matrix as numpy array.
    b: A single bias vector as numpy array.

    OUTPUT:
    Z_out: A list of constrained zonotopes of class ConstrainedZonotope from constrained_zonotope.py as a result of
    multiplying and adding W and b to each element in Z_in.
    """
    # Preallocate the output
    Z_out = []

    # Get the number of input zonotopes
    n_in = len(Z_in)

    for i in range(n_in):
        # Get the current zonotope
        Z = Z_in[i]

        # Get the zonotope's parameters
        c = Z.c
        G = Z.G

        # Do the linear transformation
        c = W @ c + b
        G = W @ G

        # Update the output
        Z_out.append(ConstrainedZonotope(c, G, Z.A, Z.b))

    return Z_out


def forward_pass_NN_con_zono(Z_in, NN_weights, NN_biases):
    """
    INPUTS:
    Z_in: A single zonotope of class Zonotope from zonotope.py.
    NN_weights: A list of numpy arrays where each element is the neural network layer's weight matrix. Its length is
    the depth of the neural network.
    NN_biases: A list of numpy arrays where each element is the neural network layer's bias. Same length as NN_weights.

    OUTPUT:
    Z_out: A list of constrained zonotopes of class ConstrainedZonotope from constrained_zonotope.py as a result of
    passing Z_in through a neural network defined by NN_weights and NN_biases
    """
    # Get depth of neural network
    n_depth = len(NN_weights)

    # Convert input zonotope into a constrained zonotope
    Z_in = ConstrainedZonotope(Z_in.c, Z_in.G)

    # Run through layers and perform ReLU activations
    Z_out = [Z_in]
    for i in range(n_depth - 1):
        W = NN_weights[i]
        b = NN_biases[i]
        Z_out = linear_layer_con_zono(Z_out, W, b)
        Z_out = ReLU_con_zono(Z_out)

    # Evaluate final layer
    Z_out = linear_layer_con_zono(Z_out, NN_weights[-1], NN_biases[-1])

    return Z_out

#### PYTORCH VERSION ####

def hpint_perm_torch(n):
    """
    This function outputs lists of matrices for calculating all permutations of half-plane intersections in the
    ReLU_con_zono_single function.
    """
    c_new = []
    D_new = []
    H_new = []
    for i in range(2 ** n - 1):
        c_new_i = torch.zeros(n, 1)
        binStr = bin(i + 1)[2:]
        for j in range(len(binStr)):
            c_new_i[n - 1 - j][0] = int(binStr[len(binStr) - 1 - j])
        c_new.append(c_new_i)
        D_new_i = torch.diag(torch.transpose(c_new_i)[0])
        D_new.append(D_new_i)
        H_new_i = torch.diag(torch.transpose(c_new_i * (-2) + 1)[0])
        H_new.append(H_new_i)

    return c_new, D_new, H_new


def ReLU_con_zono_single(Z_in):
    """
    INPUT:
    Z_in: A single constrained zonotope of class TorchConstrainedZonotope from constrained_zonotope.py.

    OUTPUT:
    Z_out: A list of constrained zonotopes of class ConstrainedZonotope from constrained_zonotope.py as a result of
    ReLU-activating Z_in.
    """
    # SETUP
    # Get the zonotope parameters
    c = Z_in.c
    G = Z_in.G
    A = Z_in.A
    b = Z_in.b

    # Get dimension of things
    n = c.shape[0]
    n_gen = G.shape[1]
    if type(A) == np.ndarray:
        n_con = A.shape[0]
    else:
        n_con = 0
    n_out_max = (2 ** n) - 1

    # Create list of output zonotopes
    Z_out = []

    # CREATE THE ZONOTOPES
    c_new, D_new, H_new = hpint_perm(n)

    for i in range(n_out_max):
        # Get new center and generator matrices
        c_i = D_new[i] @ c
        G_i = D_new[i] @ G
        G_i = np.concatenate((G_i, np.zeros((n, n))), axis=1)

        # Get new constraint arrays
        HG = H_new[i] @ G
        d_i = np.absolute(HG) @ np.ones((n_gen, 1))
        Hc = H_new[i] @ c
        d_i = 0.5 * (d_i - Hc)

        b_i = -Hc - d_i
        if type(b) == np.ndarray:
            b_i = np.concatenate((b, b_i), axis=0)

        A_i = np.concatenate((HG, np.diag(np.transpose(d_i)[0])), axis=1)
        if n_con > 0:
            A_i = np.concatenate((np.concatenate((A, np.zeros((n_con, n))), axis=1), A_i), axis=0)

        # Create output zonotope
        Z_out.append(ConstrainedZonotope(c_i, G_i, A_i, b_i))

    return Z_out

def ReLU_con_zono(Z_in):
    """
    INPUT:
    Z_in: A list of constrained zonotopes of class ConstrainedZonotope from constrained_zonotope.py.

    OUTPUT:
    Z_out: A list of constrained zonotopes of class ConstrainedZonotope from constrained_zonotope.py as a result of
    ReLU-activating each element in Z_in.
    """
    # Get the number of input zonotopes
    n_in = len(Z_in)

    # Iterate through input zonotopes and generate output
    Z_out = []

    for i in range(n_in):
        Z_i = Z_in[i]
        Z_out_i = ReLU_con_zono_single(Z_i)
        for j in range(len(Z_out_i)):
            Z_out.append(Z_out_i[j])

    return Z_out


def linear_layer_con_zono(Z_in, W, b):
    """
    INPUTS:
    Z_in: A list of constrained zonotopes of class ConstrainedZonotope from constrained_zonotope.py.
    W: A single weight matrix as numpy array.
    b: A single bias vector as numpy array.

    OUTPUT:
    Z_out: A list of constrained zonotopes of class ConstrainedZonotope from constrained_zonotope.py as a result of
    multiplying and adding W and b to each element in Z_in.
    """
    # Preallocate the output
    Z_out = []

    # Get the number of input zonotopes
    n_in = len(Z_in)

    for i in range(n_in):
        # Get the current zonotope
        Z = Z_in[i]

        # Get the zonotope's parameters
        c = Z.c
        G = Z.G

        # Do the linear transformation
        c = W @ c + b
        G = W @ G

        # Update the output
        Z_out.append(ConstrainedZonotope(c, G, Z.A, Z.b))

    return Z_out


def forward_pass_NN_con_zono_torch(Z_in, NN_weights, NN_biases):
    """
    INPUTS:
    Z_in: A single zonotope of class Zonotope from zonotope.py.
    NN_weights: A list of torch tensors where each element is the neural network layer's weight matrix. Its length is
    the depth of the neural network.
    NN_biases: A list of torch tensors where each element is the neural network layer's bias. Same length as NN_weights.

    OUTPUT:
    Z_out: A list of constrained zonotopes of class ConstrainedZonotope from constrained_zonotope.py as a result of
    passing Z_in through a neural network defined by NN_weights and NN_biases
    """
    # Get depth of neural network
    n_depth = len(NN_weights)

    # Convert input zonotope into a constrained zonotope
    Z_in = ConstrainedZonotope(Z_in.c, Z_in.G)

    # Run through layers and perform ReLU activations
    Z_out = [Z_in]
    for i in range(n_depth - 1):
        W = NN_weights[i]
        b = NN_biases[i]
        Z_out = linear_layer_con_zono(Z_out, W, b)
        Z_out = ReLU_con_zono(Z_out)

    # Evaluate final layer
    Z_out = linear_layer_con_zono(Z_out, NN_weights[-1], NN_biases[-1])

    return Z_out