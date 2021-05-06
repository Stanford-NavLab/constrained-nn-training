import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

class ConstrainedZonotope(object):
    """ Constrained Zonotope

    Example usage:
        z = ConstrainedZonotope(np.zeros((2,1)),np.eye(2))
    """
    __array_priority__ = 1000 # prioritize class mul over numpy array mul

    ### Constructors
    def __init__(self, center, generators, constraint_A=None, constraint_b=None):
        """ Constructor

        Args:
            center: np array
            generators: np array
        """
        self.c = center 
        self.G = generators
        self.A = constraint_A
        self.b = constraint_b
        self.dim = center.shape[0]
        self.order = generators.shape[1]

    def __str__(self):
        np.set_printoptions(precision=3)
        ind = '\t'
        c_str = ind + str(self.c).replace('\n','\n' + ind)
        G_str = ind + str(self.G).replace('\n','\n' + ind)
        if self.A is not None:
            A_str = ind + str(self.A).replace('\n','\n' + ind)
        else:
            A_str = ind + str(self.A)
        if self.b is not None:
            b_str = ind + str(self.b).replace('\n','\n' + ind)
        else:
            b_str = ind + str(self.b)
        print_str = 'center:\n' + c_str + '\ngenerators:\n' + G_str + \
                    '\nconstraint A:\n' + A_str + '\nconstraint b:\n' + b_str
        return print_str
        # return "center:\n {0} \ngenerators:\n {1} \nconstraint A:\n {2} \
        #     \nconstraint b:\n {3}".format(self.c, self.G, self.A, self.b)

    ### Operations
    def __add__(self, other):
        """ Minkowski addition (overloads '+') """
        c = self.c + other.c
        G = np.hstack((self.G, other.G))
        A = block_diag(self.A, other.A)
        b = np.vstack((self.b, other.b))
        return ConstrainedZonotope(c,G,A,b)

    def __rmul__(self, other):
        """ Right linear map (overloads '*') """
        # other is a scalar
        if np.isscalar(other):
            c = other * self.c
            G = other * self.G 
        # other is a matrix
        elif type(other) is np.ndarray:
            c = other @ self.c
            G = other @ self.G 
        return ConstrainedZonotope(c,G,self.A,self.b) 
    
    def __mul__(self, other):
        """ (Left) linear map (overloads '*') """
        # other is a scalar
        if np.isscalar(other):
            c = other * self.c
            G = other * self.G 
        # other is a matrix
        elif type(other) is np.ndarray:
            c = self.c @ other
            G = self.G @ other
        return ConstrainedZonotope(c,G,self.A,self.b) 

    def intersect(self, other):
        """ Intersection """
        c = self.c
        G = np.hstack((self.G, np.zeros((self.dim, other.order))))
        # no constraints case
        if self.A is None and other.A is None:
            A = np.hstack((self.G, -other.G))
            b = other.c - self.c
        else:
            q1 = self.A.shape[0]; q2 = other.A.shape[0] 
            A = np.block([[self.A, np.zeros((q1, other.order))],
                          [np.zeros((q2, self.order)), other.A],
                          [self.G, -other.G]])
            b = np.vstack((self.b, other.b, other.c - self.c))
        return ConstrainedZonotope(c,G,A,b) 
 
    ### Representations
    def hrep(self):
        pass

    def vrep(self):
        pass

    ### Plotting
    def plot(self):
        pass