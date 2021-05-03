import numpy as np
import matplotlib.pyplot as plt

class Zonotope(object):
    """ Zonotope

    Example usage:
        z = Zonotope(np.zeros((2,1)),np.eye(2))
    """
    __array_priority__ = 1000 # prioritize class mul over numpy array mul

    ### Constructors
    def __init__(self, center, generators):
        """ Constructor

        Args:
            center: np array
            generators: np array
        """
        self.c = center 
        self.G = generators
        self.dim = center.shape[0]
        self.order = generators.shape[1]

    def __str__(self):
        return "center:\n {0} \n generators:\n {1}".format(self.c, self.G)

    ### Operations
    def __add__(self, other):
        """ Minkowski addition (overloads '+') """
        c = self.c + other.c
        G = np.hstack((self.G, other.G))
        return Zonotope(c,G)

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
        return Zonotope(c,G) 
    
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
        return Zonotope(c,G) 

    ### Representations
    def hrep(self):
        pass

    def vrep(self):
        pass

    ### Plotting
    def plot(self):
        pass