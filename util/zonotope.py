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
        self.Z = np.hstack((center, generators))
        self.dim = center.shape[0]
        self.order = generators.shape[1]

    ### Printing
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

    ### Properties
    def vertices(self):
        """ Vertices of zonotope 
        
            Adapted from CORA \@zonotope\vertices.m and \@zonotope\polygon.m
            Tested on 2D zonotopes
        """
        # extract variables
        c = self.c
        G = self.G
        n = self.dim

        if n == 1:
            # compute the two vertices for 1-dimensional case
            temp = np.sum(np.abs(self.G))
            V = np.array([self.c - temp, self.c + temp])
        elif n == 2:
            # obtain size of enclosing intervalhull of first two dimensions
            xmax = np.sum(np.abs(G[0,:]))
            ymax = np.sum(np.abs(G[1,:]))

            # Z with normalized direction: all generators pointing "up"
            Gnorm = G
            Gnorm[:,G[1,:]<0] = Gnorm[:,G[1,:]<0] * -1

            # compute angles
            angles = np.arctan2(G[1,:],G[0,:])
            angles[angles<0] = angles[angles<0] + 2 * np.pi

            # sort all generators by their angle
            IX = np.argsort(angles)

            # cumsum the generators in order of angle
            V = np.zeros((2,n+1))
            for i in range(n):
                V[:,i+1] = V[:,i] + 2 * Gnorm[:,IX[i]] 

            V[0,:] = V[0,:] + xmax - np.max(V[0,:])
            V[1,:] = V[1,:] - ymax 

            # flip/mirror upper half to get lower half of zonotope (point symmetry)
            V = np.block([[V[0,:], V[0,-1] + V[0,0] - V[0,1:-1]],
                          [V[1,:], V[1,-1] + V[1,0] - V[1,1:-1]]])
            
            # consider center
            V[0,:] = c[0] + V[0,:]
            V[1,:] = c[1] + V[1,:]
        else:
            #TODO: delete aligned and all-zero generators

            # check if zonotope is full-dimensional
            if self.order < n:
                #TODO: verticesIterateSVG
                print("Vertices fro non full-dimensional zonotope not implemented yet - returning empty array")
                V = np.empty()
                return V
            
            # generate vertices for a unit parallelotope
            vert = np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).reshape(3,-1)
            V = c + G[:,:n] @ vert 
            
            #TODO: rest unimplemented

        return V
            

    ### Plotting
    def plot(self):
        pass