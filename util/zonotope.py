import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

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
            Tested on 2D zonotopes (n==2)
        """
        # extract variables
        c = self.c
        G = self.G
        n = self.dim
        m = self.order

        if n == 1:
            # compute the two vertices for 1-dimensional case
            temp = np.sum(np.abs(self.G))
            V = np.array([self.c - temp, self.c + temp])
        elif n == 2:
            # obtain size of enclosing intervalhull of first two dimensions
            xmax = np.sum(np.abs(G[0,:]))
            ymax = np.sum(np.abs(G[1,:]))
            #print('xmax: ', xmax, 'ymax: ', ymax)

            # Z with normalized direction: all generators pointing "up"
            Gnorm = G
            Gnorm[:,G[1,:]<0] = Gnorm[:,G[1,:]<0] * -1
            #print('Gnorm:\n', Gnorm)

            # compute angles
            angles = np.arctan2(G[1,:],G[0,:])
            angles[angles<0] = angles[angles<0] + 2 * np.pi
            #print('angles: ', angles)

            # sort all generators by their angle
            IX = np.argsort(angles)
            #print('IX: ', IX)

            # cumsum the generators in order of angle
            V = np.zeros((2,m+1))
            for i in range(m):
                V[:,i+1] = V[:,i] + 2 * Gnorm[:,IX[i]] 
            #print('V step 1:\n', V)

            V[0,:] = V[0,:] + xmax - np.max(V[0,:])
            V[1,:] = V[1,:] - ymax 
            #print('V step 2:\n', V)

            # flip/mirror upper half to get lower half of zonotope (point symmetry)
            V = np.block([[V[0,:], V[0,-1] + V[0,0] - V[0,1:]],
                          [V[1,:], V[1,-1] + V[1,0] - V[1,1:]]])
            #print('V step 3:\n', V)

            # consider center
            V[0,:] = c[0] + V[0,:]
            V[1,:] = c[1] + V[1,:]
            #print('V step 4:\n', V)
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
    def plot(self, ax=None, color='b', alpha=0.5):
        """ Plot function 
        
        Args (all optional):
            ax: axes to plot on, if unspecified, will generate and plot on new set of axes
            color: color 
            alpha: patch transparency (from 0 to 1)
        """
        V = self.vertices()
        xmin = np.min(V[0,:]); xmax = np.max(V[0,:])
        ymin = np.min(V[1,:]); ymax = np.max(V[1,:])

        if ax == None:
            fig, ax = plt.subplots()
        poly = Polygon(V.T, True, color=color, alpha=alpha)
        ax.add_patch(poly)
        # p = PatchCollection([poly], match_original=True)
        # ax.add_collection(p)

        # recompute the ax.dataLim
        ax.relim()
        # update ax.viewLim using the new dataLim
        ax.autoscale_view()

        # if ax == None:
        #     ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
