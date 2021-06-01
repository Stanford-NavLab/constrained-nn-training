import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.linalg import null_space
from util.zonotope import Zonotope
from matplotlib.patches import Polygon
import pypoman

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

    ### Properties
    def vertices(self):
        """
        Vertices of a constrained zonotope
        Adapted from CORA lcon2vert.m
        Tested for 2-D and conzonos with only one vertex
        TODO: Implement cases for 1-D and 3-D.
        """
        c = self.c
        G = self.G
        A = self.A
        b = self.b
        if type(A) != np.ndarray:
            return Zonotope(c, G).vertices()
        n_G = G.shape[1]
        A_ineq = np.concatenate((np.eye(n_G), -np.eye(n_G)), axis=0)
        b_ineq = np.ones((2 * n_G, 1))
        Neq = null_space(A)
        x0 = np.linalg.pinv(A) @ b
        AAA = A_ineq @ Neq
        bbb = b_ineq - A_ineq @ x0
        try:
            vertices = pypoman.compute_polytope_vertices(AAA, bbb)
        except:
            print("error plotting: ",AAA, bbb)
        if len(vertices) > 0:
            Zt = np.array([vertices[0]])
            for i in range(1, len(vertices)):
                Zt = np.concatenate((Zt, np.array([vertices[i]])), axis=0)
            V = Zt @ Neq.T + x0.T
            V = c + G @ V.T
            #print(V)

            # Rearrange points in V in CCW order
            Px = V[0]
            Py = V[1]
            cx = np.mean(Px)
            cy = np.mean(Py)
            ang = np.arctan2(Py - cy, Px - cx)
            dst = np.zeros(Px.shape)
            dtype = [('angle', float), ('distance', float), ('index', int)]
            values = []
            for i in range(len(Px)):
                dst[i] = np.linalg.norm(np.array([cx - Px[i], cy - Py[i]]))
                values.append((ang[i], dst[i], i))
            idx = np.sort(np.array(values, dtype=dtype), order=['angle', 'distance'])
            V_sorted = np.copy(V)
            for i in range(len(Px)):
                V_sorted[:, i] = V[:, idx[i][2]]
            
            return V_sorted
        return np.array([[]])


    ### Plotting
    def plot(self, ax=None, color='b', alpha=0.5):
        """ Plot function

        Args (all optional):
            ax: axes to plot on, if unspecified, will generate and plot on new set of axes
            color: color
            alpha: patch transparency (from 0 to 1)
        """
        V = self.vertices()

        if ax == None:
            fig, ax = plt.subplots()
        if V.shape[1] == 1:
            plt.plot(V[1][0], V[0][0], '.', color=color, alpha=alpha)
        elif V.shape[1] >= 1:
            poly = Polygon(V.T, True, color=color, alpha=alpha)
            ax.add_patch(poly)
        # p = PatchCollection([poly], match_original=True)
        # ax.add_collection(p)

        # recompute the ax.dataLim
        ax.relim()
        # update ax.viewLim using the new dataLim
        ax.autoscale_view()


#### --------------- PYTORCH VERSION ----------------- ####

class TorchConstrainedZonotope(object):
    """ Torch Constrained Zonotope

    Example usage:
        z = TorchwwConstrainedZonotope(torch.zeros(2,1),torch.eye(2))
    """
    __array_priority__ = 1000 # prioritize class mul over numpy array mul

    ### Constructors
    def __init__(self, center, generators, constraint_A=None, constraint_b=None):
        """ Constructor

        Args:
            center: torch tensor
            generators: torch tensor
        """
        self.c = center 
        self.G = generators
        self.dim = center.shape[0]
        self.order = generators.shape[1]
        if constraint_A is None:
            self.A = torch.zeros(0,self.order)
            self.b = torch.zeros(0,1)
        else:
            self.A = constraint_A
            self.b = constraint_b
        

    def __str__(self):
        ind = '\t'
        c_str = ind + str(self.c.data).replace('\n','\n' + ind)
        G_str = ind + str(self.G.data).replace('\n','\n' + ind)
        A_str = ind + str(self.A.data).replace('\n','\n' + ind)
        b_str = ind + str(self.b.data).replace('\n','\n' + ind)
        
        print_str = 'center:\n' + c_str + '\ngenerators:\n' + G_str + \
                    '\nconstraint A:\n' + A_str + '\nconstraint b:\n' + b_str
        return print_str

    ### Operations
    def __add__(self, other):
        """ Minkowski addition (overloads '+') """
        c = self.c + other.c
        G = torch.hstack((self.G, other.G))
        A = torch.block_diag(self.A, other.A)
        b = torch.vstack((self.b, other.b))
        return TorchConstrainedZonotope(c,G,A,b)

    def __rmul__(self, other):
        """ Right linear map (overloads '*') """
        # other is a scalar
        if np.isscalar(other):
            c = other * self.c
            G = other * self.G 
        # other is a matrix
        elif type(other) is torch.Tensor:
            c = other @ self.c
            G = other @ self.G 
        return TorchConstrainedZonotope(c,G,self.A,self.b) 
    
    def __mul__(self, other):
        """ (Left) linear map (overloads '*') """
        # other is a scalar
        if np.isscalar(other):
            c = other * self.c
            G = other * self.G 
        # other is a matrix
        elif type(other) is torch.Tensor:
            c = self.c @ other
            G = self.G @ other
        return TorchConstrainedZonotope(c,G,self.A,self.b) 

    def intersect(self, other):
        """ Intersection """
        c = self.c
        G = torch.hstack((self.G, torch.zeros(self.dim, other.order)))
        # no constraints case
        # if self.A is None and other.A is None:
        #     A = torch.hstack((self.G, -other.G))
        #     b = other.c - self.c
        # else:
        q1 = self.A.shape[0]; q2 = other.A.shape[0] 
        A1 = torch.hstack((self.A, torch.zeros(q1, other.order)))
        A2 = torch.hstack((torch.zeros(q2, self.order), other.A))
        A3 = torch.hstack((self.G, -other.G))
        A = torch.vstack((A1, A2, A3))
        b = torch.vstack((self.b, other.b, other.c - self.c))
        return TorchConstrainedZonotope(c,G,A,b) 

    def plot(self, ax=None, color='b', alpha=0.5):
        """ Plot """
        if self.A.shape[0] > 0:
            cz = ConstrainedZonotope(self.c.detach().numpy(), self.G.detach().numpy(), self.A.detach().numpy(), self.b.detach().numpy())
        else:
            cz = ConstrainedZonotope(self.c.detach().numpy(), self.G.detach().numpy())
        cz.plot(ax, color, alpha)