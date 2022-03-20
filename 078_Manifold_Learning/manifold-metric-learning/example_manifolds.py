# import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# helpful source for explicit geometric shape mappings: https://github.com/index-0/Geometric-Shapes

#-------------------------------
#---------- Helicoid  ----------
#-------------------------------

def helicoid_map(X):
    """ Map (x1, x2) to (x1*cos(x2), x1*sin(x2), x2) """
    x1 = X[0]
    x2 = X[1]
    helicoid_pt = np.array([x1*np.cos(x2), x1*np.sin(x2), x2])
    return helicoid_pt

def plot_helicoid(angle=360, r=1, c=1, current_ax=None, style='surface', title='$Helicoid$'):
    """ Apply helicoid mapping to sample grid and plot figure """
    if current_ax == None:
        fig = plt.figure(figsize=plt.figaspect(1))
        ax = fig.gca(projection='3d')
    else:
        ax = current_ax
    n = angle / 360

    u = np.linspace(0, r, endpoint=True, num=11 * n)
    v = np.linspace(-np.deg2rad(angle), np.deg2rad(angle), endpoint=True, num=22 * n)
    u, v = np.meshgrid(u, v)
    
    x = u * np.cos(v)
    y = u * np.sin(v)
    z = c * v
    
    if style == 'wireframe':
        ax.plot_wireframe(x, y, z, rstride=2, cstride=2, linewidth=1, edgecolor='black')
    else:
        ax.plot_surface(x, y, z, cmap=plt.cm.Spectral)
    ax.set_title(title);
    
def integrand_helicoid(t, x, y, Q=np.eye(2)):
    """ Functional form of arc length integal for distance estimation """
    Pth = (1 - t) * x + (y * t)
    Dff = y-x   # 2 x 1

    r = Pth[0]
    s = Pth[1]
    D = [[np.cos(s), -r * np.sin(s)],
        [np.sin(s), r * np.cos(s)],
        [0, 1]]

    v = np.matmul(D, np.matmul(Q, Dff))
    return np.linalg.norm(v) # (D*Q*(y-x))^T (D*Q*(y-x))

#---------------------------
#---------- Torus ----------
#---------------------------
    
def torus_map(X, R=5, r=1):
    """ Map (u, v) to corresponding points on the torus with major axis = R, minor axis = r """
    u = X[0]
    v = X[1]
    x = np.cos(v) * (R + r * np.cos(u))
    y = np.sin(v) * (R + r * np.cos(u))
    z = r * np.sin(u)
    torus_pt = np.array([x, y, z])
    return torus_pt

def plot_torus(R=5, r=1, current_ax='None', title='$Torus$'):
    """ Apply torus mapping to sample grid and plot figure """
    if current_ax == 'None':
        fig = plt.figure(figsize=plt.figaspect(1))
        ax = fig.gca(projection='3d')
    else:
        ax = current_ax

    u = np.linspace(0, 2 * np.pi, endpoint=True, num=30)
    v = np.linspace(0, 2 * np.pi, endpoint=True, num=30)
    u, v = np.meshgrid(u, v)

    x = np.cos(v) * (R + r * np.cos(u))
    y = np.sin(v) * (R + r * np.cos(u))
    z = r * np.sin(u)

    ax.plot_surface(x, y, z, cmap=plt.cm.Spectral)
    ax.set_title(title)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    lim = (max(abs(max(np.max(x), np.max(y), np.max(z))), abs(min(np.min(x), np.min(y), np.min(z)))))

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim);

#---------------------------------
#---------- Hyperboloid ----------
#---------------------------------

def hyperboloid_map(X):
    """ Map (x1, x2) to (x1, x2, sqrt(x1**2 + x2**2 + 1))"""
    x1 = X[0]
    x2 = X[1]
    z = np.sqrt(x1**2 + x2**2 + 1)
    hyperboloid_pt = np.array([x1, x2, z])
    return hyperboloid_pt

def hyperboloid_dist(u, v, eps=1e-5):
    """ Explicit, closed form distance between points on the hyperboloid manifold """
    inner_prod = np.dot(u[:-1], v[:-1]) - u[-1]*v[-1]
    dist = np.arccosh(-1*inner_prod)
    if np.isnan(dist):
        return eps
    else:
        return dist

def plot_hyperboloid(current_ax=None, style='surface', title='$Hyperboloid$'):
    """ Apply hyperboloid mapping to sample grid and plot figure """
    if current_ax == None:
        fig = plt.figure(figsize=plt.figaspect(1))
        ax = fig.gca(projection='3d')
    else:
        ax = current_ax
    X = np.arange(-2, 2, 0.2)
    Y = np.arange(-2, 2, 0.2)
    X, Y = np.meshgrid(X, Y)
    Z = np.sqrt(X**2 + Y**2 + 1)
    if style == 'wireframe':
        ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, linewidth=1, edgecolor='black')
    else:
        zcolors = Z - min(Z.flat)
        zcolors = zcolors/max(zcolors.flat)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.Spectral(zcolors), linewidth=1)        
    ax.set_title(title, size=20)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$');
    
def integrand_hyperboloid(t, x, y):
    """ Functional form of arc length integal for distance estimation """
    Pth = (1 - t) * x + (y * t)
    K = 1 / np.sqrt(1 + np.dot(Pth, Pth)) * Pth
    
    # matrix derivative of F
    D = [[1, 0],
         [0, 1],
         [K[0], K[1]]]
    
    # metric signature of hyperboloid (Lorentz space)
    G = [[1, 0, 0],
         [0, 1, 0],
         [0, 0, -1]]
    
    k_dot = np.array([-x + y]).T
    
    Dk = np.matmul(D, k_dot)

    v = np.sqrt(np.matmul(Dk.T, np.matmul(G, Dk)))
    return v