import numpy as np
import matplotlib.pyplot as plt
from pylab import imread

G = 1
c = 1
Dd = 1e8
Ds = Dd + 1e3
Dds = Ds - Dd
xi0 = 4.8481368111e-6 * Dd

def calcAlphaTw(xi,masses):
    n = np.size(masses,0)
    m = np.size(masses,1)
    a = np.zeros(2)
    for i in range(n):
        for j in range(m):
            xip = np.array([i,j],dtype=float)
            rsq = np.sum(np.square(xi - xip))
            if (rsq < 0.5):
                continue
            a = a + 4.0*G*masses[i,j]*(xi - xip)/(c**2*rsq)
    return a

def calcAlpha(x,masses):
    xi = xi0 * x
    atw = calcAlphaTw(xi,masses)
    return Dd*Dds*atw/(xi0*Ds)

def jacobian(x1,dx,i,j,masses):
    x2 = np.copy(x1)
    x2[i] = x2[i] + dx
    y1 = x[j] - calcAlpha(x1,masses)[j]
    y2 = x[j] + dx - calcAlpha(x2,masses)[j]
    return (y2 - y1)/dx

def magnif(x,dx,masses):
    a = jacobian(x,dx,0,0,masses)
    b = jacobian(x,dx,0,1,masses)
    c = jacobian(x,dx,1,0,masses)
    d = jacobian(x,dx,1,1,masses)
    det = np.absolute(a*d - b*c)
    if (det > 0.0):
        mag = 1.0/det
    else:
        mag = 0
    return mag

def fuzzyIndex(y,m,n):
    dist = np.zeros([m,n])
    if (any(y < 0) or y[0] >= m or y[1] >= n):
        return dist
    y1 = [np.floor(y[0]),np.floor(y[1])]
    y2 = [np.floor(y[0]),np.ceil(y[1])]
    y3 = [np.ceil(y[0]),np.floor(y[1])]
    y4 = [np.ceil(y[0]),np.ceil(y[1])]
    #dfloor = np.sqrt(np.sum((y - y1)**2))
    #dceil = np.sqrt(np.sum((y4 - y)**2))
    #dfc = np.sqrt(dfloor**2 + dceil**2)
    dist[y1[0],y1[1]] = np.sqrt(np.sum((y - y1)**2))
    dist[y2[0],y2[1]] = np.sqrt(np.sum((y - y2)**2))
    dist[y3[0],y3[1]] = np.sqrt(np.sum((y - y3)**2))
    dist[y4[0],y4[1]] = np.sqrt(np.sum((y - y4)**2))
    return dist

source = imread('test.png')
m = np.size(source,0)
n = np.size(source,1)
image = np.zeros([m,n,3])
inds = np.mgrid[0:m,0:n]
masses = 1.0/np.sqrt((inds[0]-0.5*m)**2 + (inds[1]-0.5*n)**2)
#plt.matshow(masses)
#plt.show()
for i in range(m):
    for j in range(n):
        x = np.array([i,j])
        a = calcAlpha(x,masses)
        y = x - a
        ai = np.round_(100*a)
        #print ai
        mask = fuzzyIndex(y,m,n)
        add = np.array([source[:,:,0]*mask,source[:,:,1]*mask,source[:,:,2]*mask])
        if (any(ai < 0) or ai[0] >= m or ai[1] >= n):
            continue
        image[ai[0],ai[1],:] = image[ai[0],ai[1],:] + np.sum(np.sum(add,axis=1),axis=1)
plt.imshow(image)
plt.show()
