import numpy as np
import matplotlib.pyplot as plt
from pylab import imread
from time import time

G = 1
c = 1
Dd = 1e8
Dds = 1e3
Ds = Dd + Dds
xi0 = 4.8481368111e-6 * Dd

# Newbury Eqn 3.1
def calcAlphaTw(xi,xip,masses):
    m = np.size(masses,0)
    n = np.size(masses,1)
    a = np.zeros(2)
    xi2 = np.array([xi[0]*np.ones([m,n]),xi[1]*np.ones([m,n])],dtype=float)
    rsq = np.square(xi2[0,:,:] - xip[0,:,:]) + np.square(xi2[1,:,:] - xip[1,:,:])
    rsq[rsq < 2.0] = 2.0
    masses2 = np.array([masses, masses])
    rsq2 = np.array([rsq,rsq])
    a = np.sum(np.sum(4.0*G*masses2*(xi2 - xip)/(c**2*rsq2),axis=1),axis=1)
    return a

# Newbury Eqn 3.3
def calcAlpha(x,xip,masses):
    xi = xi0 * x
    atw = calcAlphaTw(xi,xip,masses)
    return Dd*Dds*atw/(xi0*Ds)

def fuzzyIndex(y,dist,m,n):
    dist = np.zeros([m,n],dtype=float)
    if (any(np.floor(y) < 0) or np.ceil(y[0]) >= m or np.ceil(y[1]) >= n):
        return dist
    y1 = [np.floor(y[0]),np.floor(y[1])]
    y2 = [np.floor(y[0]),np.ceil(y[1])]
    y3 = [np.ceil(y[0]),np.floor(y[1])]
    y4 = [np.ceil(y[0]),np.ceil(y[1])]
    
    dist[y1[0],y1[1]] = np.sqrt(np.sum((y - y1)**2))
    dist[y2[0],y2[1]] = np.sqrt(np.sum((y - y2)**2))
    dist[y3[0],y3[1]] = np.sqrt(np.sum((y - y3)**2))
    dist[y4[0],y4[1]] = np.sqrt(np.sum((y - y4)**2))
    norm = np.sum(dist)
    if (norm > 0):
        dist = dist/norm
    return dist

# Load source image
source = imread('tv.png')
plt.imshow(source)
m = np.size(source,0)
n = np.size(source,1)
image = np.zeros([m,n,3])

# Setup mass distribution
inds = np.array(np.mgrid[0:m,0:n],dtype=float)
rsq = (inds[0]-0.5*float(m-1))**2 + (inds[1]-0.5*float(n-1))**2
rsq[rsq < 0.5] = 0.5
r = np.sqrt(rsq)
masses = 100.0/r
plt.matshow(masses)

# Preallocate arrays
a = np.zeros([2,m,n])
xip = xi0 * np.array(np.mgrid[0:m,0:n],dtype=float)
mask = np.zeros([m,n],dtype=float)

# Main processing loop
for i in range(m):
    print str(i*n)+'/'+str(m*n)
    for j in range(n):
        x = np.array([i,j])
        a[:,i,j] = calcAlpha(x,xip,masses)
        y = x - a[:,i,j]
        # Average nearby source pixels
        mask = fuzzyIndex(y,mask,m,n)
        image[i,j,0] = image[i,j,0] + np.sum(source[:,:,0]*mask)
        image[i,j,1] = image[i,j,1] + np.sum(source[:,:,1]*mask)
        image[i,j,2] = image[i,j,2] + np.sum(source[:,:,2]*mask)
plt.figure()
plt.imshow(image)
plt.show()
