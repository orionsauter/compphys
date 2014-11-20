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
    m = np.size(masses,0)
    n = np.size(masses,1)
    a = np.zeros(2)
    xi2 = np.array([xi[0]*np.ones([m,n]),xi[1]*np.ones([m,n])],dtype=float)
    xip = xi0 * np.array(np.mgrid[0:m,0:n],dtype=float)
    rsq = np.square(xi2[0,:,:] - xip[0,:,:]) + np.square(xi2[1,:,:] - xip[1,:,:])
    rsq[rsq < 2.0] = 2.0
    masses2 = np.array([masses, masses])
    rsq2 = np.array([rsq,rsq])
    a = np.sum(np.sum(4.0*G*masses2*(xi2 - xip)/(c**2*rsq2),axis=1),axis=1)
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
    dist = dist/np.sum(dist)
    return dist

#source = imread('test.png')
source = imread('orion.png')
plt.imshow(source)
m = np.size(source,0)
n = np.size(source,1)
image = np.zeros([m,n,3])
inds = np.array(np.mgrid[0:m,0:n],dtype=float)
r = np.sqrt((inds[0]-0.5*float(m-1))**2 + (inds[1]-0.5*float(n-1))**2) + 1
masses = 100.0/r
plt.matshow(masses)
for i in range(m):
    print str(i*n)+'/'+str(m*n)
    for j in range(n):
        x = np.array([i,j])
        a = calcAlpha(x,masses)
        y = x - a
        mask = fuzzyIndex(y,m,n)
        add = np.array([source[:,:,0]*mask,source[:,:,1]*mask,source[:,:,2]*mask])
        image[i,j,:] = image[i,j,:] + np.sum(np.sum(add,axis=1),axis=1)
plt.figure()
plt.imshow(image)
plt.show()
