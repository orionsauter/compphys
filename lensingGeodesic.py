import numpy as np
import matplotlib.pyplot as plt

G = 1.0

def calcPotential(masses):
    m = np.size(masses,0)
    n = np.size(masses,1)
    inds = np.mgrid[0:m,0:n]
    phi = np.zeros([m,n])
    for i in range(m):
        for j in range(n):
            g = -G*masses[i,j]
            r = np.sqrt((inds[0]-i)**2+(inds[1]-j)**2)
            r[i,j] = 1
            dphi = g/r
            dphi[i,j] = 0
            phi = phi + dphi
    return phi

def getMetric(phi):
    m = np.size(phi,0)
    n = np.size(phi,1)
    gddDiag = np.zeros([m,n,4])
    gddDiag[:,:,0] = -1 - 2.0*phi[:,:]
    gddDiag[:,:,1] = 1 - 2.0*phi[:,:]
    gddDiag[:,:,2] = gddDiag[:,:,1]
    gddDiag[:,:,3] = gddDiag[:,:,1]
    return gddDiag

def getChristof(gddDiagIn,x):
    gddDiag = np.zeros_like(gddDiagIn)
    if (x[2] > 0):
        gddDiag[:,:,0] = ( 1.0 + gddDiagIn[:,:,0])/x[2] - 1.0
        gddDiag[:,:,1] = (-1.0 + gddDiagIn[:,:,1])/x[2] + 1.0
        gddDiag[:,:,2] = (-1.0 + gddDiagIn[:,:,2])/x[2] + 1.0
        gddDiag[:,:,3] = (-1.0 + gddDiagIn[:,:,3])/x[2] + 1.0
    else:
        gddDiag[:] = gddDiagIn
    christof = np.zeros([4,4,4])
    m = np.size(gddDiag,0)
    n = np.size(gddDiag,1)
    dg = np.zeros([m,n,4,4],dtype=float)
    temp = np.gradient(gddDiag[:,:,0])
    dg[:,:,0,1] = temp[0]
    dg[:,:,0,2] = temp[1]
    temp = np.gradient(gddDiag[:,:,1])
    dg[:,:,1,1] = temp[0]
    dg[:,:,1,2] = temp[1]
    temp = np.gradient(gddDiag[:,:,2])
    dg[:,:,2,1] = temp[0]
    dg[:,:,2,2] = temp[1]
    temp = np.gradient(gddDiag[:,:,3])
    dg[:,:,3,1] = temp[0]
    dg[:,:,3,2] = temp[1]
    #dg[:,:,:,3] = (-1.0 + gddDiag[:,:,:])/x[2]
    for i in range(4):
        for j in range(4):
            christof[i,j,i] = 0.5 * (1.0/gddDiag[x[0],x[1],i]) * dg[x[0],x[1],i,j]
            christof[i,i,j] = christof[i,j,i]
            if (i != j):
                christof[i,j,j] = -christof[i,j,i]
    return christof

m = 9
n = 9
inds = np.mgrid[0:m,0:n]
masses = 1.0/np.sqrt((inds[0]-0.5*m)**2 + (inds[1]-0.5*n)**2)
phi = calcPotential(masses)
gddDiag = getMetric(phi)
christof = getChristof(gddDiag,[0,3,1])
print christof

#plt.matshow(phi)
#plt.show()
