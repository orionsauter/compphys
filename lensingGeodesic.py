import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

G = 1.0
pi = np.pi
rho = 0.1

def diffEq(mpf,r):
    dfdr = G * (mpf[0] + 4.0*pi*r**3*mpf[1])/(r*(r-2*G*mpf[0]))
    dpdr = -G * (rho + mpf[1]) * dfdr
    dmdr = 4.0*pi*r**2*rho
    return np.array([dmdr,dpdr,dfdr])

def getMetric(r,rStar,mpf):
    gdd = np.zeros([4,4])
    chrudd = np.zeros([4,4,4])
    imax = len(rStar)-1
    ri = 0
    for i in range(imax):
        if (rStar[i] < r[1]):
            ri = i
    k = (1.0 - 2.0*G*mpf[ri,0]/r[1])
    
    gdd[0,0] = -k
    gdd[1,1] = 1.0/k
    gdd[2,2] = r[1]**2
    gdd[3,3] = (r[1]*np.sin(r[2]))**2

    chrudd[0,0,3] = chrudd[0,3,0] = G*mpf[ri,0]/(k*r[1]**2)
    chrudd[1,0,0] = k*G*mpf[ri,0]/r[1]**2
    chrudd[1,1,1] = -chrudd[0,0,3]
    chrudd[1,2,2] = -k*r[1]
    chrudd[1,3,3] = chrudd[1,2,2]*np.sin(r[2])**2
    chrudd[2,1,2] = chrudd[2,2,1] = 1.0/r[1]
    chrudd[2,3,3] = -np.sin(r[2])*np.cos(r[2])
    chrudd[3,1,3] = chrudd[3,3,1] = chrudd[2,1,2]
    chrudd[3,2,3] = chrudd[3,3,2] = 1.0/np.tan(r[1])
    return [gdd,chrudd]

def geodesicEq(rrd,t,rStar,mpf):
    r = rrd[0:4]
    rdot = rrd[4:8]
    m = getMetric(r,rStar,mpf)
    gdd = m[0]
    chrudd = m[1]
    rddot = np.zeros(4)
    for i in range(4):
        for j in range(4):
            rddot = rddot - chrudd[:,i,j] * rdot[i] * rdot[j]
    return np.concatenate((rdot,rddot))
    
    
mpf0 = np.array([0.0,0.1,0.0])
rStar = np.arange(0.1,2.0,0.01)
out = odeint(diffEq,mpf0,rStar)
out = np.array(out)
neg = next(i for i in range(np.size(out,0)) if out[i,1] < 0)
mpf = out[0:(neg-1),:]
rStar = rStar[0:(neg-1)]
rrd = np.array([0.0,10.0,pi*0.1,0.0,0.0,-0.5,0.01,0.0])
out = odeint(geodesicEq,rrd,np.arange(0,50.0,0.1),args=(rStar,mpf))
ax = plt.subplot(111,polar=True)
ax.plot(out[:,2],out[:,1])
plt.show()
