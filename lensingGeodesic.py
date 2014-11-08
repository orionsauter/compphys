import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

G = 1.0
pi = np.pi
rho = 0.1

def diffEq(mp,r):
    dpdr = -G**2 * (rho + mp[1]) * (mp[0] + 4.0*pi*r**3*mp[1])/(r*(r-2*G*mp[0]))
    dmdr = 4.0*pi*r**2*rho
    return np.array([dmdr,dpdr])
    

mp0 = np.array([0.0,0.1])
r = np.arange(1.0,2.0,0.01)
out = odeint(diffEq,mp0,r)
out = np.array(out)
neg = next(i for i in range(np.size(out,0)) if out[i,1] < 0)
out = out[0:(neg-1),:]
r = r[0:(neg-1)]
plt.plot(r,out[:,1])
plt.show()
