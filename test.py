import numpy as np
from src import simpleKuboOscillator3, kOU
import matplotlib.pyplot as plt


def returnTraj(Kubo, x0):
    X = [x0]
    for ang in Kubo.Theta:
        xnew = np.cos(ang)*X[-1][0] - np.sin(ang)*X[-1][1]
        ynew = np.sin(ang)*X[-1][0] + np.cos(ang)*X[-1][1]
        X.append([xnew, ynew])
    X = np.array(X)
    return X                

### Initalise the oscillator
kpar = np.array([1., 3.0])
Kubo = simpleKuboOscillator3(1.5, 0., kOU, kpar)


x0 = np.array([1., 0.])         # Initial condition 
evalt = np.linspace(0., 8., 100) # Observation times

### Simulate some data
#X = Kubo.sim(x0, evalt)

Data = np.loadtxt('wind1.txt')

tt = Data[:,0]

t0 = tt[0]
T = tt[-1]

Theta = Data[:,-1]
X = np.column_stack(( np.cos(Theta), np.sin(Theta) ))

fig = plt.figure()
ax = fig.add_subplot(111)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

IND = np.linspace(0., tt.size-1, 200, dtype=np.intp)
ind_ = np.array([IND[i] for i in range(IND.size-1) if i % 2 == 0 ])

#for ind, g, c in zip([ind_], [.5], ['b']):
for ind, g, c in zip([ind_, IND], [.5, .5], ['b', 'r']):

    kpar = np.array([1., g])
    Kubo = simpleKuboOscillator3(1.5, 0., kOU, kpar)

#    ind = np.linspace(0, tt.size-1, N, dtype=np.intp)

    evalt = (tt[ind] - t0)#/T
    X_ = X[ind,:]

    Kubo.fit(evalt, X_)
    a, b = Kubo.posteriorPrec_par(.01, .01, None)

    IE = np.concatenate(([0.], np.cumsum(Kubo.Theta)))

    ax.step(Kubo.evalt, IE , c + '-', alpha=0.2)
    ax.plot(Kubo.evalt, IE , '+')

    yy = returnTraj(Kubo, X_[0, ])

    ax2.plot(Kubo.evalt, Kubo.evalt, c + '+-')
#ax2.plot(Kubo.evalt, yy[:,0], c + '-')
#    ax2.plot(np.exp(0.01*Kubo.evalt)*yy[:,0], np.exp(0.01*Kubo.evalt)*yy[:,1], c + '-')
        
plt.show()

"""


### Observation model
from scipy.stats import norm, multivariate_normal



Ktraj = returnTraj(Kubo, X_[0,])

def obsModel(Y, KuboTraj, noise):
    lp = 0.
    for i in range(Y.shape[0]):
        lp += np.sum( norm.logpdf( Y[i,:],
                                   loc=KuboTraj[i,:],
                                   scale = noise ) )
    return lp

def logPfull(Y, Kubo, KuboTraj, noise, asq):
    p1 = obsModel(Y, KuboTraj, noise)
    p2 = Kubo.posterior(asq, np.zeros(Kubo.Theta.size))
    return np.log(p1) + np.log(p2)


fig2 = plt.figure()
ax = fig2.add_subplot(111)
"""

plt.show()

