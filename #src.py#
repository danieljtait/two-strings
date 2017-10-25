import numpy as np
from scipy.integrate import dblquad
from scipy.stats import multivariate_normal


########
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.dot(v1_u, v2_u))
#    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_angle(x, y):
    dot = x[0]*y[0] + x[1]*y[1]
    det = x[0]*y[1] - x[1]*y[0]
    return np.arctan2(det, dot)

#    angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
#    if (angle < 0):
#        angle += 2*np.pi
#    return angle

class GaussProc:
    def __init__(self, kernel, kpar):
        self.kernel = kernel
        self.kpar = kpar

        # stored values
        self.evalt = None
        self.LCov = None
        self.mean = None
        
    def getCov(self, evalt, storeType=1):
        C = np.array([[self.kernel(s, t, self.kpar) for t in evalt] for s in evalt])
        L = np.linalg.cholesky(C)
        if storeType == 0:
            return C
        else:
            self.evalt = evalt
            self.LCov = L
    
    def sim(self, evalt=None):
        if evalt==None:
            ## Values are already stored
            z = np.random.normal(size=self.evalt.size)
            return np.dot(self.LCov, z)

class IntGaussProcess(GaussProc):
    def __init__(self, kernel, kpar):
        GaussProc.__init__(self, kernel, kpar)

    def IKern(self, t, s):
        return self.kernel(s, t, self.kpar)

    # for the vector t = [t_0,t_1,...,t_n] returns the n x n
    # cov of the integrated GP
    def getCov(self, evalt, storeType=1):
        C = np.zeros((evalt.size - 1, evalt.size - 1))
        for i in range(evalt.size - 1):
            for j in range(evalt.size - 1):
                C[i,j] = dblquad(self.IKern,
                                 evalt[i], evalt[i+1],
                                 lambda x: evalt[j], lambda x: evalt[j+1])[0]
                C[i,j] = C[i,j]
        L = np.linalg.cholesky(C)
        if storeType == 0:
            return C
        else:
            self.t0 = evalt[0]
            self.evalt = evalt[1:]
            self.LCov = L

class MHfitKuboOscillator:
    def __init__(self, Data, KuboOscillator):
        self.Kubo = KuboOscillator
        self.Cov0 = KuboOscillator.IntGaussProc.getCov(Data[:,0], 0)

        ## map the X data on to the sphere
        X = Data[:,1:]
        Theta = []
        for i in range(X.shape[0]-1):
            Theta.append( get_angle( X[i,], X[i+1,] ) )
        self.Theta = np.array(Theta)

    def posterior(self, a, ki):
        theta_ = self.Theta + ki
        return multivariate_normal.pdf(theta_,mean=np.zeros(self.Theta.size),cov=a*self.Cov0)
        

    ##
    # Initalise the prior and proposal distribution for the parameter
    # a
    def MH_setup_a(self, aPrior, aProposal):
        self.aPrior = aPrior
        self.aProposal = aProposal


    def MHstep_a(self, aPrior, aProposal):
        anew = aPrior.rvs(self.acur)
                
####
#         
class simpleKuboOscillator2:
    def __init__(self, a, b, kernel, kpar):
        self.a = a
        self.b = b
        self.basePar = [1., kpar[-1]]
        self.IntGaussProc = IntGaussProcess(kernel, self.basePar)

    def fit(self, evalt, X):
        self.IntGaussProc.getCov(evalt) # Store the covariance function

        ## map the X data on to the sphere
        Theta = []
        for i in range(X.shape[0]-1):
            Theta.append( get_angle( X[i,], X[i+1,] ) )
        self.Theta = np.array(Theta)

    def sim(self, x0, evalt):
        C = self.IntGaussProc.getCov(evalt, 0)
        L = np.linalg.cholesky(C)
        Igp = self.a*np.dot(L, np.random.normal(size=L.shape[0]))
        X = [x0]
        for i in range(Igp.size):
            xnew = np.cos(Igp[i])*X[-1][0] - np.sin(Igp[i])*X[-1][1]
            ynew = np.sin(Igp[i])*X[-1][0] + np.cos(Igp[i])*X[-1][1]
            X.append([xnew, ynew])
        X = np.array(X)
        return X

    def posterior(self, asq, ki):
        theta_ = self.Theta + ki
        Cov0 = np.dot(self.IntGaussProc.LCov, self.IntGaussProc.LCov)
        return multivariate_normal.pdf(theta_, mean=np.zeros(self.Theta.size), cov=asq*Cov0)

class simpleKuboOscillator3:
    def __init__(self, a, b, kernel, kpar):
        self.a = a
        self.b = b
        self.basePar = [1., kpar[-1]]

    def fit(self, evalt, X):
        self.evalt = evalt
        self.C0 = makeCov_IkOU(evalt, self.basePar) # Store the covariance function

        Theta = []
        for i in range(X.shape[0]-1):
            Theta.append( get_angle( X[i, ], X[i+1, ] ) )
        self.Theta = np.array(Theta)

    def sim(self, x0, evalt):
        C0 = makeCov_IkOU(evalt, self.basePar)
        L = self.a*np.linalg.cholesky(C0)
        Igp = np.dot(L, np.random.normal(size=evalt.size-1))
        X = [x0]
        for i in range(Igp.size):
            xnew = np.cos(Igp[i])*X[-1][0] - np.sin(Igp[i])*X[-1][1]
            ynew = np.sin(Igp[i])*X[-1][0] + np.cos(Igp[i])*X[-1][1]
            X.append([xnew, ynew])
        X = np.array(X)
        return X        

    def posterior(self, asq, ki):
        theta_ = self.Theta + ki
        return multivariate_normal.pdf(theta_, mean=np.zeros(self.Theta.size), cov=asq*self.C0)

    def posteriorPrec_par(self, a0, b0, ki):
        aPost = a0 + 0.5*self.Theta.size
        Q = np.dot(self.Theta, np.dot(np.linalg.inv(self.C0), self.Theta))
        bPost = b0 + 0.5*Q
        return aPost, bPost
        

class simpleKuboOscillator:
    def __init__(self, a, b, IntGaussProc):
        self.a = a
        self.b = b
        self.IntGaussProc = IntGaussProc

    def fit(self, evalt):
        self.IntGaussProc.getCov(evalt)

    def sim(self, x0, evalt):
        self.IntGaussProc.getCov(evalt, 1)
        Igp = self.IntGaussProc.sim()
        X = [x0]
        for i in range(Igp.size):
            xnew = np.cos(Igp[i])*X[-1][0] - np.sin(Igp[i])*X[-1][1]
            ynew = np.sin(Igp[i])*X[-1][0] + np.cos(Igp[i])*X[-1][1]
            X.append([xnew, ynew])
        X = np.array(X)
        return X

    def updateVarPar(self, anew):
        self.IntGaussProc.LCov *= np.sqrt(anew)

######
# Predefined kernel functions
def kOU(s, t, par):
    return par[1]*par[0]*np.exp(-par[1]*np.abs(s-t))

def IkOU(s, t, s0, t0, kpar, isVar=False):
    if isVar:
        return 2*((t - t0) + (np.exp(kpar[-1]*(t0-t)) - 1.)/kpar[-1])
    else:
        ### check s < t0
        if s <= t0:
            expr1 = (np.exp(kpar[-1]*(s-t0)) - np.exp(kpar[-1]*(s0-t0)))/kpar[-1]
            expr2 = (np.exp(kpar[-1]*(s-t)) - np.exp(kpar[-1]*(s0-t)))/kpar[-1]
            return expr1 - expr2

def makeCov_IkOU(tt, kpar):
    N = tt.size - 1
    var = IkOU(tt[1:],tt[1:],
               tt[:-1],tt[:-1], kpar, True)
    C = np.diag(var)
    for i in range(N):
        for j in range(i):
            C[i, j] = IkOU(tt[j+1], tt[i+1], tt[j], tt[i], kpar)
            C[j, i] = C[i,j]
    return C

"""
kpar = [1., 1.]
evalt = np.linspace(0., 10., 50)

GP = GaussProc(kOU, kpar)
GP.getCov(evalt, 1)

IGP = IntGaussProcess(kOU, kpar)
Kubo = simpleKuboOscillator(None,None,IGP)
x0 = np.array([1., 0.])


IGP2 = IntGaussProcess(kOU, [0.5, 1.0])

IGP.getCov(evalt, 1)
IGP2.getCov(evalt, 1)


Kubo2 = simpleKuboOscillator2(5., None, kOU, kpar)


X1 = Kubo2.sim(x0, evalt)
Kubo2.a = 1.5
X2 = Kubo2.sim(x0, evalt)

Kubo2.fit(evalt, X2)


print("Metropolis Hastings Initalised")
Data = np.column_stack((evalt, X2))
MHkubo = MHfitKuboOscillator(Data, Kubo2)
print("")
kk = np.zeros(MHkubo.Theta.size)

from scipy.stats import expon, gamma
from scipy.integrate import quad

Kubo3 = simpleKuboOscillator3(1.5, None, None, kpar)
Kubo3.fit(evalt, X2)

######
# conditional on k, b the posterior is just a 
a0 = 1.
b0 = 1.
gammaPrior = gamma(a0, scale=1./b0)

def integrand(l):
    return Kubo3.posterior(1./l, kk)*gammaPrior.pdf(l)
#    return MHkubo.posterior(1./l, kk)*gammaPrior.pdf(l)

aPost = a0 + 0.5*MHkubo.Theta.size
S0 = MHkubo.Cov0
S0inv = np.linalg.inv(S0)
Q = np.dot(MHkubo.Theta, np.dot(np.linalg.inv(S0), MHkubo.Theta))
bPost = b0 + 0.5*Q
C = quad(integrand, 0., np.inf)[0]

gammaPosterior = gamma(a=aPost, scale=1./bPost)
m, v, s, k = gammaPosterior.stats(moments='mvsk')
print(m, (1./1.5)**2 )
#######


import matplotlib.pyplot as plt

ll = np.linspace(0.1, 1., 50)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ll, [integrand(l)/C for l in ll])
ax.plot(ll, gamma.pdf(ll, a=aPost, scale=1./bPost), 'r+')


fval = gamma.pdf(ll, a=aPost, scale=1./bPost)
nSim = 100
for nt in range(10):
    kk = np.random.randint(low=-4, high=4, size=MHkubo.Theta.size)
    vec = MHkubo.Theta + 2*np.pi*kk
    aPost = a0 + 0.5*MHkubo.Theta.size
    Q = np.dot(vec, np.dot(S0inv, vec))
    bPost = b0 + 0.5*Q
    fval += gamma.pdf(ll, a=aPost, scale=1./bPost)
ax.plot(ll, fval, 'k-.')


fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.plot(evalt[1:], MHkubo.Theta, 'o')

plt.show()




"""
