import numpy as np
from scipy.integrate import quad, dblquad


c = 0.5

def integrand(y, x):
    return c*np.exp(-c*np.abs(y-x))

def integrand2(y, x):
    return c*np.exp(c*(y-x))

s0 = 0.
s1 = 1.
t0 = 1.
t1 = 2.

print(dblquad(integrand, s0, s1, lambda x: t0, lambda x: t1)[0])


def cov(s, t, s0, t0, kpar, isVar=False):
    if isVar:
        return 2*((t - t0) + (np.exp(kpar[-1]*(t0-t)) - 1.)/kpar[-1])

    else:
        ### check s < t0
        if s <= t0:
            expr1 = (np.exp(kpar[-1]*(s-t0)) - np.exp(kpar[-1]*(s0-t0)))/kpar[-1]
            expr2 = (np.exp(kpar[-1]*(s-t)) - np.exp(kpar[-1]*(s0-t)))/kpar[-1]
            return expr1 - expr2

def makeCOV_integratedOU(tt, kpar):
    N = tt.size - 1
    var = cov(tt[1:],tt[1:],
              tt[:-1],tt[:-1], kpar, True)
    C = np.diag(var)
    for i in range(N):
        for j in range(i):
            C[i, j] = cov(tt[j+1], tt[i+1], tt[j], tt[i], kpar)
            C[j, i] = C[i,j]
    print(C)


kpar = np.array([1., 0.5])
tt = np.linspace(0., 5., 100)
makeCOV_integratedOU(tt, kpar)
print("Done")
C2 = np.zeros((tt.size-1, tt.size-1))
for i in range(tt.size-1):
    for j in range(i+1):
        C2[i, j] = dblquad(integrand, tt[i], tt[i+1], lambda x: tt[j], lambda x: tt[j+1])[0]
        C2[j, i] = C2[i, j]
print("And done")
#print cov(s1, t1, s0, t0, [1., 0.5])


