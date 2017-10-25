import numpy as np

I = np.diag(np.ones(2))
A = np.array([[0., 1.], [-1., 0.]])

def Phi(e):
    return I + A*e

def T(e):
    return 0.5*(I + np.dot(Phi(e), Phi(e)))

def T3(e):
    F = Phi(e)
    return (1./6)*(2*I + 3*F + np.dot(F, np.dot(F, F)))

x0 = np.random.normal(size=2)
x0 /= np.linalg.norm(x0)
x1 = np.dot(T(0.1), x0)

def com(A,B):
    return np.dot(A,B) - np.dot(B,A)

B = np.random.normal(size=4).reshape(2, 2)
B /= 2.
B = np.zeros((2, 2))
def Phi2(e):
    return I + B*0.25 + A*e


def L(e):
    return 0.5*(I + np.dot(Phi2(e), Phi2(e)))

print("")
e1 = 0.05
e2 = 0.05

x2 = np.dot(L(e2), np.dot(L(e1), x0))
print("x2: ",x2)

e1 = np.linspace(-1.1, 1.1)
e2 = e1.copy()

Z = np.zeros((e1.size, e2.size))
for i in range(e1.size):
    for j in range(e2.size):
        y = np.dot(L(e2[j]), np.dot(L(e1[i]), x0))
        Z[i,j] = np.linalg.norm(y - x2)**2


import matplotlib.pyplot as plt
from scipy.integrate import quad, odeint

def g(t):
    return np.cos(t)
def dXdt(X, t):
    return np.dot(B + A*g(t), X)

tt = np.linspace(0., 5., 100)
sol = odeint(dXdt, x0, tt)

def Phi(e, t0, t1):
    return I + B*(t1-t0) + A*e

def T(e, t0, t1):
    return 0.5*(I + np.dot(Phi(e, t0, t1), Phi(e, t0, t1)))

tvals = np.linspace(0., 5., 10)
X = [x0]
for i in range(tvals.size-1):
    E = quad(g, tvals[i], tvals[i+1])[0]
    y = np.dot(T(E, tvals[i], tvals[i+1]), X[-1])
    X.append(y)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tt, sol)
ax.plot(tvals, X, 'o')

fig2 = plt.figure()
ax = fig2.add_subplot(111)
levels = np.linspace(Z.min() + 0.1, .5, 3)
CS = ax.contour(e1, e2, Z, levels)
plt.clabel(CS, inline=1)

def objFunc(ee, x2=x2):
    y = np.dot(L(ee[1]), np.dot(L(ee[0]), x0))
    res1 = np.linalg.norm(y - x2)
    res2 = np.sqrt((y[0] - x2[0])**2 + (y[1] - x2[1])**2)
    return np.linalg.norm(y - x2)**2


from scipy.optimize import minimize
res = minimize(objFunc, [0.3, 0.3])
Q = np.linalg.inv(res.hess_inv)
ax.plot(res.x[0], res.x[1], 'o')

Z2 = np.zeros(Z.shape)
for i in range(e1.size):
    for j in range(e2.size):
#        y = np.dot(L(e2[j]), np.dot(L(e1[i]), x0))
        eta = np.array([e1[i], e2[j]]) - res.x
        Z2[i, j] = res.fun + 0.5*np.dot(eta, np.dot(Q, eta))

cs2 = ax.contour(e1, e2, Z2, levels, colors='k')
plt.clabel(cs2, inline=1)


print("---------")
X = np.array(X)
x0 = X[0,]
x1 = X[1,]
x2 = X[2,]

fig3 = plt.figure()
ax = fig3.add_subplot(111)

ax.plot(tt[tt <= tvals[2]], sol[tt <= tvals[2],:])
ax.plot(tvals[:2], X[:2,], 'o')

from trajResample import gaussEllipseSim
mean = np.array([0.1, 0.1])
cov = np.array([[1.0, 0.3],
                [0.3, 1.0]])
cov *= 3.
sol_ = sol[tt <= tvals[2],:]
delta = 0.01
for nt in range(10):
    res = minimize(objFunc, [0.3, 0.3], args=(sol_[-1,],))
    Q = np.linalg.inv(res.hess_inv)
    print(Q)
    e = gaussEllipseSim(mean, cov,
                        Q, res.x, delta)
    w = np.dot(L(e[0]), x0)
    z = np.dot(L(e[1]), w)
    print(res.fun)
    ax.plot([tvals[1]],w[0], 'bo')
    ax.plot([tvals[2]],z[0], 'bo', alpha=0.2)
    ax.plot([tvals[1]],w[1], 'rs')
    ax.plot([tvals[2]], z[1], 'rs', alpha=0.2)



plt.show()
