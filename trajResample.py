import numpy as np
import matplotlib.pyplot as plt

def ellipseSample(Q,c, size=1):
    rootc = np.sqrt(c)
    L = np.linalg.cholesky(Q)

    l, U = np.linalg.eig(Q)
    rootl = np.sqrt(l)

    D = np.diag(rootl)
    Dinv = np.linalg.inv(D)

    print(Q)
    print(np.dot(U, np.dot(D, np.dot(D, U.T))))
    A = np.linalg.inv(np.dot(D, U.T))
    z = np.random.normal(size=2*size).reshape(size, 2)
    for i in range(size):
        z[i, ] = ballSample(rad=2*c)
        print(c, np.dot(z[i, ], z[i, ]))
        z_ = z[i, ].copy()

        z[i, ] = np.dot(A, z_)


    return z

def ballSample(size=1, rad=1.):
    while True:
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)

        vv = np.array([x, y])
        if np.linalg.norm(vv) <= rad:
            break
    return vv

def gaussEllipseSim(mean, LCov, Q, centre, rad):
    while True:
        z = np.random.normal(size=2)
        z = mean + np.dot(LCov, z)
        if np.dot(z - centre, np.dot(Q, z - centre)) <= rad:
            break
    return z

"""
Q = np.array([[3.04, 2.22],
              [2.22, 1.84]])

xx = np.linspace(-2., 5.)
yy = np.linspace(-2., 5.)

Z = np.zeros((xx.size, yy.size))

from scipy.stats import multivariate_normal

mean = np.array([1., 0.1])
cov = np.array([[1.5, -0.3],[-0.3, 2.5]])
L = np.linalg.cholesky(cov)

for i in range(xx.size):
    for j in range(yy.size):
        Z[i,j] = multivariate_normal.pdf( [xx[i], yy[j]],
                                          mean=mean,
                                          cov=cov)

plt.contour(xx, yy, Z.T)

rv = np.zeros((100, 2))
for nt in range(rv.shape[0]):
    rv[nt, ] = gaussEllipseSim(mean, L, Q, [-.5, 1.5], 0.25)
plt.plot(rv[:,0], rv[:,1], 'k+')

plt.show()
"""
