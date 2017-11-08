import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

plt.close("all")
palette = colors.LinearSegmentedColormap.from_list("new", 
        [[1,0,0],[0,1, 0],[0,0,1]], N=10)

rng = np.random.RandomState()

def dynamics(M, stime=2000, h=0.3, init=None):
    if init is None:
        init = np.random.randn(3)
    n = M.shape[0]
    x = np.zeros([stime, n])
    o = np.zeros([stime, n])
    x[0] = np.zeros(n)
    x[0,:3] = init
    o[0] = x[0].copy()
    for t in range(1,stime):
        x[t] = x[t-1] + h*(-x[t-1] + np.dot(M, o[t-1]))
        o[t] = np.tanh(x[t])
    return o

def esp(m, h=0.1, epsilon=.001):

    m = m/np.abs(np.linalg.eigvals(m)).max()
    
    e = np.linalg.eigvals(m)
    rho = abs(e).max()
    x = e.real
    y = e.imag
        
    # solve quadratic equations
    target = 1.0 - epsilon/2.0
    a = x**2*h**2 + y**2*h**2
    b = 2*x*h - 2*x*h**2
    c = 1 + h**2 - 2*h - target**2
    # just get the positive solutions
    sol = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    # and take the minor amongst them
    effective_rho = sol.min()
        
    m *= effective_rho
    return m


n = 3
M = rng.randn(n, n)

Ma = (M - M.T)*0.5
Ms = (M + M.T)*0.5

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter([1],[1],[1], s=50, c="blue")
ax.scatter([0],[0],[0], s=200, c="green")

init = np.ones(3)

for t,alpha in enumerate(np.linspace(0.4, 1.0, 10)):
    m = Ms*alpha + Ma*(1 - alpha)
    m = esp(m, h=0.08)
    x = dynamics(m, h=0.08, init=init)
    ax.plot(*x[:,:3].T,  color=palette(t))
    ax.set_xlim([-.6, 1.6])
    ax.set_ylim([-.6, 1.6])
    ax.set_zlim([-.6, 1.6])
plt.show()

