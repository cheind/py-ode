import numpy as np
import matplotlib.pyplot as plt

EARTH_MASS = 5.972e24 # kg
OBJECT_MASS = 2. #kg
G = 6.67408e-11 # Gravitational constant m3 kg-1 s-2
T = 24 * 3600 # Time for desired orbital period s
RADIUS = ((G * EARTH_MASS)/ ((2*np.pi/T)**2))**(1/3) # m
TANGENTIAL_SPEED = RADIUS * np.pi * 2 / T

def force(position):
    # Fg = G * m1 * m2 / r^2
    r = np.linalg.norm(position)
    f = G * EARTH_MASS * OBJECT_MASS / (r*r)
    return -position * f / r

class ExplicitEuler:

    def __init__(self, x0, v0, mass):
        self.x = x0
        self.v = v0
        self.invmass = 1. / mass

    def update(self, h, forcegen):
        xnew = self.x + h * self.v
        self.v += h * forcegen(self.x) * self.invmass
        self.x = xnew
        return self.x, self.v

class SymplecticEuler:

    def __init__(self, x0, v0, mass):
        self.x = x0
        self.v = v0
        self.invmass = 1. / mass

    def update(self, h, forcegen):
        self.x += h * self.v
        self.v += h * forcegen(self.x) * self.invmass
        return self.x, self.v

def energy(x, v):
    return 0.5 * OBJECT_MASS * np.dot(v,v) - G * EARTH_MASS * OBJECT_MASS / np.linalg.norm(x)

def run():
    steps = 1000
    periods = 20
    h = T * periods / steps

    xs = np.zeros((steps + 1, 2))
    vs = np.zeros((steps + 1, 2))
    es = np.zeros((steps + 1, 2))
    ts = np.zeros((steps + 1, 2))

    xs[0] = np.asarray([RADIUS, 0])    
    vs[0] = np.asarray([0, TANGENTIAL_SPEED])
    es[0] = energy(xs[0], vs[0])
    solver = SymplecticEuler(xs[0], vs[0], OBJECT_MASS)

    for i in range(steps):
        xs[i + 1], vs[i + 1] = solver.update(h, force)
        es[i + 1] = energy(xs[i + 1], vs[i + 1])
        ts[i + 1] = ts[i] + h

    fig = plt.figure()
    p = fig.add_subplot(111, aspect='equal')
    p.scatter(0, 0, s=40, marker='o', color='b')
    p.scatter(xs[:, 0], xs[:, 1], s=1, marker='o', alpha=0.5, antialiased=True)    

    fig = plt.figure()
    p = fig.add_subplot(111)
    p.plot(ts, es)

    plt.show()



if __name__ == "__main__":
    run()        