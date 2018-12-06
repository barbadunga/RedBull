import numpy as np

G = 6.67e-11
SI2MGAL = 10e4


class AnomalyObj():
    def __init__(self, ksi=0, eta=0, dzeta=0):
        self.x0 = ksi
        self.y0 = eta
        self.z0 = dzeta
        self.pstn = (ksi, eta, dzeta)


class Sphere(AnomalyObj):
    def __init__(self, ksi=0, eta=0, dzeta=0, density=0, radius=0):
        super().__init__(ksi, eta, dzeta)
        self.density = density
        self.radius = radius
        self.volume = 4 / 3 * np.pi * radius * radius * radius

    def g_z(self, x, y):
        dx = self.x0 - x
        dy = self.y0 - y
        rho = (dx ** 2 + dy ** 2 + self.z0 ** 2) ** 1.5
        g_z = G * self.density * self.volume * self.z0 / rho
        return g_z * SI2MGAL


def mk_axis(amin, amax, delta):
    return np.arange(amin, amax, delta)


def mk_grid(ax, ay):
    X, Y = np.meshgrid(ax, ay)
    return X, Y