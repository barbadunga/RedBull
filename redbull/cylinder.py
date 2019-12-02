import numpy as np

G = 6.67e-11
SI2MGAL = 10e4
T2NT = 10e9
CM = 10e-7
SI2ETVES = 10e9


class AnomalyObj():
    def __init__(self, ksi=0, dzeta=0):
        self.x = ksi
        self.z = dzeta
        self.pstn = (ksi, dzeta)
        self.props = {}

    def addprop(self, key, val):
        self.props[key] = val


class Cylinder(AnomalyObj):
    def __init__(self, ksi, dzeta, radius):
        super().__init__(ksi, dzeta)
        self.radius = radius


def gz(x, cylinder):
    density = cylinder.props['density']
    radius2 = cylinder.radius * cylinder.radius
    sigma = np.pi * radius2 * density
    dx = cylinder.x - x
    dz = cylinder.z
    r2 = dx ** 2 + dz ** 2
    g = 2 * G * sigma * cylinder.z / r2
    return SI2MGAL * g


def bx(x, cylinder, a):
    mag = cylinder.props['magnetization']
    ax, az = dircos(cylinder.props['inclination'], cylinder.props['declination'], a)
    mx, mz = ax * mag, az * mag
    vxz = kernelxz(x, cylinder)
    vzz = kernelzz(x, cylinder)
    res = mz * vxz - vzz * mx
    res *= CM * T2NT
    return res


def bz(x, cylinder, a):
    mag = cylinder.props['magnetization']
    ax, az = dircos(cylinder.props['inclination'], cylinder.props['declination'], a)
    mx, mz = ax * mag, az * mag
    vxz = kernelxz(x, cylinder)
    vzz = kernelzz(x, cylinder)
    res = mx * vxz + vzz * mz
    res *= CM * T2NT
    return res


def kernelzz(x, cylinder):
    dx = cylinder.x - x
    dz = cylinder.z
    r2 = dx ** 2 + dz ** 2
    r4 = r2 ** 2
    return 2 * (dz ** 2 - dx ** 2) / r4


def kernelxz(x, cylinder):
    dx = cylinder.x - x
    dz = cylinder.z
    r2 = dx ** 2 + dz ** 2
    r4 = r2 ** 2
    return 4 * dx * dz / r4


def dircos(inc, dec, a):
    d2r = np.pi / 180
    vect = [np.cos(d2r * inc) * np.cos(d2r * (a - dec)),
            np.sin(d2r * inc)]
    return vect