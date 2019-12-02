import numpy as np

G = 6.67e-11
SI2MGAL = 10e4
T2NT = 10e9
CM = 10e-7
SI2ETVES = 10e9


class AnomalyObj():
    def __init__(self, ksi=0, eta=0, dzeta=0):
        self.x = ksi
        self.y = eta
        self.z = dzeta
        self.pstn = (ksi, eta, dzeta)
        self.props = {}

    def addprop(self, key, val):
        self.props[key] = val


class Sphere(AnomalyObj):
    def __init__(self, ksi=0, eta=0, dzeta=0, radius=0):
        super().__init__(ksi, eta, dzeta)
        self.radius = radius
        self.volume = 4 / 3 * np.pi * radius * radius * radius


def gz(x, y, sphere):
    density = sphere.props['density']
    dx = sphere.x - x
    dy = sphere.y - y
    rho = (dx ** 2 + dy ** 2 + sphere.z ** 2) ** 1.5
    gz = G * density * sphere.volume * sphere.z / rho
    return gz * SI2MGAL


def tf(x, y, sphere, t0, pmag=None):
    fx, fy, fz = dircos(t0[1], t0[2])
    tx, ty, tz = t0[0] * fx, t0[0] * fy, t0[0] * fz
    mag = sphere.props['magnetization']
    if isinstance(mag, float) or isinstance(mag, int):
        ax, ay, az = dircos(sphere.props['inclination'], sphere.props['declination'])
        mx, my, mz = mag * ax, mag * ay, mag * az
    else:
        raise ValueError
    dx = sphere.x - x
    dy = sphere.y - y
    dz = sphere.z
    dotprod = mx * dx + my * dy + mz * dz
    rho = dx ** 2 + dy ** 2 + sphere.z ** 2
    r5 = rho ** 2.5
    moment = sphere.volume
    bx = moment * (3 * dotprod * dx - rho * mx) / r5
    by = moment * (3 * dotprod * dy - rho * my) / r5
    bz = moment * (3 * dotprod * dz - rho * mz) / r5
    X, Y, Z = fx * bx, fy * by, fz * bz
    res = np.sqrt((tx + X) ** 2 + (ty + Y) ** 2 + (tz + Z) ** 2)
    res *= CM * T2NT
    return res


def bx(xp, yp, sphere):
    if xp.shape != yp.shape:
        raise ValueError("Input arrays xp, yp must have same shape!")
    size = len(xp)
    res = np.zeros(size, dtype=np.float)
    mag = sphere.props['magnetization']
    ax, ay, az = dircos(sphere.props['inclination'], sphere.props['declination'])
    mx, my, mz = ax * mag, ay * mag, az * mag
    v1 = kernelxx(xp, yp, sphere)
    v2 = kernelxy(xp, yp, sphere)
    v3 = kernelxz(xp, yp, sphere)
    res = v1 * mx + v2 * my + v3 * mz
    res *= CM * T2NT
    return res


def by(xp, yp, sphere):
    if xp.shape != yp.shape:
        raise ValueError("Input arrays xp, yp must have same shape!")
    size = len(xp)
    res = np.zeros(size, dtype=np.float)
    mag = sphere.props['magnetization']
    ax, ay, az = dircos(sphere.props['inclination'], sphere.props['declination'])
    mx, my, mz = ax * mag, ay * mag, az * mag
    v2 = kernelxy(xp, yp, sphere)
    v4 = kernelyy(xp, yp, sphere)
    v5 = kernelyz(xp, yp, sphere)
    res = v2 * mx + v4 * my + v5 * mz
    res *= CM * T2NT
    return res


def bz(xp, yp, sphere):
    if xp.shape != yp.shape:
        raise ValueError("Input arrays xp, yp must have same shape!")
    size = len(xp)
    res = np.zeros(size, dtype=np.float)
    mag = sphere.props['magnetization']
    ax, ay, az = dircos(sphere.props['inclination'], sphere.props['declination'])
    mx, my, mz = ax * mag, ay * mag, az * mag
    v3 = kernelxz(xp, yp, sphere)
    v5 = kernelyz(xp, yp, sphere)
    v6 = kernelzz(xp, yp, sphere)
    res = v3 * mx + v5 * my + v6 * mz
    res *= CM * T2NT
    return res


def gxx(xp, yp, sphere):
    res = np.zeros(len(xp), dtype=np.float)
    density = sphere.props['density']
    res = density * kernelxx(xp, yp, sphere)
    res *= G * SI2ETVES
    return res


def gxy(xp, yp, sphere):
    res = np.zeros(len(xp), dtype=np.float)
    density = sphere.props['density']
    res = density * kernelxy(xp, yp, sphere)
    res *= G * SI2ETVES
    return res


def gxz(xp, yp, sphere):
    res = np.zeros(len(xp), dtype=np.float)
    density = sphere.props['density']
    res = density * kernelxz(xp, yp, sphere)
    res *= G * SI2ETVES
    return res


def gyy(xp, yp, sphere):
    res = np.zeros(len(xp), dtype=np.float)
    density = sphere.props['density']
    res = density * kernelyy(xp, yp, sphere)
    res *= G * SI2ETVES
    return res


def gyz(xp, yp, sphere):
    res = np.zeros(len(xp), dtype=np.float)
    density = sphere.props['density']
    res = density * kernelyz(xp, yp, sphere)
    res *= G * SI2ETVES
    return res


def gzz(xp, yp, sphere):
    res = np.zeros(len(xp), dtype=np.float)
    density = sphere.props['density']
    res = density * kernelzz(xp, yp, sphere)
    res *= G * SI2ETVES
    return res


def kernelxx(xp, yp, sphere):
    if xp.shape != yp.shape:
        raise ValueError("Input arrays xp, yp must have same shape!")
    dx = sphere.x - xp
    dy = sphere.y - yp
    dz = sphere.z
    r_2 = (dx ** 2 + dy ** 2 + dz ** 2)
    r_5 = r_2 ** (2.5)
    volume = sphere.volume
    return volume * (((3 * dx ** 2) - r_2) / r_5)


def kernelxy(xp, yp, sphere):
    if xp.shape != yp.shape:
        raise ValueError("Input arrays xp, yp must have same shape!")
    dx = sphere.x - xp
    dy = sphere.y - yp
    dz = sphere.z
    r_2 = (dx ** 2 + dy ** 2 + dz ** 2)
    r_5 = r_2 ** (2.5)
    volume = sphere.volume
    return volume * ((3 * dx * dy) / r_5)


def kernelxz(xp, yp, sphere):
    if xp.shape != yp.shape:
        raise ValueError("Input arrays xp, yp must have same shape!")
    dx = sphere.x - xp
    dy = sphere.y - yp
    dz = sphere.z
    r_2 = (dx ** 2 + dy ** 2 + dz ** 2)
    r_5 = r_2 ** (2.5)
    volume = sphere.volume
    return volume * ((3 * dx * dz) / r_5)


def kernelyy(xp, yp, sphere):
    if xp.shape != yp.shape:
        raise ValueError("Input arrays xp, yp must have same shape!")
    dx = sphere.x - xp
    dy = sphere.y - yp
    dz = sphere.z
    r_2 = (dx ** 2 + dy ** 2 + dz ** 2)
    r_5 = r_2 ** (2.5)
    volume = sphere.volume
    return volume * (((3 * dy ** 2) - r_2) / r_5)


def kernelyz(xp, yp, sphere):
    if xp.shape != yp.shape:
        raise ValueError("Input arrays xp, yp must have same shape!")
    dx = sphere.x - xp
    dy = sphere.y - yp
    dz = sphere.z
    r_2 = (dx ** 2 + dy ** 2 + dz ** 2)
    r_5 = r_2 ** (2.5)
    volume = sphere.volume
    return volume * ((3 * dy * dz) / r_5)


def kernelzz(xp, yp, sphere):
    if xp.shape != yp.shape:
        raise ValueError("Input arrays xp, yp must have same shape!")
    dx = sphere.x - xp
    dy = sphere.y - yp
    dz = sphere.z
    r_2 = (dx ** 2 + dy ** 2 + dz ** 2)
    r_5 = r_2 ** (2.5)
    volume = sphere.volume
    return volume * (((3 * dz ** 2) - r_2) / r_5)


def mk_axis(amin, amax, delta):
    return np.arange(amin, amax, delta)


def mk_grid(ax, ay):
    X, Y = np.meshgrid(ax, ay)
    return X, Y


def dircos(inc, dec):
    d2r = np.pi / 180
    vect = [np.cos(d2r * inc) * np.cos(d2r * dec),
            np.cos(d2r * inc) * np.sin(d2r * dec),
            np.sin(d2r * inc)]
    return vect


def line_path(px, py, grd):
    return grd[px], grd[py]


def save_data(x, y, z, filename):
    with open(filename, 'w') as f:
        f.write('x\ty\tz')
        for _ in z:
            f.write(f'{x}\t{y}\t{z}')