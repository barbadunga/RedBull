import redbull as rb
import matplotlib.pyplot as plt

x = rb.mk_axis(0, 1277500, 2500)
y = rb.mk_axis(0, 1277500, 2500)
X, Y = rb.mk_grid(x, y)

shr1 = rb.Sphere(630000, 630000, 10000, 2600, 1000)
shr2 = rb.Sphere(600000, 600000, 20000, 2600, 1500)
shr3 = rb.Sphere(630000, 630000, 40000, 2600, 2500)

dg1 = shr1.g_z(X, Y)
dg2 = shr2.g_z(X, Y)
dg3 = shr3.g_z(X, Y)

g = dg2 + dg1 + dg3

with open('some.dat', 'w') as f:
    for x in X:
        f.write(str(x))

plt.contourf(X, Y, g, 30)
plt.colorbar()
plt.show()