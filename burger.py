
"""
Nonlinear burgers equation (time dependent)
"""


from firedrake import *

# Square mesh of some resultion
n = 30
mesh = UnitSquareMesh(n,n)

# Lagrange polys of degree 2 and 1 (linear)
V2 = VectorFunctionSpace(mesh, "CG", 2)
V1 = VectorFunctionSpace(mesh, "CG", 1)

u_ = Function(V2, name='velocity')
u = Function(V2, name='velocity_next')
v = TestFunction(V2)

x = SpatialCoordinate(mesh)
ic = project(as_vector([sin(pi*x[0]), 0]), V2)

u_.assign(ic)
u.assign(ic)

nu = 0.0001
dt = 1.0/n
F = ( inner((u-u_)/dt, v) + inner(dot(u,nabla_grad(u)), v) + nu*inner(grad(u),grad(v)) )*dx


output = VTKFile('burger.pvd')

t = 0.0
T = 0.5
while (t <= T):
    solve(F == 0, u)
    u_.assign(u)
    t += dt
    output.write( project(u, V1, name='velocity') )


import matplotlib.pyplot as plt
fig, axes = plt.subplots()
colors = tripcolor(u, axes=axes)
fig.colorbar(colors)
plt.savefig('burger.png')
