

from firedrake import *

mesh = UnitSquareMesh(10,10)

V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))

a = ( inner(grad(u), grad(v)) + inner(u,v) ) * dx
L = inner(f,v) * dx


u = Function(V)
solve(a == L, u, solver_parameters = {'ksp_type': 'cg', 'pc_type': 'none'})


#VTKFile("helmholtz.pvd").write(u)

import matplotlib.pyplot as plt
fig, axes = plt.subplots()
colors = tripcolor(u, axes=axes)
fig.colorbar(colors)
plt.savefig('works.png')
