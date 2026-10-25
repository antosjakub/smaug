"""
Fokker-Planck (Smoluch)
"""


# LinearVariationalSolver



from firedrake import *

# Square mesh of some resultion
n = 100
lim = 3.0
mesh = RectangleMesh(n,n, lim,lim, -lim,-lim)

# Lagrange polys of degree 2
V = FunctionSpace(mesh, "CG", 2)

u_ = Function(V, name='prob')
u = Function(V, name='prob_next')
v = TestFunction(V)

x = SpatialCoordinate(mesh)
ic = Function(V)
ic.interpolate( 1/(2*pi) * exp(- 0.5*(x[0]**2 + x[1]**2)) )

u_.assign(ic)
u.assign(ic)

alpha = 1.0
beta = 1.0

well_depth = [0.8067, 1.0391, 1.0101, 0.7722, 1.0593, 1.0706, 0.9805, 0.9893, 0.9514, 1.1419]
a1 = well_depth[0]
a2 = well_depth[1]

#dt = 1.0/n
dt = 0.1
print(f"dt={dt}")

pot = 0.25 * ( (x[0]**2-a1**2)**2 + (x[1]**2-a2**2)**2 )
#grad_pot = [ 1/4 * 2*(x[0]**2-a1**2)*(2*x[0]), 1/4 * 2*(x[1]**2-a2**2)*(2*x[1]) ]

F = ( inner((u-u_)/dt, v) + alpha*inner(u, inner(grad(pot), grad(v))) + alpha/beta*inner(grad(u), grad(v)) )*dx

output = VTKFile('fp.pvd')

t = 0.0
n_steps = 10

animate = True

if animate:
    output_freq = 1
    us = []
    us.append(ic.copy(deepcopy=True))

for step in range(n_steps):
    print(f"step = {step}")
    solve(F == 0, u)
    u_.assign(u)
    t += dt
    output.write( project(u, V, name='prob') )
    if animate and step % output_freq == 0:
        us.append(u.copy(deepcopy=True))
    step += 1



from firedrake.pyplot import FunctionPlotter, tripcolor, tricontour, tricontourf

import os
os.makedirs('fp_img/', exist_ok=True)

if animate:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import Normalize
    nsp = 64

    print(len(us))

    fn_plotter = FunctionPlotter(mesh, num_sample_points=nsp)
    fig, axes = plt.subplots()
    axes.set_aspect('equal')
    colors = tripcolor(ic, num_sample_points=nsp, vmin=1, vmax=2, axes=axes)
    #colors = tricontourf(ic, num_sample_points=nsp, axes=axes, levels=40)
    cbar = fig.colorbar(colors)

    lines = tricontour(ic, num_sample_points=nsp, axes=axes, levels=25, colors='black', linewidth=0.4)

    def update_frame(u):
        global lines
        data = fn_plotter(u)
        colors.set_array(data)
        colors.norm = Normalize(vmin=data.min(), vmax=data.max())
        cbar.update_normal(colors)
        lines.remove()
        lines = tricontour(u, num_sample_points=nsp, axes=axes, levels=25, colors='black', linewidth=0.4)

        return ()

    animation = FuncAnimation(fig, update_frame, frames=us)
    animation.save("fp_img/fp.gif", writer='pillow', fps=1, dpi=300)


import matplotlib.pyplot as plt
fig, axes = plt.subplots()
colors = tripcolor(ic, axes=axes)
#colors = tricontourf(ic, axes=axes, levels=50)
lines = tricontour(ic, axes=axes, levels=25, colors='black', linewidth=0.4)
fig.colorbar(colors)
plt.savefig('fp_img/fp_ic.png', dpi=300)

import matplotlib.pyplot as plt
fig, axes = plt.subplots()
colors = tripcolor(u, axes=axes)
#colors = tricontourf(u, axes=axes, levels=40)
lines = tricontour(u, axes=axes, levels=25, colors='black', linewidth=0.4)
fig.colorbar(colors)
plt.savefig('fp_img/fp_tc.png', dpi=300)
