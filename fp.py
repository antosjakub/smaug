"""
Fokker-Planck (Smoluch)
"""


# LinearVariationalSolver



from firedrake import *

# Square mesh of some resultion
n = 200
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


print(len(us))

import os
os.makedirs('fp_img/', exist_ok=True)

if animate:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import Normalize
    nsp = 16

    fn_plotter = FunctionPlotter(mesh, num_sample_points=nsp)
    fig, axes = plt.subplots()
    axes.set_aspect('equal')
    #colors = tripcolor(ic, num_sample_points=nsp, vmin=1, vmax=2, axes=axes)
    colors = tripcolor(ic, num_sample_points=nsp, axes=axes)
    cbar = fig.colorbar(colors)

    def update_frame(u):
        data = fn_plotter(u)
        colors.set_array(data)
        colors.norm = Normalize(vmin=data.min(), vmax=data.max())
        cbar.update_normal(colors)
        return (colors,)

    animation = FuncAnimation(fig, update_frame, frames=us)
    try:
        animation.save("fp_img/fp.gif", writer='pillow', fps=1.5)
    except:
        print("Failed to write movie! Try installing `ffmpeg`.")


import matplotlib.pyplot as plt
fig, axes = plt.subplots()
colors = tripcolor(ic, axes=axes)
fig.colorbar(colors)
plt.savefig('fp_img/fp_ic.png')

import matplotlib.pyplot as plt
fig, axes = plt.subplots()
colors = tripcolor(u, axes=axes)
fig.colorbar(colors)
plt.savefig('fp_img/fp_tc.png')
