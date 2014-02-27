# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from __future__ import division
import time
import numpy as np
from pylab import ion
import matplotlib as mpl
import matplotlib.pylab as p
from matplotlib import pyplot as plt
from matplotlib import animation
ion()

# <markdowncell>

# Define the successive overrelaxation algorithm's "step" (each gridpoint gets updated once)

# <codecell>

def SOR_step(vx, vy, P, Nx, Ny, boundaries=[], omega=1):
    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            if (i,j) not in boundaries:
                new_vx = 0.25*(vx[i+1,j] + vx[i-1,j] + vx[i,j+1] + vx[i,j-1] \
                           -h/2.*vx[i,j]*(vx[i+1,j]-vx[i-1,j]) \
                           -h/2.*vy[i,j]*(vx[i,j+1]-vx[i,j-1]) \
                           -h/2.*(P[i+1,j]-P[i-1,j])
                            )
                residual = new_vx - vx[i,j]

                vx[i,j] = vx[i,j]+omega*residual

# <markdowncell>

# Parameters

# <codecell>

Nx = 100
Ny = 100
h = 2
L = 3
nu = 1
rho = 10e3
V0 = 1
domain_width = 10 # m
domain_height = 5 # m
omega = 1.2

# <markdowncell>

# Set initial velocities and pressures in entire region

# <codecell>

vx = np.zeros((Nx, Ny))
vy = np.zeros((Nx, Ny))
P =  np.zeros((Nx,Ny))

# <codecell>

pixel_width = domain_width/Nx
pixel_height = domain_height/Ny
plate_y0 = int(np.round(Ny/2+h/2/pixel_height))
plate_y1 = int(np.round(Ny/2-h/2/pixel_height))
plate_x0 = int(np.round(Nx/2-L/2/pixel_width))
plate_x1 = int(np.round(Nx/2+L/2/pixel_width))

# <markdowncell>

# Specify boundary conditions around edges

# <codecell>

vx[:,0] = 0 #bottom
vx[:,-1] = 0 #top
vx[0,:] = V0 #left
vx[-1,:] = 0 #rigfht

vy[0,:] = 0
vy[-1,:] = 0
vy[:,0] = 0
vy[:,-1] = 0

P[0,:] = 0
P[-1,:] = 0
P[:,0] = 0
P[:,-1] = 0

# <markdowncell>

# Specify boundary conditions on plates

# <codecell>

vx[plate_x0:plate_x1+1,plate_y0] = 0
vx[plate_x0:plate_x1+1,plate_y1] = 0
boundaries = []
for i in range(plate_x0, plate_x1+1):
    for j in [plate_y0, plate_y1]:
        boundaries.append( (i,j) )

# <codecell>

#%pylab inline

# <markdowncell>

# Plot initial conditions

# <codecell>

f1=plt.figure(1, figsize=(15,3), dpi=60)
f1.clf()
ax1=f1.add_subplot(111)
ax1.imshow(vx.T, interpolation='nearest', cmap=plt.cm.jet, origin='lower')
title(r"Intitial conditions, $v_x$")

# <headingcell level=2>

# Perform simulation!

# <codecell>

iteration = 0
error = 100
old_vx = vx.copy()
while np.max(error) > 0.0001:
    SOR_step(vx, vy, P, Nx, Ny, boundaries=boundaries, omega=omega)
    error = np.abs(vx-old_vx)
    old_vx = vx.copy()
    #     if iteration % 10 == 0:
    #         ax1.imshow(vx.T, interpolation='nearest', cmap=plt.cm.jet,
    #                vmin=0, vmax=2, origin='lower')
    #         plt.draw()
    #         plt.show()
    #         time.sleep(0.2)
    iteration += 1
print iteration

# <markdowncell>

# Plot results

# <codecell>

f2=plt.figure(2, figsize=(8,6.5), dpi=60)
f2.clf()
ax2=f2.add_subplot(111)
im2 = ax2.imshow(vx.T, interpolation='nearest', cmap=plt.cm.jet, origin='lower')
f2.colorbar(im2, ax=ax2)
title(r"Converged solution, $v_x(x,y)$")
xlabel(r"$x$"); ylabel(r"$y$")
tight_layout()

