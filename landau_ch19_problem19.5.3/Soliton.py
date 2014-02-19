#!/usr/bin/env python

#-- Analyze what is physical here! Numerical analysis if solutions are real or artifacts!; Periodic boundary conditions!

""" From "A SURVEY OF COMPUTATIONAL PHYSICS", Python eBook Version
   by RH Landau, MJ Paez, and CC Bordeianu
   Copyright Princeton University Press, Princeton, 2011; Book  Copyright R Landau, 
   Oregon State Unv, MJ Paez, Univ Antioquia, C Bordeianu, Univ Bucharest, 2011.
   Support by National Science Foundation , Oregon State Univ, Microsoft Corp"""  

# Soliton.py: Solves Korteweg de Vries equation for a soliton.
from __future__ import division
from __future__ import with_statement
from visual import *
import matplotlib.pylab as p;
from mpl_toolkits.mplot3d import Axes3D ;
import numpy

import numpy as np
from pylab import ion
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.optimize import curve_fit
from scipy.weave import inline, converters
import mpmath as mp

import sys
import time
import cPickle as pickle

from JSAnimation import IPython_display, HTMLWriter

from smartFormat import smartFormat
from plotGoodies import plotDefaults

plotDefaults()

p.ion()

dx = 0.2  # default: 0.4
dt = 0.001 # default: 0.1

mu = 0.1
epsilon = 0.2

fac = mu*dt/(dx**3)

max_t = 200
x_range = 800

t_samp = np.arange(0, max_t+dt, dt)
max_t_idx = len(t_samp)-1
max_t = np.max(max_t)

x_samp = np.arange(-x_range/2, x_range/2, dx)
num_x = len(x_samp)
max_x_idx = num_x-1
x_range = np.max(x_samp)-np.min(x_samp)

t_betw_plt = 1
t_steps_betw_plt = int(t_betw_plt/dt)
t_betw_plt = t_steps_betw_plt*dt
mt  = int(max_t/t_betw_plt)
u   = np.zeros( (num_x, 3), float)
spl = np.zeros( (num_x, 1+int(max_t/dt/t_steps_betw_plt)), float)
x_skip = 1

#-- Amplitude: 3c/\epsilon; temporal "frequency": sqrt(c/(4\mu))
c = 4*mu

#-- Define function for initial conditions...
init_fn = lambda x: 0.5*(1-((mp.exp(2*x)-1)/(mp.exp(2*(x))+1)))
init_fn = lambda x: 1 #0.5*(1-((mp.exp(2*x)-1)/(mp.exp(2*(x))+1)))

u[:,0] = np.array([ init_fn(x) for x in x_samp ])

#-- ... and force all time-step endpoints to be the same as the IC endpoints
u[0,1] = u[0,0] 
u[0,2] = u[0,0]
u[-1,1] = u[-1,0] 
u[-1,2] = u[-1,0]

spl[:, 0] = u[:, 0]

for  i in range (1, num_x-1):                              # First time step
    a1 = epsilon*dt*(u[i + 1, 0] + u[i, 0] + u[i - 1, 0])/(dx*6.)
    if i > 1 and  i < num_x-2:
        a2 = u[i+2,0]+2.*u[i-1,0]-2.*u[i+1,0]-u[i-2,0]
    else: 
        a2 = u[i-1, 0] - u[i+1, 0]
    a3 = u[i+1, 0] - u[i-1, 0] 
    u[i, 1] = u[i, 0] - a1*a3 - fac*a2/3.

#-- C implementation
j_start = 1
j_end = max_t_idx+1
c_integrator = """
    double a1, a2, a3;
    int m=1, i, j;
    
    //-- Iterate over time
    for (j=j_start; j<=j_end; j++) {
        //-- Iterate over spatial coordinates
        for (i=1; i<num_x-1; i++) {
            a1 = epsilon*dt*(u(i + 1, 1)  +  u(i, 1)  +  u(i - 1, 1))/(3.*dx);
            if ((i > 1) && (i < num_x-2))
                a2 = u(i+2,1) + 2.*u(i-1,1) - 2.*u(i+1,1) - u(i-2,1);
            else
                a2 = u(i-1, 1) - u(i+1, 1);
            a3 = u(i+1, 1) - u(i-1, 1);
            u(i, 2) = u(i,0) - a1*a3 - 2.*fac*a2/3.;
        }
        
        //-- Save time slices to spl (values-to-plot) array
        if (j % t_steps_betw_plt ==  0) {
            for (i=1; i<num_x-2; i++)
                spl(i, m) = u(i, 2);
            m += 1;
        }
    
        //-- Shift time sequence back one
        for (i=0; i<num_x; i++) {
            u(i, 0) = u(i, 1);
            u(i, 1) = u(i, 2);
        }
    }
"""
inline(c_integrator,
       ['u', 'spl', 'dt', 'dx', 'fac', 'epsilon', 'j_start',
        'j_end', 'num_x', 't_steps_betw_plt'],
       type_converters=converters.blitz)

spl_ma = np.ma.array(spl, mask=np.isnan(spl), fill_value=0)
cmap = p.cm.ocean
cmap.set_bad(color='k',alpha=1)

#-- X coordinates
x = list(range(0, num_x, x_skip))

#-- Y coordinates
y = list(range(0, mt, t_steps_betw_plt))

#-- 2D grids with X and Y coordinates
X, Y = p.meshgrid(x, y)
def functz(spl):
    z = spl[X, Y] 
    return z

#s = surf(x, y, 20*spl)
#fig  = p.figure(1)                                         # create figure
#fig.clf()
#ax = Axes3D(fig)                                              # plot axes
#ax.plot_surface(X*dx*x_skip, Y*dt*t_steps_betw_plt, spl[X, Y],
#                cmap=p.cm.bone)#  color = 'r')                            # red wireframe
#ax.plot_surface(X*dx*x_skip, Y*dt*t_steps_betw_plt, spl[X, Y],
#                cmap=p.cm.bone)#  color = 'r')                            # red
#ax.set_xlabel('Positon')                                     # label axes
#ax.set_ylabel('Time')
#ax.set_zlabel('Disturbance')
#spl = spl-numpy.min(spl)
#spl = spl/numpy.max(spl)
f2=p.figure(figsize=(10,4), dpi=90)
f2.clf()
ax2=f2.add_subplot(111)
ax2.imshow(spl_ma.T, interpolation='nearest', cmap=p.cm.ocean,
           vmin=0, vmax=2, origin='lower')
p.xlabel(r"Position index, $x/\Delta x$")
p.ylabel(r"Time index, $t/\Delta t$")
p.axis('image')
#p.axis((0,numpy.max(x),0,numpy.max(y)))
p.tight_layout()
p.show()                                # Show figure, close Python shell
f2.savefig('output.png', dpi=600)
x=raw_input()
