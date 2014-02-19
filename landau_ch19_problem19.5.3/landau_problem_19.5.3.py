# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Problem 19.5.3: 

# <markdowncell>

# The MIT License (MIT)
# 
# Copyright (c) 2014 J.L. Lanfranchi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# <codecell>

#!/usr/bin/env python

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
import scipy.signal as sig
import mpmath as mp

import sys
import time
import cPickle as pickle

from JSAnimation import IPython_display, HTMLWriter

from smartFormat import smartFormat
from plotGoodies import plotDefaults, generateColorCycle

plotDefaults()

p.ion()

# <headingcell level=2>

# User-set simulation parameters

# <markdowncell>

# Note that the multi-precision math module, mpmath, is called in defining the tanh to avoid the exponential overflowing mere mortal floating-point data types.

# <codecell>

#-- Spatial and temporal step sizes...
dx = 0.1  # default: 0.4
dt = 0.001 # default: 0.1

mu = 0.1
epsilon = 0.2

fac = mu*dt/(dx**3)              

max_t = 500
x_range0 = 800

t_betw_plt = 1

# <markdowncell>

# Define a function to use for initial conditions...

# <codecell>

#-- tanh
init_fn = lambda x: 0.5*(1-((mp.exp(2*x)-1)/(mp.exp(2*(x))+1)))

#-- Initialize to all 1's to test numerical stability
# init_fn = lambda x: 1

# <headingcell level=2>

# Initialization

# <codecell>

t_samp = np.arange(0, max_t+dt, dt)
max_t_idx = len(t_samp)-1
max_t = np.max(max_t)

x_samp = np.arange(-x_range0/2, x_range0/2+dx, dx)
num_x = len(x_samp)
max_x_idx = num_x-1
x_range = np.max(x_samp)-np.min(x_samp)
print max_x_idx
t_skip = int(t_betw_plt/dt)
t_betw_plt = t_skip*dt
mt  = int(max_t/t_betw_plt)
t_samp_skip = np.arange(0, mt+1)*t_betw_plt
u   = np.zeros( (num_x, 3), float)
spl = np.zeros( (num_x, 1+int(max_t/dt/t_skip)), float)
x_skip = 1

# #-- 2D grids with X and Y coordinates
# X, Y = p.meshgrid(x, y)

# #-- Amplitude: 3c/\epsilon; temporal "frequency": sqrt(c/(4\mu))
# c = 4*mu

# <markdowncell>

# Put initial data into u array

# <codecell>

u[:,0] = np.array([ init_fn(x) for x in x_samp ])

#-- force all time-steps to have same spatial endpoints as the IC endpoints
u[0,1] = u[0,0] 
u[0,2] = u[0,0]
u[-1,1] = u[-1,0] 
u[-1,2] = u[-1,0]

spl[:, 0] = u[:, 0]

# <markdowncell>

# Test that the stability condition is satisfied

# <codecell>

trunc_error_bigo = dt**3+dt*dx**2
stab_cond = dt/dx*(epsilon*np.abs(u[:,0])+4*mu/dx**2)
print trunc_error_bigo, np.max(np.abs(stab_cond))

# <headingcell level=2>

# C-code for the integration routine.

# <markdowncell>

# Note the included `j_start` and `j_end` parameters, which allow this routine to be run in spurts for intermediate display or whatnot from a calling Python routine.

# <codecell>

#-- C implementation of "normal" time step integrator
c_integrator = """
double a1, a2, a3;
int spl_col=1;
int i, j;

//-- Iterate over time
for (j=j_start; j<=j_end; j++) {

    //-- Iterate over spatial coordinates (exclude the endpoints)
    for (i=1; i<num_x-1; i++) {
        a1 = epsilon * dt * ( u(i+1,1) + u(i,1) + u(i-1,1) ) / (3*dx);
        if ((i >= 2) && (i <= num_x-3))
            a2 = u(i+2,1) + 2*u(i-1,1) - 2*u(i+1,1) - u(i-2,1);
        else
            a2 = u(i-1,1) - u(i+1,1);
        a3 = u(i+1,1) - u(i-1,1);
        
        u(i,2) = u(i,0) - a1*a3 - 2*fac*a2/3;
        
        //-- Save computed sample to spl array
        if (j % t_skip == 0)
            spl(i,spl_col) = u(i,2);
    }
    
    //-- Save endpoints to spl array
    if (j % t_skip == 0) {
        spl(0,spl_col) = u(0,2);
        spl(num_x-1,spl_col) = u(0,2);
        
        //-- Increment spl column counter
        spl_col += 1;
    }
    

    //-- Roll all values in u back by one position
    for (i=0; i<num_x; i++) {
        u(i, 0) = u(i, 1);
        u(i, 1) = u(i, 2);
    }
    
}
"""

# <headingcell level=2>

# Run simulation

# <markdowncell>

# First time step is unique to get the algorithm started. This is implemented in pure Python, because high performance on a single step isn't a big deal.

# <codecell>

for  i in range (1, num_x-1):
    a1 = epsilon*dt*(u[i + 1, 0] + u[i, 0] + u[i - 1, 0])/(dx*6.)
    if i > 1 and  i < num_x-2:
        a2 = u[i+2,0]+2.*u[i-1,0]-2.*u[i+1,0]-u[i-2,0]
    else: 
        a2 = u[i-1, 0] - u[i+1,0]
    a3 = u[i+1,0] - u[i-1,0]
    u[i,1] = u[i,0] - a1*a3 - fac*a2/3

# <markdowncell>

# Remaining time steps are executed below, and use the compiled C function for higher performance.

# <codecell>

j_start = 1
j_end = max_t_idx
inline(c_integrator,
       ['u', 'spl', 'dt', 'dx', 'fac', 'epsilon', 'j_start',
        'j_end', 'num_x', 't_skip'],
       type_converters=converters.blitz)

# <headingcell level=2>

# Analyze and plot results

# <markdowncell>

# Stuff the result into a **masked array** named `spl_ma`. This allows for `NaN` values to exist within the array without breaking all other numerical operations on the array (so normalizing the array works as if no `NaN` values were present, for example). This also allows for plotting the data without breaking the plotting routines.
# 
# Subsequent arrays are derived from `spl_ma` for plotting and data analysis.

# <codecell>

#-- Masked array with results, masking off any NaN entries
spl_ma = np.ma.array(spl, mask=np.isnan(spl), fill_value=0)

#-- Normalized version of the array, range from 0-1 (linear)
spl_lin_norm = (spl_ma-spl_ma.min())/(spl_ma.max()-spl_ma.min())

#-- Normalized version of the array, range from 0-1 (log scale)
spl_log_norm = np.log10((spl_ma-spl_ma.min())/(spl_ma.max()-spl_ma.min())*(10-1e0)+1e0)

#-- "Derivative" array to examine numerical instabilities. Range from 0-1 (linear)
spl_diff = np.diff(spl_ma, axis=0)
spl_diff = (spl_diff-spl_diff.min())/(spl_diff.max()-spl_diff.min())

# <markdowncell>

# 2D colormap plot

# <codecell>

cmap = p.cm.ocean
cmap.set_bad(color='k',alpha=1)

f2=p.figure(figsize=(15,1.5), dpi=40)
f2.clf()
ax2=f2.add_subplot(111)
ax2.imshow(spl_ma.T, interpolation='bicubic', cmap=p.cm.ocean,
           vmin=0, vmax=2, origin='lower')
p.xlabel(r"Position index, $x/\Delta x$")
p.ylabel(r"Dec. time index, $\frac{t}{\Delta t\cdot\mathrm{DF}}$")
p.axis('image')
#p.axis((0,numpy.max(x),0,numpy.max(y)))
p.tight_layout()
p.show()
f2.savefig('output.png', dpi=600)

# <codecell>

cmap = p.cm.ocean
cmap.set_bad(color='k',alpha=1)

f2=p.figure(figsize=(10,10), dpi=40)
f2.clf()
ax2=f2.add_subplot(111)
ax2.imshow(spl_ma[3300:4700,:].T, interpolation='bicubic', cmap=p.cm.ocean,
           vmin=0, vmax=2, origin='lower')
p.xlabel(r"Position, $x$")
p.ylabel(r"Time, $t$")
p.axis('image')
p.xticks([])
p.yticks([])
title(r"Detail, central region of above plot")
#p.axis((0,numpy.max(x),0,numpy.max(y)))
p.tight_layout()
p.show()
f2.savefig('output2.png', dpi=600)

# <markdowncell>

# Locate peaks, crudely, in the data

# <codecell>

pk_indices = []
#pk_indices, = np.array(sig.argrelmax(spl_ma[:,ind], order=5, mode='clip'))
for idx in xrange(0, size(spl_ma,1)):
    indices = sig.argrelmax(spl_ma[:,idx], order=5, mode='clip')[0]
    indices.sort()
    pk_indices.append(indices)
print len(pk_indices) #, "peaks identified"

# <markdowncell>

# Filter peaks by their prominence: If a peak's prominence is above a given threshold, then record it. Otherwise, ignore it.

# <codecell>

prom_thresh = 0.05
good_pk_ind = []
good_pk_amp = []
for idx in xrange(0, size(spl_ma,1)):
    good_pk_ind.append([])
    good_pk_amp.append([])
    
    if len(pk_indices[idx]) == 0:
        continue
    
    amp = spl_ma[:,idx]
    for pkindind in xrange(0,len(pk_indices[idx])):
        pkind = pk_indices[idx][pkindind]
        if pkindind == 0:
            left_neighb_ind = 0
        else:
            left_neighb_ind = pk_indices[idx][pkindind-1]
            
        if pkindind == len(pk_indices[idx])-1:
            right_neighb_ind = pk_indices[idx][len(pk_indices[idx])-1]
        else:
            right_neighb_ind = pk_indices[idx][pkindind+1]
        
        left_minamp = np.min(amp[left_neighb_ind:pkind])
        right_minamp = np.min(amp[pkind:right_neighb_ind+1])
        
        if left_minamp < amp[pkind]-prom_thresh and \
                right_minamp < amp[pkind]-prom_thresh:
            good_pk_ind[-1].append(pkind)
            good_pk_amp[-1].append(amp[pkind])

# <markdowncell>

# Plot data and the peaks that were found (for one timeslice) to verify the above algorithm is working

# <codecell>

ind = 500
good_pk_x = x_samp[good_pk_ind[ind]]
f20 = figure(20, figsize=(7,4))
f20.clf()
ax20 = f20.add_subplot(111)
ax20.plot(x_samp, spl_ma[:,ind], 'b-', label="Data slice")
ax20.plot(good_pk_x, good_pk_amp[ind], 'o',
          markersize=5, markeredgecolor=(1,0,0,0.75),
          markeredgewidth=1,
          markerfacecolor='none', label="Identified peaks")
ax20.set_xlim(-300,200);ax20.set_ylim(0,2.1);
legend(loc="lower left")
title(r"Slice at $t="+str(t_samp_skip[ind])+r"$")
xlabel(r"$x$")
tight_layout();

# <codecell>

len(t_samp_skip)

# <markdowncell>

# Track the tallest peak through the data history to find its velocity

# <codecell>

peakind = np.argmax(spl_ma,0)
peakval = np.array([spl_ma[peakind[n],n] for n in xrange(size(spl_ma,1))])
peak_instvel = np.diff(x_samp[peakind])/(t_betw_plt)
startind = 100
peak_vel, b = np.polyfit(t_samp_skip[startind:], x_samp[peakind[startind:]], 1) 
print peak_vel, b

# <codecell>

f=figure(figsize=(5,8))
ax1 = f.add_subplot(211)
ax1.plot(t_samp_skip[1:], x_samp[peakind[1:]], 'b-', lw=3)
ax1.plot(t_samp_skip[1:], peak_vel*t_samp_skip[1:]+b, 'r-')
title("Peak position over time");xlabel(r"$t$");ylabel(r"$x$")
ax2 = f.add_subplot(212)
ax2.plot(t_samp_skip, peakval)
ax3 = ax2.twinx()
ax3.plot(t_samp_skip[2:]-t_betw_plt/2, peak_instvel[1:], 'r-')
title("Peak height over time");xlabel(r"$t$")
tight_layout();

# <codecell>

sum(peak_instvel)*t_betw_plt

# <codecell>

t_betw_plt

# <codecell>


