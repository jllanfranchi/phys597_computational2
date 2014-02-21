# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Problem 19.5.3: KdV Soliton Simulation

# <codecell>

__author__ = "J.L. Lanfranchi"
__email__ = "jll1062@phys.psu.edu"
__copyright__ = "Copyright 2014 J.L. Lanfranchi"
__credits__ = ["J.L. Lanfranchi"]
__license__ = """Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

# <codecell>

#!/usr/bin/env python

from __future__ import division
from __future__ import with_statement

import numpy as np
from pylab import ion
import matplotlib as mpl
import matplotlib.pylab as p
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.optimize import curve_fit
from scipy.weave import inline, converters
import scipy.signal as sig
import mpmath as mp
import lmfit
from IPython.display import display as D
from JSAnimation import IPython_display, HTMLWriter

import sys
import time
import cPickle as pickle
import copy

from smartFormat import smartFormat
from plotGoodies import plotDefaults, generateColorCycle

plotDefaults()
plt.ion()

BASE_OBJECTS = %who_ls

# <markdowncell>

# Note that the final couple of imports are of custom modules I've written; code for these can be found in my GitHub repo at
# 
# https://github.com/jllanfranchi/pygeneric

# <headingcell level=3>

# Load previous results

# <codecell>

DATA_LOADED = False
try:
    with open("results.pk", 'rb') as dataFile:
        SAVE_VARS = pickle.load(dataFile)
    for key in SAVE_VARS.keys():
        setattr(__builtin__, key, SAVE_VARS[key])
    %xdel SAVE_VARS
    DATA_LOADED = True
except:
    print "not loaded!"

# <headingcell level=2>

# User-set simulation parameters

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
# 
# Note that the multi-precision math module, mpmath, is called in defining the tanh to avoid the exponential overflowing mere mortal floating-point data types.

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
x_skip = 1

t_skip = int(t_betw_plt/dt)
t_betw_plt = t_skip*dt
mt  = int(max_t/t_betw_plt)
t_samp_skip = np.arange(0, mt+1)*t_betw_plt
u   = np.zeros( (num_x, 3), float)
spl = np.zeros( (num_x, 1+int(max_t/dt/t_skip)), float)

# #-- 2D grids with X and Y coordinates
# X, Y = p.meshgrid(x, y)

# #-- Amplitude: 3c/\epsilon; temporal "frequency": sqrt(c/(4\mu))
# c = 4*mu

# <markdowncell>

# Put initial data into u array, which is used to store data history necessary to perform the integration

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
print "Truncation error:", trunc_error_bigo
print "Stability:", np.max(np.abs(stab_cond)), "(must be < 1)"

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
# 
# Note that there are various ways of integrating C (and other languages) into Python. The method I use below just seemed like the simplest to get working (for me) yet offered most of the performance advantages. There are also ways (**cython**) to add nominal markup to Python code to declare some types, and then the Python code is compiled into a form that's about as fast as C.
# 
# More info about what I used, **scipy.weave.inline**:
# http://docs.scipy.org/doc/scipy/reference/generated/scipy.weave.inline.html

# <codecell>

j_start = 1
j_end = max_t_idx
inline(c_integrator,
       ['u', 'spl', 'dt', 'dx', 'fac', 'epsilon', 'j_start',
        'j_end', 'num_x', 't_skip'],
       type_converters=converters.blitz)

# <headingcell level=3>

# Save results to disk

# <codecell>

SAVE_VARS = {'spl':spl, 
    'dx':dx,'dt':dt,'mu':mu,'epsilon':epsilon,'fac':fac,'max_t':max_t,'x_range0':x_range0,
    't_betw_plt':t_betw_plt,
    't_samp':t_samp,'max_t_idx':max_t_idx,'max_t':max_t,'x_samp':x_samp,'num_x':num_x,
    'max_x_idx':max_x_idx,'x_range':x_range,'x_skip':x_skip,'t_skip':t_skip,
    't_betw_plt':t_betw_plt,'mt':mt,'t_samp_skip':t_samp_skip,'u':u,
    'trunc_error_bigo':trunc_error_bigo,'stab_cond':stab_cond}
with open("results.pk", 'wb') as dataFile:
    pickle.dump(SAVE_VARS, dataFile, -1)

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
#spl_lin_norm = (spl_ma-spl_ma.min())/(spl_ma.max()-spl_ma.min())

#-- Normalized version of the array, range from 0-1 (log scale)
#spl_log_norm = np.log10((spl_ma-spl_ma.min())/(spl_ma.max()-spl_ma.min())*(10-1e0)+1e0)

#-- "Derivative" array to examine numerical instabilities. Range from 0-1 (linear)
#spl_diff = np.diff(spl_ma, axis=0)
#spl_diff = (spl_diff-spl_diff.min())/(spl_diff.max()-spl_diff.min())

# <codecell>

__builtin__.

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

# <headingcell level=2>

# Analyze soliton-like behavior in the data

# <headingcell level=3>

# Find peaks

# <headingcell level=4>

# Simplistic peak finding

# <markdowncell>

# Locate peaks, crudely, in the data

# <codecell>

pk_indices = []
for idx in xrange(0, size(spl_ma,1)):
    indices = sig.argrelmax(spl_ma[:,idx], order=5, mode='clip')[0]
    indices.sort()
    pk_indices.append(indices)

# <markdowncell>

# Look at the results of this for one time slice

# <codecell>

ind = len(t_samp_skip)-1
pk_x = x_samp[pk_indices[ind]]
pk_amp = spl_ma[pk_indices[ind],ind]
f20 = figure(20, figsize=(7,4))
f20.clf()
ax20 = f20.add_subplot(111)
ax20.plot(x_samp, spl_ma[:,ind], 'b-', label="Data")
ax20.plot(pk_x, pk_amp, 'o',
          markersize=2, markeredgecolor=(1,0,0,0.75),
          markeredgewidth=1,
          markerfacecolor='none', label="Identified peaks, argrelmax")
ax20.set_xlim(-400,400);ax20.set_ylim(0,2.1);
legend(loc="lower left", frameon=False)
title(r"Slice at $t="+str(t_samp_skip[ind])+r"$")
xlabel(r"$x$")
tight_layout();

# <markdowncell>

# This obviously picks out far more peaks than are relevant to the solitons here.

# <headingcell level=4>

# Filter peaks by prominence

# <markdowncell>

# My strategy to fix this is to filter peaks by their prominence: If a peak's prominence is above a given threshold, then record it. Otherwise, ignore it.
# 
# Note that this will fail for two closely-spaced peaks (hence that have little prominence relative to on another) at the top of a mountain.

# <codecell>

prom_thresh = 0.075
good_pk_ind = []
good_pk_amp = []
#-- Run through all time slices
for idx in xrange(0, size(spl_ma,1)):
    good_pk_ind.append([])
    good_pk_amp.append([])
    
    if len(pk_indices[idx]) == 0:
        continue
    
    amp = spl_ma[:,idx]
    
    #-- Loop over each peak located at this time slice
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

# Plot data and the peaks that were found (for one timeslice) to check the above algorithm is working reasonably

# <codecell>

ind = len(t_samp_skip)-1
good_pk_x = x_samp[good_pk_ind[ind]]
f20 = figure(20, figsize=(7,4))
f20.clf()
ax20 = f20.add_subplot(111)
ax20.plot(x_samp, spl_ma[:,ind], 'b-', label="Data")
ax20.plot(good_pk_x, good_pk_amp[ind], 'o',
          markersize=3, markeredgecolor=(1,0,0,0.75),
          markeredgewidth=1,
          markerfacecolor='none', label="Identified peaks, prominence filt")
ax20.set_xlim(-300,200);ax20.set_ylim(0,2.1);
legend(loc="lower left", frameon=False)
title(r"Slice at $t="+str(t_samp_skip[ind])+r"$")
xlabel(r"$x$")
tight_layout();

# <markdowncell>

# Looks okay, at least for this time slice!

# <headingcell level=3>

# Track peaks as they move

# <markdowncell>

# Now try to track the peaks as they move through time

# <codecell>

#-- NOTE: algo assumes tracks are created but *not destroyed*!
track_xinds = []
track_tinds = []
track_xs = []
track_ts = []

for t_idx in xrange(0, size(spl_ma, 1)):
    time_now = t_samp_skip[t_idx]
    #-- Sort the peaks in descending-amplitude order
    pk_indices = np.array(good_pk_ind[t_idx])
    x_pk = x_samp[list(pk_indices)]
    pk_sortind = pk_indices[list(np.argsort(x_pk)[::-1])]
    unpaired_pk_ind = list(pk_sortind.copy())
    
    #-- Loop through each of the previously-ID'd tracks and find
    #   the peak in the current time slice closest to the track's
    #   last x-value
    for last_pk_n in xrange(0,len(track_xs)):
        if len(unpaired_pk_ind) == 0:
            break
        x_pk = x_samp[unpaired_pk_ind]
        last_pk_x = track_xs[last_pk_n][-1]
        dist = np.abs(x_pk - last_pk_x)
        
        #-- If same dist, argmin returns first match, which will
        #   correspond to the peak with highest amplitude
        closest_ind = unpaired_pk_ind[np.argmin(dist)]

        #-- Record this peak in its track
        track_xinds[last_pk_n].append(closest_ind)
        track_xs[last_pk_n].append(x_samp[closest_ind])
        track_tinds[last_pk_n].append(t_idx)
        track_ts[last_pk_n].append(time_now)
        
        #-- Record that this peak has found its family
        unpaired_pk_ind.remove(closest_ind)
    
    #-- Create new tracks for any remaining unpaired indices
    for pk_ind in unpaired_pk_ind:
        track_xinds.append([pk_ind])
        track_xs.append([x_samp[pk_ind]])
        track_tinds.append([t_idx])
        track_ts.append([time_now])

# <markdowncell>

# Plot the tracks individually and on top of the colormapped image to see if they look reasonable

# <codecell>

generateColorCycle(cmap=p.cm.spectral, n_colors=len(track_tinds));

# <codecell>

f = figure(12)
f.clf()
ax = f.add_subplot(111)
for (t,x) in zip(track_tinds, track_xinds):
    ax.plot(x,t,lw=1,alpha=1)
p.xlabel(r"Position, $x$")
p.ylabel(r"Time, $t$");

# <codecell>

cmap = p.cm.ocean
cmap.set_bad(color='k',alpha=1)

f2=p.figure(13, figsize=(9,9), dpi=30)
f2.clf()
ax2=f2.add_subplot(111)
ax2.imshow(spl_ma.T, interpolation='bicubic', cmap=p.cm.ocean,
           vmin=0, vmax=2, origin='lower')

for (t,x) in zip(track_tinds, track_xinds):
    ax2.plot(x,t,lw=1,color='r',alpha=1)
p.xlabel(r"Position, $x$")
p.ylabel(r"Time, $t$")
p.axis('image')
p.xticks([])
p.yticks([])
ax2.set_xlim(3700,4300)
title(r"Detail, central region of above plot; peak tracks added in red")
#p.axis((0,numpy.max(x),0,numpy.max(y)))
p.tight_layout()
p.show()
f2.savefig('output3.png', dpi=600)

# <markdowncell>

# This looks pretty good, but not as smooth as I'd expect. It may be due to the fact that the recorded data comes only once per second, while the actual simulation runs at 1000$\times$ that rate. There might also be quantization issues, so a peak can hop back and forth between samples and a simple "max" won't know that the actual max should be somewhere between locations of the $x$-discretization.
# 
# --------
# **TODO**: Note that I don't store all of that data because it was causing memory overflow. One solution would be to extract the peak information *during* the simulation, storing only the peak track info or other estimated parameters, while discarding the rest of the data that is generated between the 1 s intervals. Another solution is to write the data to disk during the simulation, and then post-process it one chunk at a time.
# 
# ----------

# <headingcell level=3>

# Perform fit: Fit ideal soliton solutions near each peak location

# <markdowncell>

# I am performing this step next as I hope it'll refine the peak locations and provide some useful and maximally information-preserving (even providing *more* information than what I have to this point) smoothing for the tracks seen in the above plot.
# 
# The other alternatives, such as moving averages or low-pass filtering, will all *remove* information, so I want to use those only as a last resort.
# 
# Note that a simple linear superposition of individual solitons is *not* how it really works because of the nonlinearity of the diffeq's. But that's where I'm going to start, albeit even more crudely than this implies.
# 
# All I'm going to do right now is fit shapes to the data, without removing a fitted shape after it's been used. That might be a next step (although that'd still be using the linear assumption). The better approach might involve non-linear independent component analysis (ICA) of some kind, but even these might be ill-suited to the KdV mixing of solitons since the nonlinear combination of signals must be time-aware. See info on ICA at:
# https://en.wikipedia.org/wiki/Independent_component_analysis
# and refer to the paper Jorge (Zañudo) posted as far as how signals mix in KdV-land.
# 
# ---

# <markdowncell>

# Note that a simple way to do curve fitting is to use **scipy.optimize.curve_fit**. However, for including arbitrary constraints (and easily changing, adding, and removing them) and making fitting a bit more user-friendly, I have been very happy with **lmfit**; see documentation at **http://lmfit.github.io/lmfit-py**

# <headingcell level=4>

# Define model(s) to use for fitting

# <markdowncell>

# Prototype function(s) that can be fit to the data. It needn't be the case, but I set
# 
# + **`params`** as the first argument, containing the **lmfit** parameter object with parameters that are possible for the fitting routine to optimize
# + **`x`**, the $x$-sample locations, as the second argument,
# 
# in keeping with how **lmfit** requires a residual function to be defined. 

# <headingcell level=5>

# Soliton

# <codecell>

def soliton(params, x):
    c = params['c'].value
    eps = params['eps'].value
    xi0 = params['xi0'].value
    """Inputs:
    x   : array of x sample locations
    c   : wave speed
    xi0 : initial phase of soliton"""
    return (c/2)/(np.cosh((np.sqrt(eps)/2)*(x-xi0)))**2

# <headingcell level=4>

# Define error (residual) function

# <codecell>

def residual(params, x, data):
    model_y = soliton(params, x)
    err = (model_y - data)
    return err

# <headingcell level=4>

# Do the fit!

# <markdowncell>

# Test on a single time slice first, to make sure the fitting works okay.

# <codecell>

t_ind = len(t_samp_skip)-1
pk_x_ind = good_pk_ind[t_ind][-1]
pk_x = x_samp[pk_x_ind]

y = spl_ma[:,t_ind]
w = np.zeros_like(y)
w[pk_x_ind-40:pk_x_ind+40] = 1
y = y*w
#y = y * np.exp(-(x_samp-pk_x)**2/15.)

# <markdowncell>

# Define parameters for fit

# <codecell>

params = lmfit.Parameters()
params.add('c',
           value=1, min=0, max=1e2, vary=True)
params.add('eps',
           value=3, min=1, max=10, vary=True)
params.add('xi0',
           value=pk_x, min=np.min(x_samp), max=np.max(x_samp), vary=True)

# <markdowncell>

# Instantiate minimizer object. Note that a callback function can be specified by passing it with the `iter_cb` argument. This is handy if you want to check on the progress of the fit.

# <codecell>

minObj = lmfit.Minimizer(residual, params,
                         fcn_args=None, fcn_kws={'x':x_samp,'data':spl_ma[:,ind]},
                         iter_cb=None, scale_covar=True)

# <codecell>

out = minObj.leastsq(xtol=1e-67, ftol=1e-67, maxfev=100000)
lmfit.report_fit(params)

# <codecell>

fit_y = soliton(params, x_samp)

f20 = figure(20, figsize=(8,6))
f20.clf()
ax20 = f20.add_subplot(111)
ax20.plot(x_samp, spl_ma[:,ind], '-', color=(0.6,0.6,0.6), lw=5, label="Orig. data")
ax20.plot(x_samp, y, 'c-', lw=5, label="Cut-off data")
ax20.plot(x_samp, fit_y, 'k--', alpha=1, lw=2., label=r"Pseudo-soliton fit")
ax20.set_xlim(pk_x-10,pk_x+10);ax20.set_ylim(-.1,2.1);
legend(loc="upper right", frameon=False)
title(r"Slice at $t="+str(t_samp_skip[ind])+r"$")
xlabel(r"$x$")
tight_layout();

# <codecell>

#
for (t,x) in zip(track_tinds, track_xinds):
    

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

f=figure(30, figsize=(5,8))
f.clf()
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

# <markdowncell>

# This last plot looks terrible (the slope of the tracked peak). But it looks a lot like a pulse position modulated signal where the density of the pulses appears to track the amplitude, so some low-pass filtering on the data is sure to give a more readily interpreted result.

