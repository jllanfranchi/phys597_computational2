#!/usr/bin/env python

from __future__ import division
from __future__ import with_statement

import numpy as np
from pylab import ion
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.weave import inline, converters

import sys
import time
import cPickle as pickle

from smartFormat import smartFormat
from genericUtils import *


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


#-- Turn live-interactive plotting on (makes updated plots appear animated)
ion()

#-- Adjust the font used on the plots
font = {'family' : 'serif', 'weight' : 'normal', 'size'   : 8}
mpl.rc('font', **font)

c_lax_wendroff = """
py::list ret;
double beta2 = beta*beta;
double u_i2;
double u_ip12;
double u_im12;
double this_diff;
double max_ydiff = 0;

int j;
int i = 0;
//u_i2 = u0(i)*u0(i);
//u_ip12 = u0(i+1)*u0(i+1);

for (j=0; j<n_skip; j++) {
    for (i=1; i<m-1; i++) {
        //u_im12 = u_i2;
        //u_i2 = u_ip2;
        u_i2 = u0(i)*u0(i);
        u_im12 = u0(i-1)*u0(i-1);
        u_ip12 = u0(i+1)*u0(i+1);
        
        //-- Lax - Wendroff scheme
        u(i) = u0(i) 
               - 0.25*beta*(u_ip12 - u_im12) 
               + 0.125*beta2 * ( (u0(i+1)+u0(i))*(u_ip12-u_i2)
                                 - (u0(i)+u0(i-1))*(u_i2-u_im12) );
        
        this_diff = fabs(u(i)-u(i-1));
        if (this_diff > max_ydiff)
            max_ydiff = this_diff;
        
        //-- Update "present step" array element with what was just computed as
        //   the next step" value for this array element
        //u0(i) = u(i);
    }
    for (i=1; i<m-1; i++)
        u0(i) = u(i);
}
//-- Enforce boundary conditions
//u(0) = 0;
//u(m-1) = 0;

ret.append(max_ydiff);
return_val = ret;
"""

m = 500
c = 1.0
#dx = 1./m
dx = 2*np.pi/m
dt = dx/10
epsilon = 1.0
beta = epsilon*dt/dx
u = np.zeros((m+1),float)
u0 = np.zeros((m+1), float)
uf = np.zeros((m+1),float)
T_final = 1000
maxN = int(T_final/dt)

print "dt =", dt, ", dx =", dx, ", epsilon =", epsilon, ", beta =", beta

x = np.arange(-(m/2)*dx,(m/2)*dx,dx)

print len(x)

#-- beta = 0.01
#-- epsilon = 0.2
#-- dx = 1e-3
#-- dt = 1e-4
#-- beta = epsilon*dt/dx = 0.02

prob = 1
if prob == 0:
    def finalFun(x, t):
        return -np.exp( - 10.*(x - 1.5 - c*t)**2 ) \
                + np.exp( - 10.*(x + 1.5 + c*t)**2 ) # Exact 

elif prob == 1:
    def finalFun(x, t):
        a0 = -1.0 
        fx = 1 #4*np.pi
        return a0/2*np.sin(fx*x-c*t)+a0/2*np.sin(fx*x+c*t)

u0 = finalFun(x, 0)
u = np.zeros_like(u0)

fig1 = plt.figure(1, figsize=(5,10), dpi=120)
fig1.clf()
ax1 = fig1.add_subplot(211)
ax1.plot(x,u0,'-',color=(.6,.6,.6),lw=6,label="initial cond")
l_ns, = ax1.plot(x,u,'o-',markersize=2, color='b',
                 markerfacecolor=(0.8,0,0,.25),
                 markeredgecolor=(0.8,0,0,.25),
                 lw=0.5,
                 label="numerical soln")
ax1.legend(loc="best")
ax1.set_xlim(-np.pi,np.pi)
#ax1.set_ylim(-1,1)
ax1.set_ylim(min(u0),max(u0))
ax1.set_xlabel(r"Spatial dimension, $x$")
ax1.set_title(r"Spatial wave depiction")

ax2 = fig1.add_subplot(212)
l_d1max, = ax2.plot(0,0, '-o',
                    color='g',
                    markerfacecolor='g',
                    markeredgecolor='g',
                    markersize=3,
                    label=r"Max $|\partial_x u|$",
                    lw=1.0)
ax3 = ax2.twinx()
l_d2max, = ax3.plot(0,0, '-^',
                    color='b',
                    markerfacecolor='b',
                    markeredgecolor='b',
                    markersize=3,
                    label=r"Max $|\partial_{xx} u|$",
                    lw=1.0)
ax2.set_xlabel(r"Time index, $j$")
ax2.set_xlim(0, maxN)
ax2.set_ylim(0,500)
ax3.set_xlim(0, maxN)
ax3.set_ylim(0,500)
ax2.set_title(r"Maximum spatial derivatives at a given time step")
ax2.legend(loc="center left")
ax3.legend(loc="center right")
plt.tight_layout()            

#-- Note: Time steps are indexed with j and spatial coordinates with i.
#   The previous solution is preserved in u0 for use in computing the new
#   solution, which is incrementally stored into the u array.
#
#   Once the computation is complete for the new solution, the u array is
#   copied into u0 for use in the next time step.

max_first_deriv = []
max_second_deriv = []
nskiplist = []
allj = []            
n_skip = 1
j = 0
while True:
    try:
        out = inline(c_lax_wendroff, ['u', 'u0', 'beta', 'm', 'n_skip'],
                     type_converters=converters.blitz)
        j += n_skip
        allj.append(j)
        slope = out[0]/dx
        max_first_deriv.append(slope)
        max_second_deriv.append(np.max(np.abs(np.diff(u0, n=2)))/dx**2)
        #wstdout(str(max_second_deriv[-1]) + " " + str(max_first_deriv[-1])+" ")
        n_skip = min( max(int(1e8*m/100/max_second_deriv[-1]**2), 10), 500)
        n_skip = 2000
        #wstdout(str(n_skip) + ";  ")
        nskiplist.append(n_skip)
        #print out[0]/dx
        l_ns.set_ydata(u)
        l_d1max.set_data(allj, max_first_deriv)
        l_d2max.set_data(allj, max_second_deriv)
        ax1.set_ylim(np.floor(min(np.min(u0),-0.0)*2)/2,
                     np.ceil(max(np.max(u0),0.0)*2)/2)
        ax2.set_ylim(0,np.max(max_first_deriv))
        ax3.set_ylim(0,np.max(max_second_deriv))
        ax2.set_xlim(0,j)
        ax3.set_xlim(0,j)
        plt.draw()
        if j >= maxN or slope > 2000:
            break
    except KeyboardInterrupt:
        raw_input()

#fig2 = plt.figure(2)
#fig2.clf()
#ax = fig2.add_subplot(111)
#ax.plot(nskiplist, 'm-', lw=3)
#ax.set_ylabel("n skip")
#plt.tight_layout()

#plt.show()
