# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <rawcell>

# #!/usr/bin/env python

# <codecell>

from __future__ import division
from __future__ import with_statement

import numpy as np
from pylab import ion
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.optimize import curve_fit
from scipy.weave import inline, converters

import sys
import time
import cPickle as pickle

from JSAnimation import IPython_display, HTMLWriter

from smartFormat import smartFormat
from plotGoodies import plotDefaults

plotDefaults()

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

#-- Turn live-interactive plotting on (makes updated plots appear animated)
ion()

#-- Adjust the font used on the plots
font = {'family' : 'serif', 'weight' : 'normal', 'size'   : 8}
#mpl.rcParams('font', **font)

# <codecell>

class WaxWendroff:
    def __init__(self):
        self.c_lax_wendroff = """
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
        u0(i) = u(i);
     }
}
//for (i=1; i<m-1; i++)
//    u0(i) = u(i);
//-- Enforce boundary conditions
//u(0) = 0;
//u(m-1) = 0;

ret.append(max_ydiff);
return_val = ret;
"""

        self.m = 1000
        self.c = 1.0
        #dx = 1./m
        self.dx = 2*np.pi/self.m
        self.dt = self.dx/10
        self.epsilon = 1.0
        self.beta = self.epsilon*self.dt/self.dx
        self.u = np.zeros((self.m+1),float)
        self.u0 = np.zeros((self.m+1), float)
        self.uf = np.zeros((self.m+1),float)
        self.T_final = 100
        self.maxN = int(self.T_final/self.dt)
        
        print "dt =", self.dt, ", dx =", self.dx, \
                ", epsilon =", self.epsilon, ", beta =", self.beta
        
        self.x = np.arange(-(self.m/2)*self.dx,(self.m/2)*self.dx,self.dx)
        
        print len(self.x)
        
        #-- beta = 0.01
        #-- epsilon = 0.2
        #-- dx = 1e-3
        #-- dt = 1e-4
        #-- beta = epsilon*dt/dx = 0.02
        
        self.prob = 1
        if self.prob == 0:
            def finalFun(x, t):
                return -np.exp( - 10.*(x - 1.5 - self.c*t)**2 ) \
                        + np.exp( - 10.*(x + 1.5 + self.c*t)**2 ) # Exact 
        
        elif self.prob == 1:
            def finalFun(x, t):
                a0 = -1.0 
                fx = 1 #4*np.pi
                return a0/2*np.sin(fx*x-self.c*t)+a0/2*np.sin(fx*x+self.c*t)
        
        self.u0 = finalFun(self.x, 0)
        self.u = np.zeros_like(self.u0)
        
        self.fig1 = plt.figure(1, figsize=(5,10), dpi=120)
        self.fig1.clf()
        self.ax1 = self.fig1.add_subplot(211)
        self.ax1.plot(self.x, self.u0, '-',
                      color=(.6,.6,.6), lw=6, label="initial cond")
        self.l_ns, = self.ax1.plot(self.x, self.u, 'o-',
                                  markersize=2,
                                  color='b',
                                  markerfacecolor=(0.8,0,0,.25),
                                  markeredgecolor=(0.8,0,0,.25),
                                  lw=0.5,
                                  label="numerical soln")
        self.ax1.legend(loc="best")
        self.ax1.set_xlim(-np.pi,np.pi)
        self.ax1.set_ylim(-1,1)
        self.ax1.set_xlabel(r"Spatial dimension, $x$")
        self.ax1.set_title(r"Spatial wave depiction")
        
        self.ax2 = self.fig1.add_subplot(212)
        self.l_ms, = self.ax2.plot(0,0, '-o',
                                  color='k',
                                  markerfacecolor='g',
                                  markersize=3,
                                  lw=1.0)
        self.ax2.set_xlabel(r"Time index, $j$")
        #ax2.set_ylabel(r"Maximum spatial slope")
        self.ax2.set_xlim(0, self.maxN)
        self.ax2.set_ylim(0,500)
        self.ax2.set_title(r"Maximum spatial slope at a given time step")
        plt.tight_layout()            
        
        #-- Note: Time steps are indexed with j and spatial coordinates with i.
        #   The previous solution is preserved in u0 for use in computing the
        #   new solution, which is incrementally stored into the u array.
        #
        #   Once the computation is complete for the new solution, the u array
        #   is copied into u0 for use in the next time step.
    
        #def init(self):
        self.l_ns.set_data(self.x, finalFun(self.x,0))
        self.l_ms.set_data(0,0)
        self.maxslopelist = []
        slf.nskiplist = []
        self.allj = []            
        self.n_skip = 1
        self.j = 0
        #return self.l_ns, self.l_ms
    
    def animate(self, ii):
        print "Iteration number, ii:", ii
        out = inline(self.c_lax_wendroff, ['self.u', 'self.u0', 'self.beta',
                                           'self.m', 'self.n_skip'],
                     type_converters=converters.blitz)
        self.j += self.n_skip
        self.allj.append(j)
        self.slope = out[0]/self.dx
        self.maxslopelist.append(self.slope)
        self.n_skip = min( max(int(5e4/self.slope**2), 10), 1000)
        self.n_skip = 100
        self.nskiplist.append(n_skip)
        print out[0]/self.dx
        self.l_ns.set_ydata(self.u)
        self.l_ms.set_xdata(self.allj)
        self.l_ms.set_ydata(self.maxslopelist)
        self.ax2.set_ylim(0,np.max(self.maxslopelist))
        self.ax2.set_xlim(0,self.j)
        self.fig1.canvas.draw()
        #plt.draw()
        #if j >= maxN or slope > 2000:
        #    break
        #return l_ns, l_ms

#fig2 = plt.figure(2)
#fig2.clf()
#ax = fig2.add_subplot(111)
#ax.plot(nskiplist, 'm-', lw=3)
#ax.set_ylabel("n skip")
#plt.tight_layout()
ww = WaxWendroff()
animation.FuncAnimation(ww.fig1, ww.animate, frames=20, blit=True)

# <codecell>

plt.show()

# <codecell>


