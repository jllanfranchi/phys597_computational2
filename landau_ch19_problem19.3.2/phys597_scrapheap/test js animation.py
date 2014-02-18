# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# This is a minimal test of making an animation inine with an IPython notebook.

# <markdowncell>

# Import stuff!

# <codecell>

#%load ~/mypy/bin/stdHeader.py

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
from IPython.display import HTML

from smartFormat import smartFormat
from plotGoodies import plotDefaults

plotDefaults()


# <codecell>

#%load ~/mypy/bin/author.py

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

# <markdowncell>

# Define a simple class that works the same way I intend to implement my code

# <codecell>

class Stuff:
    def __init__(self):
        self.x = np.arange(0,2*np.pi,0.1)
        self.f = plt.figure(1)
        self.ax = self.f.add_subplot(111)
        self.l, = self.ax.plot(np.sin(self.x))
        
    def animate(self,i):
        self.l.set_ydata(np.sin(self.x+i*2*np.pi/40))
        self.f.canvas.draw()
        return self.l,

# <codecell>

s = Stuff()
animation.FuncAnimation(s.f, s.animate, frames=40, interval=30)

# <codecell>

animation.FuncAnimation?

# <codecell>


