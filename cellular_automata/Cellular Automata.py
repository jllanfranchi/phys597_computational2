# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Cellular Automata

# <headingcell level=2>

# Initialization

# <codecell>

from __future__ import division

import numpy as np
import bitarray as ba

import matplotlib as mpl

from plotGoodies import generateColorCycle
#generateColorCycle(cmap=mpl.cm.binary, n_colors=1, set_it=True)

# <codecell>

%pylab inline

# <headingcell level=2>

# Create CA class

# <codecell>

class CellularAutomata:
    def __init__(self, fieldWidth, initCond, ruleNum, ruleWidth, tSteps):
        self.constructField(fieldWidth=fieldWidth, tSteps=tSteps)
        self.IC(width=fieldWidth, initCond=initCond)
        self.setRule(ruleNum=ruleNum, ruleWidth=ruleWidth)
    
    def constructField(self, fieldWidth, tSteps):
        self.fieldWidth = fieldWidth
        self.tSteps = tSteps
        self.field = np.zeros((tSteps,fieldWidth))

    def IC(self, width=100, initCond='rand'):
        self.initCond = initCond
        if initCond is 'rand':
            self.initialArray = np.random.randint(0,2,width).astype(np.bool)
        elif initCond is 'zeros':
            self.initialArray = np.zeros(shape=(width,),dtype=np.bool)
        elif initCond is 'ones':
            self.initialArray = np.ones(shape=(width,),dtype=np.bool)
        elif initCond is 'oneone':
            self.initialArray = np.zeros(shape=(width,),dtype=np.bool)
            self.initialArray[np.round(width/2)] = True
        self.field[0,:] = self.initialArray
        
    def setRule(self, ruleNum, ruleWidth):
        self.ruleWidth = ruleWidth
        self.ruleHalfWidth = np.ceil(ruleWidth/2)
        self.ruleNum = ruleNum
        self.bitValues = np.array([2**n for n in xrange(ruleWidth)])
        self.bitStr = binary_repr(ruleNum, width=2**self.ruleWidth)
        self.bitArr = ba.bitarray(self.bitStr)
        self.ruleLUT = np.array(self.bitArr.tolist()[::-1], dtype=np.bool)
        print "Rule", ruleNum, ":", self.bitArr.to01()
                
    def iterate(self):
        for stepN in xrange(1,self.tSteps):
            self.field[stepN,:] = self.ruleLUT[
                np.correlate(self.field[stepN-1],
                             self.bitValues,mode='same').astype(np.int)]
        
    def display(self, figsize=None, interpolation='nearest'):
        #if figsize is None:
        #    figsize = ()
        self.fig = figure(figsize=figsize)
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self.field, interpolation=interpolation,
                  cmap=mpl.cm.Reds,
                  aspect='equal', extent=None,
                  origin='upper', alpha=0.8)
        self.ax.set_xticks(np.arange(-0.5,self.fieldWidth+0.5))
        self.ax.set_yticks(np.arange(-0.5,self.tSteps+0.5))
        #self.ax.grid(b=True,which='major',
        #             alpha=1, linestyle='-', color='w', linewidth=2)
        self.ax.set_axes('image')
        self.ax.xaxis.set_ticklabels([])
        self.ax.yaxis.set_ticklabels([])
        axis('off')
        tight_layout()

# <headingcell level=2>

# Try out some width-3 rules & IC

# <codecell>

ca = CellularAutomata(fieldWidth=200, tSteps=100,
                      initCond='oneone',
                      ruleNum=90, ruleWidth=3)
ca.iterate()
ca.display()

# <codecell>

ca = CellularAutomata(fieldWidth=200, tSteps=100,
                      initCond='rand',
                      ruleNum=90, ruleWidth=3)
ca.iterate()
ca.display()

# <codecell>

ca = CellularAutomata(fieldWidth=200, tSteps=100,
                      initCond='oneone',
                      ruleNum=110, ruleWidth=3)
ca.iterate()
ca.display()

# <codecell>

ca = CellularAutomata(fieldWidth=200, tSteps=100,
                      initCond='rand',
                      ruleNum=110, ruleWidth=3)
ca.iterate()
ca.display()

# <codecell>

ca = CellularAutomata(fieldWidth=200, tSteps=100,
                      initCond='oneone',
                      ruleNum=150, ruleWidth=3)
ca.iterate()
ca.display()

# <codecell>

ca = CellularAutomata(fieldWidth=200, tSteps=100,
                      initCond='rand',
                      ruleNum=150, ruleWidth=3)
ca.iterate()
ca.display()

# <headingcell level=2>

# Try out width-5 rules

# <codecell>

ca = CellularAutomata(fieldWidth=200, tSteps=100,
                      initCond='oneone',
                      ruleNum=7232300, ruleWidth=5)
ca.iterate()
ca.display()

# <codecell>

ca = CellularAutomata(fieldWidth=200, tSteps=100,
                      initCond='rand',
                      ruleNum=7232300, ruleWidth=5)
ca.iterate()
ca.display(interpolation='nearest')

# <codecell>

ca = CellularAutomata(fieldWidth=200, tSteps=100,
                      initCond='oneone',
                      ruleNum=337232386, ruleWidth=5)
ca.iterate()
ca.display(interpolation='bilinear')

