# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Setup

# <codecell>

from __future__ import division

%pylab inline
%load_ext line_profiler
import time, sys, random
import matplotlib as mpl
import scipy as sp
import scipy.linalg as la
ion()

# <codecell>

def wstdout(x):
    sys.stdout.write(x)
    sys.stdout.flush()

def wstderr(x):
    sys.stderr.write(x)
    sys.stderr.flush()

# <headingcell level=1>

# Overview

# <markdowncell>

# **TODO:** Calculate the correlation length (temporal) of magnetization
# $$\langle M(t)M(0) \rangle = \sum_n S_n(t) \sum_{n'} S_{n'}$$
# and/or the spin dot product:
# $$\langle \vec S(t)\cdot \vec S\rangle = \frac 1 T \sum_{t'}\sum_n S_n(t+t') S_n(t') $$
# but this final 

# <headingcell level=1>

# Histogram class

# <codecell>

class Histogram:
    '''Histogram with dynamically-expanding range and fixed bin width.'''
    def __init__(self, step=1.0, initMin=0.0, initMax=1.0, figNum=1):
        '''Initialize histogram'''
        self.figNum = figNum
        self.step = step
        
        #-- boundaries are integer multiples of the step size;
        #   bin range is [binLeftEdge, binRightEdge)
        self.min  = int(floor(initMin/self.step))*self.step
        self.max  = int(ceil(initMax/self.step))*self.step
        self.len = int(floor((self.max-self.min)/self.step))
        
        self.bins = [0]*self.len
        
    def addVal(self, val):
        binNum = int(floor((val-self.min)/self.step))
        if binNum < 0:
            self.bins = [0]*abs(binNum) + self.bins
            self.bins[0] += 1
            self.min += binNum*self.step
            self.len += abs(binNum)
        elif binNum > self.len-1:
            extendBy = binNum-self.len+1
            self.bins.extend([0]*(extendBy))
            self.bins[binNum] += 1
            self.max += extendBy*self.step
            self.len += extendBy
        else:
            self.bins[binNum] += 1
            
    def stats(self):
        binArray = np.array(self.bins)
        self.nSamp = binArray.sum()
        self.x_vals = np.linspace(self.min, self.max-self.step, self.len)
        self.mean = np.sum((self.x_vals)*binArray) \
                    / self.nSamp + self.step/2
        self.ml = self.x_vals[np.argmax(self.bins)]+self.step/2
        self.stdev = np.sqrt(np.sum(binArray*(self.x_vals+self.step/2)**2)\
                / self.nSamp - self.mean**2)
    
    def reportStats(self):
        self.stats()
        print "nSamp:", self.nSamp, \
                "mean:", self.mean, \
                "stdev:", self.stdev, \
                "ml:", self.ml
        
    def plot(self, figNum=None, **kwargs):
        self.stats()
        if figNum == None:
            figNum = self.figNum
        else:
            self.figNum = figNum
        fig = plt.figure(figNum)
        fig.clf()
        ax = fig.add_subplot(111)
        ax.bar(self.x_vals, self.bins, width=self.step, **kwargs)
        ax.set_xlim(self.min, self.max)

# <markdowncell>

# $\sigma^2 = \langle x^2 \rangle - \langle x \rangle^2$

# <headingcell level=2>

# Test histogram...

# <codecell>

h1 = Histogram(initMin=0.5,initMax=0.5,step=0.01, figNum=1)

# <codecell>

for n in xrange(100000):
    h1.addVal(random.random())
h1.plot()
h1.reportStats()

# <codecell>

h2 = Histogram(step=0.1, figNum=2)

# <codecell>

for n in xrange(1000000):
    h2.addVal(np.random.randn())
h2.plot()
h2.reportStats()

# <headingcell level=1>

# Ising class

# <codecell>

class Lattice:
    def __init__(self, N=10, extH=0, init='rand', temp=1.0,
                 histEnergyStep=None):
        if temp == None:
            temp = 2.26918531421/2
        if histEnergyStep == None:
            histEnergyStep = 2*np.abs(extH) + 4
        self.histEnergyStep = histEnergyStep
        self.generateLattice(N, init)
        self.setExtH(extH)
        self.computeEnergy()
        self.setTemp(temp)
        self.resetRecord(histEnergyStep=histEnergyStep)
        
    def setExtH(self, extH):
        self.extH = extH
        
    def resetRecord(self, histEnergyStep=None):
        if histEnergyStep == None:
            histEnergyStep = self.histEnergyStep
        if not hasattr(self, 'record'):
            self.record = {}
        self.computeEnergy()
        self.record['mag']       = []
        self.record['config']    = []
        self.record['flipCount'] = 0
        self.record['flipAtt']   = 0
        self.record['hist']      = Histogram(step=histEnergyStep,
                                             initMin=self.energy,
                                             initMax=self.energy)

    def setTemp(self, temp):
        self.temp = temp
        
    def generateLattice(self, N, init='rand'):
        self.N = N
        self.N2 = N**2
        self.nextUp = self.N2-self.N
        self.nextDown = self.N2+self.N
        if init == 'rand':
            self.lattice = np.random.randint(low=0, high=2, size=(N,N))*2-1
        elif init in ['up', '+1', '+', 1]:
            self.lattice = np.ones(shape=(N,N),dtype=int)
        elif init in ['down', '-1', '-', -1]:
            self.lattice = -np.ones(shape=(N,N),dtype=int)
        elif init in ['checker', 'checkerboard']:
            self.lattice = np.ones(shape=(N,N),dtype=int)*2-1
            for rowN in xrange(N):
                if rowN % 2 == 0:
                    self.lattice[rowN,::2] *= -1
                else:
                    self.lattice[rowN,1::2] *= -1
        self.ravelLat = self.lattice.ravel()

    def computeEnergy(self):
        self.energy = -self.lattice.sum()*self.extH \
                -np.sum(self.lattice * np.roll(self.lattice, 1, axis=0)) \
                -np.sum(self.lattice * np.roll(self.lattice, 1, axis=1))
        return self.energy
                
    def mhStep(self, N=1, record=True):
        self.record['flipAtt'] += N
        for n in xrange(N):
            #-- Pick a random element of the unravelled array
            ind = np.random.randint(low=0, high=self.N2)
            thisSpin = self.ravelLat[ind]
            s = self.extH
            for i in self.neighborIndices(ind):
                s += self.ravelLat[i]
            flipEnergy = s*thisSpin

            #print ind, thisSpin, [self.ravelLat[i] for i in 
            #                     self.neighborIndices(ind)], flipEnergy

            #-- Flip if energy reduction or (unif rand #) < (boltz factor)
            if (flipEnergy < 0) or \
                    (random.random() < np.exp(-(flipEnergy) \
                                               /(self.temp))):
                #self.ravelLat[ind] *= -1 #= -thisSpin
                self.ravelLat[ind] = -thisSpin
                
                #-- TODO: Should this be indented like this or dedented?
                if record:
                    #-- Record energy in histogram
                    self.energy += 2*flipEnergy
                    self.record['hist'].addVal(self.energy)
                    self.record['flipCount'] += 1

    def neighborIndices(self, ind):
        rowN = ind // self.N
        rowStartInd = rowN * self.N

        return [(ind+self.nextUp)%self.N2,    # up
                (ind+self.nextDown)%self.N2,  # down
                rowStartInd+(ind-1)%self.N,   # left
                rowStartInd+(ind+1)%self.N]   # right
    
    def testNI(self):
        #-- Pick a random element of the unravelled array
        ind = np.random.randint(low=0, high=self.N2)
        thisSpin = self.ravelLat[ind]
        rowN = ind // self.N
        rowStartInd = rowN * self.N
        nInds = self.neighborIndices(ind)
        self.ravelLat[nInds] = -1

    def mhMCStep(self, record=True):
        for n in range(self.N2):
            self.mhStep(record=record)
            
    def plotLattice(self, lattice=None):
        if lattice is None:
            lattice = self.lattice
        f1 = figure(1)
        f1.clf()
        imshow(lattice, interpolation='nearest',
               vmin=-1, vmax=1, cmap=mpl.cm.Greys_r);
        
    def plotHist(self, figNum=2, **kwargs):
        self.record['hist'].plot(figNum=figNum, **kwargs)
    

# <headingcell level=2>

# Test Ising...

# <headingcell level=3>

# Small lattice: 6$\times$6

# <codecell>

ll = Lattice(N=6, init='checker', extH=0, temp=2*1.134592657)
print ll.energy
ll.testNI()
ll.plotLattice()

# <codecell>

ll.mhStep(N=1000)
print ll.energy
print ll.computeEnergy()
ll.plotLattice()
ll.plotHist(linewidth=0.1, color=(0.1,0.5,0.6))
ll.record['hist'].reportStats()

# <headingcell level=3>

# Big lattice: 100$\times$100

# <markdowncell>

# Initialize

# <codecell>

l = Lattice(N=128, init='rand', extH=0, temp=1.134592657/2)
print "energy:", l.energy
l.plotLattice()

# <markdowncell>

# Equilibrate

# <codecell>

%time l.mhStep(record=False, N=1000000)
l.computeEnergy()
print l.energy
l.plotLattice()

# <markdowncell>

# Reset counters

# <codecell>

l.resetRecord()

# <markdowncell>

# Run actual sim at approx. the desired temperature

# <codecell>

l.mhStep(record=True, N=500000)
l.plotLattice()
l.plotHist(linewidth=0.0, color=(0.1,0.5,0.6))
l.record['hist'].reportStats()

# <headingcell level=1>

# Simulated Annealing

# <markdowncell>

# Create a generic class for performing simulated annealing and displaying the results, derived from Lattice class above.

# <codecell>

class Anneal(Lattice):
    def anneal(self):
        pass
    
    def testAnnealing(self, **kwargs):
        self.e_min = -self.N*self.N*(2+self.extH)
        
        self.anneal(**kwargs)
        self.steps = np.cumsum(self.anneal_mcsHist)

        wstdout("Energy = " + str(self.energy) + "\n")
        self.plotLattice()
        self.plotHist(linewidth=0.0, color=(0.1,0.5,0.6))
        
        fig3 = figure(3)
        fig3.clf()
        ax1 = fig3.add_subplot(111)
        
        ax1.plot(self.steps, self.annealSched,
                 'k--', lw=3, label=r"$T$")
        ax1.set_xlabel(r"Step number")
        ax1.set_ylabel(r"$T$", color='k')
        
        ax2 = ax1.twinx()
        ax2.plot(self.steps,
                 (np.array(self.annealEHist)-self.e_min)/self.N**2,
                 'g-', lw=3, label=r"$E$")
        ax2.set_xlabel(r"Step number")
        ax2.set_ylabel(r"$(E-E_\mathrm{min})/N$", color='g')
        [tl.set_color('g') for tl in ax2.get_yticklabels()]

# <headingcell level=2>

# Linear schedule

# <markdowncell>

# This implements a simple linear reduction in temperature.

# <codecell>

class LinAnneal(Anneal):
    def anneal(self, Ti=1000.1, Tf=0.1, slope=-10, mcs=None, nEqlib=None):
        if mcs == None:
            mcs = self.N*self.N
        if nEqlib == None:
            nEqlib = mcs * 100
            
        self.annealSched = np.arange(Ti,Tf+slope,slope)
        self.annealEHist = []
        self.anneal_mcsHist = []
        
        #wstdout("Equilibrating, N="+str(nEqlib)+"...\n")
        #self.mhStep(N=nEqlib, record=False)
        self.resetRecord()
        
        #self.anneal_mcsHist.append(0)
        #self.computeEnergy()
        #self.annealEHist.append(self.energy)
            
        wstdout("Running SA, N="+str(mcs*len(self.annealSched))+"...\n")
        for T in self.annealSched:
            self.setTemp(T)
            self.mhStep(N=mcs, record=True)
            self.anneal_mcsHist.append(mcs)
            self.annealEHist.append(self.energy)

# <headingcell level=3>

# Test linear schedule...

# <codecell>

linSA = LinAnneal(N=50, init='checker', extH=0)
print "energy:", l.energy
linSA.plotLattice()

# <codecell>

linSA.testAnnealing(Ti=10.01, Tf=0.01, 
                    slope=-0.01, nEqlib=int(1e4), mcs=int(1e2))

# <markdowncell>

# Same initialization, but now try a different slope

# <codecell>

linSA.generateLattice(N=50, init='checker')
linSA.resetRecord()
linSA.testAnnealing(Ti=10.01, Tf=0.01, 
                    slope=-0.1, nEqlib=int(1e4), mcs=int(1e3))

# <codecell>


