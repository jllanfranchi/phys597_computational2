#!/usr/bin/env python

from __future__ import division
from __future__ import with_statement

import numpy as np
from pylab import ion
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

import sys
import time
import cPickle as pickle

from smartFormat import smartFormat


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


def createChain(nSteps):
    """State is taken to be direction of travel; there are 4 directions, so
    four states: right (0), up (1), left (2), and down (3). The only allowed
    transitions are
        state -> state + 1 (modulo 4)
        state -> state - 1 (modulo 4)
        state -> state
    
    Then, it must be checked that the coordinate isn't already in the chain;
    if it is, then None is returned; otherwise, the algo repeats until a
    chain of nSteps is reached.
    """

    #-- Initialize chain to start at (0,0) and move to (0,1) (i.e., move up)
    chainCoords = [(0,0)]
    chainCoords = [(0,0)]
    coord = (0,1)
    chainCoords.append(coord)
    state = 1
    length = 1
    while True:
        randVal = np.random.randint(low=-1, high=2)
        state = (state + randVal) % 4
        
        if state is 0:
            coord = (coord[0]+1, coord[1])
        elif state is 1:
            coord = (coord[0], coord[1]+1)
        elif state is 2:
            coord = (coord[0]-1, coord[1])
        elif state is 3:
            coord = (coord[0], coord[1]-1)
   
        if coord in chainCoords:
            return None

        chainCoords.append(coord)
        length += 1
    
        if length == nSteps:
            return chainCoords


def measureChain(chain):
    """Measures the Euclidean distance from the startpoint to endpoint of
    a chain"""
    return np.sqrt((chain[-1][0] - chain[0][0])**2
                   + (chain[-1][1] - chain[0][1])**2)


formatDic = {'sigFigs': 4, 'demarc': "", 'threeSpacing': False, 'rightSep':""}


def powerLaw(x, power, multFact, offset):
    return multFact*(x**power) + offset


def powerLawLatex(power, multFact=1, offset=0, pcov=None):
    offsetStr = smartFormat(offset, alwaysShowSign=True, **formatDic)
    if not (offsetStr[0] == "+" or offsetStr[0] == "-"):
        offsetStr = "+" + offsetStr
    latex = r"$" + smartFormat(multFact, **formatDic) + \
            r" \cdot N^{" + smartFormat(power, **formatDic) + r"} " + \
            offsetStr + \
            r"$"
    return latex


def exponential(x, expExponent, multFact=1):
    return multFact * np.exp(np.array(x)*expExponent)


def exponentialLatex(expExponent, multFact=1, pcov=None):
    latex = r"$" + smartFormat(multFact, **formatDic) + \
            r"\cdot e^{" + smartFormat(expExponent, **formatDic) + \
            r"\cdot N}$"
    return latex


def expPower(x, expExponent, powerLawExp, multFact):
    x = np.array(x)
    return multFact * np.exp(x*expExponent) * x**powerLawExp


def expPowerLatex(expExponent, powerLawExp, multFact, pcov=None):
    latex = r"$" + smartFormat(multFact, **formatDic) + \
            r"\cdot e^{" + smartFormat(expExponent, **formatDic) + \
            r"\cdot N}\cdot N^{" + smartFormat(powerLawExp, **formatDic) + \
            r"}$"
    return latex


class SimulationData:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Simulation:
    def __init__(self):
        self.sd = SimulationData()
        self.sd.simulationCompleted = False
        self.sd.postprocCompleted = False
        self.stateFilename = "p7x28_state.pk"

    def saveState(self, filename=None):
        if filename == None:
            filename = self.stateFilename
        with open(filename, 'wb') as stateFile:
            pickle.dump(self.sd, stateFile, -1)

    def loadState(self, filename=None):
        if filename == None:
            filename = self.stateFilename
        with open(filename, 'rb') as stateFile:
            self.sd = pickle.load(stateFile)
 
    def runSimulation(self, targetSuccesses=10, stepsRange=(4,50),
                      plotting=False):
        #-- Reset state variables for a new simulation run
        self.sd.simulationCompleted = False
        self.sd.postprocCompleted = False

        timeLastSaved = time.time()
    
        self.sd.targetSuccesses = targetSuccesses
        self.sd.stepsInChains = range(stepsRange[0],stepsRange[1])

        self.sd.allChainFinalCoords = []
        self.sd.allMeanChainFinalCoords = []
        self.sd.meanChainFinalCoords = []
        self.sd.chainSquareLengthAvg = []
        self.sd.successRatio = []
        self.sd.timingAvg = []

        if plotting:
            self.fig1 = plt.figure(1)
            self.fig1.clf()
            self.ax1 = fig1.add_subplot(111)
            line, = ax1.plot([], [], 'ko-', lw=2)
            self.ax1.set_xlim(-20,20)
            self.ax1.set_ylim(-20,20)
            ax1.axis('image')
            plt.draw()

        for stepsThisChain in self.sd.stepsInChains:
            startTime = time.time()
            successfulChains = []
            chainSquareLengths = []
            chainFinalCoords = []
            meanChainFinalCoord = []
            nSuccesses = 0
            trialN = 0
            while nSuccesses < self.sd.targetSuccesses:
                trialN += 1
                chain = createChain(stepsThisChain)
                if chain == None:
                    continue
                successfulChains.append(chain)
                chain = np.array(chain)
                chainSquareLengths.append(measureChain(chain)**2)
                chainFinalCoords.append(chain[-1,:])
                nSuccesses += 1
                if plotting:
                    line.set_data(chain[:,0],chain[:,1])
                    self.ax1.set_xlim(-20,20)
                    self.ax1.set_ylim(-20,20)
                    plt.draw()
                    time.sleep(0.005)
            chainFinalCoords = np.array(chainFinalCoords)

            self.sd.allChainFinalCoords.append(chainFinalCoords)
            self.sd.allMeanChainFinalCoords.append(meanChainFinalCoord)
            self.sd.meanChainFinalCoord = np.mean(chainFinalCoords, 0)
            self.sd.chainSquareLengthAvg.append(np.mean(chainSquareLengths))
            self.sd.successRatio.append(nSuccesses / trialN)
            self.sd.timingAvg.append( (time.time()-startTime)/nSuccesses )

            sys.stdout.write("\nstepsThisChain = " + str(stepsThisChain) + "\n")
            sys.stdout.write("  nSuccesses/nTrials = " + str(nSuccesses) + "/" 
                             + str(trialN) + " = "
                             + str(self.sd.successRatio[-1]) + "\n")
            sys.stdout.write("  time/success = " +
                             str(self.sd.timingAvg[-1]) + "\n")
            sys.stdout.flush()

            if (time.time() - timeLastSaved) > 60*5:
                self.saveState()
                timeLastSaved = time.time()

        self.sd.allMeanChainFinalCoords = \
                np.array(self.sd.allMeanChainFinalCoords)

        #-- TODO: mean of final-position vector (r_N vector)
        #np.sqrt(allMeanChainFinalCoords[:,0]**2+
        #        allMeanChainFinalCoords[:,1]**2)
        self.sd.simulationCompleted = True

        self.saveState()

    def postproc(self):
        """Perform curve fitting to the data"""

        #-- Update state
        self.sd.postprocCompleted = False

        #-- Check that simulation data is present
        if not self.sd.simulationCompleted:
            raise Exception("No simulation run; cannot perform curve fit!")
   
        #-- Same x data is used for *all* the below curve fits
        x = self.sd.stepsInChains

        #============================================================
        # Fit success fraction with const * exponential * power law
        #============================================================
        y = self.sd.successRatio
        #-- Weight variance by data size to make small data points equally
        #   important to fit to as large data points
        sigma = list(np.array(y))
        p0 = (-0.117, 0.1, 2)
        popt1, pcov1 = curve_fit(f=expPower, xdata=x, ydata=y, sigma=sigma,
                                 p0=p0)
        self.sd.fit1 = expPower(x, *popt1)
        self.sd.fit1eqn = expPowerLatex(*popt1) 
        print popt1, pcov1, "\n"
       
        #============================================================
        # TODO: Fit the final position data
        #============================================================
        #y = (self.sd.chainLengthAvg)
        #sigma = list(np.array(y))
        #popt2, pcov2 = curve_fit(powerLaw, x, y, sigma=sigma)
        #self.sd.fit2 = powerLaw(x, *popt2)
        #self.sd.fit2eqn = powerLawLatex(*popt2) 
        #print popt2, pcov2, "\n"

        #============================================================
        # Fit R_N^2 with const * power-law + const
        #============================================================
        y = self.sd.chainSquareLengthAvg
        #-- Weight variance by data size to make small data points equally
        #   important to fit to as large data points
        sigma = list(np.array(y))
        popt3, pcov3 = curve_fit(f=powerLaw, xdata=x, ydata=y, sigma=sigma)
        self.sd.fit3 = powerLaw(x, *popt3)
        self.sd.fit3eqn = powerLawLatex(*popt3) 
        print popt3, pcov3, "\n"

        #============================================================
        # Exponential fit to wall-clock time (not as good a fit as
        #   exp*power, so this is commented out)
        #============================================================
        #y = (self.sd.timingAvg)
        ##p0 = (0.0985, 0.1, 1.65e-5)
        #p0 = (0.0985, 1)
        #sigma = list(np.array(y))
        #popt4, pcov4 = curve_fit(f=exponential, xdata=x, ydata=y, sigma=sigma,
        #                         p0=p0, )
        #self.sd.fit4 = exponential(x, *popt4)
        #self.sd.fit4eqn = exponentialLatex(*popt4) 
        #print popt4, pcov4, "\n"

        #============================================================
        # Exponential * power-law fit to wall-clock time
        #============================================================
        y = self.sd.timingAvg
        #-- Initial guess
        p0 = (0.129, 0, 2.981e-3)
        #-- Weight variance by data size to make small data points equally
        #   important to fit to as large data points
        sigma = list(np.array(y))
        popt4, pcov4 = curve_fit(f=expPower, xdata=x, ydata=y, sigma=sigma,
                                 p0=p0, )
        self.sd.fit4 = expPower(x, *popt4)
        self.sd.fit4eqn = expPowerLatex(*popt4) 
        print popt4, pcov4, "\n"

        #-- Update state
        self.sd.postprocCompleted = True

    def plotResults(self, savePlot=True):
        """Plot the data and the fit curves"""

        if not self.sd.simulationCompleted:
            raise Exception("No simulation has been run; cannot plot results!")

        if not self.sd.postprocCompleted:
            self.postproc()

        self.fig2 = plt.figure(2, figsize=(7,12), dpi=80)
        self.fig2.clf()
        self.ax21 = self.fig2.add_subplot(311)
        self.ax21.plot(self.sd.stepsInChains, self.sd.successRatio,
                       'bo', label="data", markersize=4)
        self.ax21.plot(self.sd.stepsInChains, self.sd.fit1,
                       'r-', label=self.sd.fit1eqn, linewidth=2, alpha=0.75)
        self.ax21.set_title(
            "Non-intersecting 2D random-walk chains;" +
            " stop condition: " + str(self.sd.targetSuccesses) +
            " successfully-built chains")
        self.ax21.set_ylabel(r"Success fraction $f(N)$")
        self.ax21.set_yscale('log')
        self.ax21.grid(which='major', b=True)
        self.ax21.legend(loc="best", fancybox=True, shadow=True)

        #-- TODO: average of final position plot
        #self.ax22 = fig2.add_subplot(412)
        #self.ax22.plot(self.sd.stepsInChains, self.sd.chainLengthAvg,
        #               'bo', label="data", markersize=4)
        #self.ax22.plot(self.sd.stepsInChains, self.sd.fit2,
        #               'r-', label=self.sd.fit2eqn, linewidth=2, alpha=0.75)
        #self.ax22.set_ylabel(r"$\langle R_N \rangle$")
        ##self.ax22.set_yscale('log')
        #ax22.grid(which='major', b=True)
        #ax22.legend(loc="best", fancybox=True, shadow=True)

        self.ax23 = self.fig2.add_subplot(312)
        self.ax23.plot(self.sd.stepsInChains, self.sd.chainSquareLengthAvg,
                       'bo', label="data", markersize=4)
        self.ax23.plot(self.sd.stepsInChains, self.sd.fit3,
                       'r-', label=self.sd.fit3eqn, linewidth=2, alpha=0.75)
        self.ax23.set_ylabel(r"$\langle R_N^2\rangle$")
        self.ax23.grid(which='major', b=True)
        self.ax23.legend(loc="upper left", fancybox=True, shadow=True)

        self.ax24 = self.fig2.add_subplot(313)
        self.ax24.plot(self.sd.stepsInChains, self.sd.timingAvg,
                       'bo', label="data", markersize=4)
        self.ax24.plot(self.sd.stepsInChains, self.sd.fit4,
                       'r-', label=self.sd.fit4eqn, linewidth=2, alpha=0.75)
        self.ax24.set_xlabel(r"Nmber of steps in walk, $N$")
        self.ax24.set_ylabel("Wall-clock time per successful chain (s)")
        self.ax24.set_yscale('log')
        self.ax24.grid(which='major', b=True)
        self.ax24.legend(loc="upper left", fancybox=True, shadow=True)

        self.fig2.tight_layout()

        if savePlot:
            self.fig2.savefig("2014-01-14_problem7x28_plots.pdf")
            self.fig2.savefig("2014-01-14_problem7x28_plots.png", dpi=120)

        plt.show()


if __name__ == "__main__":
    #-- Instantiate the Simulation object
    sim = Simulation()

    #-- Try to load the sim data from any previous run; if no data saved
    #   to disk in the default location, run a new simulation
    try:
        sim.loadState()
    except Exception as e:
        print "Error({0}: {1}".format(e.errno, e.strerror)
        sim.runSimulation(targetSuccesses=500, stepsRange=(4,101))

    #-- *Always* perform post-processing and plotting (allows easy modification
    #   of the postprocessing (curve fitting) and plotting routines
    #   without needing to re-run the simulation, which can take hours)
    sim.postproc()
    sim.plotResults()

