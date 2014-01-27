# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Electrostatics

# <markdowncell>

# Testing Gauss-Seidel and related algorithms for solving Laplace's equation, $\nabla^2\phi=0$, with boundary conditions of fixed potential around the edges of a rectangular grid (i.e., metal surrounding the region).

# <headingcell level=2>

# Import modules used

# <codecell>

import numexpr
import numpy as np
import scipy.signal as sig
from matplotlib.pyplot import *
import matplotlib as mpl
from scipy.weave import inline, converters

# <headingcell level=2>

# Testing the basic Gauss-Seidel algorithm

# <markdowncell>

# First test of in-line C code in Python...

# <codecell>

nRows = 250
nCols = 250
size = (nRows, nCols)
grid = np.zeros(size)
#interior = np.zeros(size)
metal = np.zeros(size, dtype=np.bool)
thresh = 0.00001
maxIter = 1000
iterN = np.array([0], dtype=np.int)
#maxDiff = np.array([0.0], dtype=np.double)

grid[0,:] = 1
grid[-1,:] = -2
grid[:,0] = 1
grid[:,-1] = 0

origGrid = grid.copy()

code = """
py::list ret;
double newGval;
double maxDiff = 0.0;
double thisDiff;
int iterN = 0;

for (int iter=0; iter<maxIter; iter++) {
    for (int rowN=1; rowN<nRows-1; rowN++) {
	    for (int colN=1; colN<nCols-1; colN++) {
	        if (metal(rowN,colN) == 0) {
	            newGval = (grid(rowN-1,colN) + grid(rowN+1,colN)
                    + grid(rowN,colN-1) + grid(rowN,colN+1))*0.25;
	            thisDiff = fabs(newGval-grid(rowN,colN));
                if (thisDiff > maxDiff)
                    maxDiff = thisDiff;
	            grid(rowN,colN) = newGval;
	        }
	    }
	}
	iterN++;

    if (maxDiff < thresh)
        break;
    else
        maxDiff = 0.0;
}
ret.append(maxDiff);
ret.append(iterN);
return_val = ret;
"""

out = inline(code,
             ['nRows', 'nCols', 'grid', 'metal', 'thresh', 'maxIter'],
             type_converters=converters.blitz)
print out
bone()
fig1 = figure(1)
imshow(origGrid, interpolation='none');
title(r"Initial conditions")
colorbar()
axis('image')
fig2 = figure(2)
imshow(grid, interpolation='none');
title(r"Solution found")
colorbar()

# <headingcell level=2>

# Object-oriented implementation

# <markdowncell>

# Implement the above, as well as modifications on the basic algorithm, in the form of a nice object-oriented class.

# <codecell>

class Grid2D:
    def __init__(self, nRows, nCols):
        self.sz = (nRows, nCols)
        self.metal = np.zeros(self.sz, dtype=np.bool)
        self.grid = np.zeros(self.sz)
        self.oGrid = self.grid.copy()
        self.nRows = nRows
        self.nCols = nCols
        self.metal[:,0] = True
        self.metal[:,-1] = True
        self.metal[0,:] = True
        self.metal[-1,:] = True
        
    def basicBoundaries(self):
        self.grid += self.metal
        
    def lrtbBoundaries(self, left, right, top, bottom):
        self.grid[:,0] = left
        self.grid[:,-1] = right
        self.grid[0,:] = top
        self.grid[-1,:] = bottom
        
    def gaussSeidelSolver(self, tol=0.001, maxIter=1000):
        self.origGrid = self.grid.copy()
        grid = self.grid
        metal = self.metal
        nCols = self.nCols
        nRows = self.nRows

        code = """
            py::list ret;
            double newGval;
            double maxDiff = 0.0;
            double thisDiff;
            int iterN = 0;
            
            for (int iter=0; iter<maxIter; iter++) {
                for (int rowN=1; rowN<nRows-1; rowN++) {
                    for (int colN=1; colN<nCols-1; colN++) {
                        if (metal(rowN,colN) == 0) {
                            newGval = (grid(rowN-1,colN) + grid(rowN+1,colN)
                                + grid(rowN,colN-1) + grid(rowN,colN+1))*0.25;
                            thisDiff = fabs(newGval-grid(rowN,colN));
                            if (thisDiff > maxDiff)
                                maxDiff = thisDiff;
                            grid(rowN,colN) = newGval;
                        }
                    }
                }
                iterN++;
            
                if (maxDiff < tol)
                    break;
                else
                    maxDiff = 0.0;
            }
            ret.append(maxDiff);
            ret.append(iterN);
            return_val = ret;
            """
        out = inline(code,
                     ['nRows', 'nCols', 'grid', 'metal', 'tol', 'maxIter'],
                     type_converters=converters.blitz)
        print out
        #self.grid = grid

    def checkerboardSolver(self, tol=0.001, maxIter=1000):
        self.origGrid = self.grid.copy()
        grid = self.grid
        metal = self.metal
        nCols = self.nCols
        nRows = self.nRows

        code = """
            py::list ret;
            double newGval;
            double maxDiff = 0.0;
            double thisDiff;
            int iterN = 0;
            
            for (int iter=0; iter<maxIter; iter++) {
                for (int rowN=1; rowN<nRows-1; rowN++) {
                    for (int colN=1; colN<nCols-1; colN++) {
                        if (metal(rowN,colN) == 0) {
                            newGval = (grid(rowN-1,colN) + grid(rowN+1,colN)
                                + grid(rowN,colN-1) + grid(rowN,colN+1))*0.25;
                            thisDiff = fabs(newGval-grid(rowN,colN));
                            if (thisDiff > maxDiff)
                                maxDiff = thisDiff;
                            grid(rowN,colN) = newGval;
                        }
                    }
                }
                iterN++;
            
                if (maxDiff < tol)
                    break;
                else
                    maxDiff = 0.0;
            }
            ret.append(maxDiff);
            ret.append(iterN);
            return_val = ret;
            """
        out = inline(code,
                     ['nRows', 'nCols', 'grid', 'metal', 'tol', 'maxIter'],
                     type_converters=converters.blitz)
        print out


    def jacobiFilterSolver(self, tol=0.001, maxIter=1000):
        protoFilt = np.array([[0,1,0],[1,0,1],[0,1,0]])
        checkEvery = 10
        iterN = 0
        while True:
            newGrid = sig.convolve2d(protoFilt, self.grid)
            iterN += 1
            if iterN >= maxIter:
                break
            elif iterN % checkEvery == 0:
                maxDiff = numexpr.evaluate("abs(newGrid-self.grid)>tol")
                if not maxDiff.any():
                    break
        

    def plotResults(self):
        bone()
        fig1 = figure(1)
        imshow(self.origGrid, interpolation='none');
        colorbar()
        axis('image')
        fig2 = figure(2)
        imshow(self.grid, interpolation='none');
        colorbar()

# <markdowncell>

# Now that the class has been defined, I'll try it out by (roughly) repeating the above experiment.

# <codecell>

g2d = Grid2D(25,25)
g2d.lrtbBoundaries(1,0,1,-2)
g2d.gaussSeidelSolver(tol=0.00001, maxIter=1000)
g2d.plotResults()

# <codecell>


