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
        self.grid = np.zeros(self.sz)
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
        return out[0], out[1]

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
                //-- Red squares
                for (int rowN=1; rowN<nRows-1; rowN++) {
                    for (int colN=1; colN<nCols-1; colN++) {
                        if ((rowN%2 == colN%2) && (metal(rowN,colN) == 0)) {
                            newGval = (grid(rowN-1,colN) + grid(rowN+1,colN)
                                + grid(rowN,colN-1) + grid(rowN,colN+1))*0.25;
                            thisDiff = fabs(newGval-grid(rowN,colN));
                            if (thisDiff > maxDiff)
                                maxDiff = thisDiff;
                            grid(rowN,colN) = newGval;
                        }
                    }
                }
                //-- Black squares
                for (int rowN=1; rowN<nRows-1; rowN++) {
                    for (int colN=1; colN<nCols-1; colN++) {
                        if ((rowN%2 != colN%2) && (metal(rowN,colN) == 0)) {
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
        return out[0], out[1]

    def overrelaxationSolver(self, w=1.0, tol=0.001, maxIter=1000):
        '''Note that w=1 is equivalent to the above method!'''
        
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
                //-- Red squares: row and col have same parity
                for (int rowN=1; rowN<nRows-1; rowN++) {
                    for (int colN=1; colN<nCols-1; colN++) {
                        if ((rowN%2 == colN%2) && (metal(rowN,colN) == 0)) {
                            newGval = (grid(rowN-1,colN) + grid(rowN+1,colN)
                                + grid(rowN,colN-1) + grid(rowN,colN+1))*(float)w*0.25
                                + (1-(float)w)*grid(rowN,colN);
                            thisDiff = fabs(newGval-grid(rowN,colN));
                            if (thisDiff > maxDiff)
                                maxDiff = thisDiff;
                            grid(rowN,colN) = newGval;
                        }
                    }
                }
                //-- Black squares: row even, col odd or vice versa
                for (int rowN=1; rowN<nRows-1; rowN++) {
                    for (int colN=1; colN<nCols-1; colN++) {
                        if ((rowN%2 != colN%2) && (metal(rowN,colN) == 0)) {
                            newGval = (grid(rowN-1,colN) + grid(rowN+1,colN)
                                + grid(rowN,colN-1) + grid(rowN,colN+1))*(float)w*0.25
                                + (1-(float)w)*grid(rowN,colN);
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
                     ['w', 'nRows', 'nCols', 'grid', 'metal', 'tol', 'maxIter'],
                     type_converters=converters.blitz)
        return out[0], out[1]

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
        fig1 = figure()
        imshow(self.origGrid, interpolation='none');
        colorbar()
        axis('image')
        fig2 = figure()
        imshow(self.grid, interpolation='none');
        colorbar()

# <markdowncell>

# Now that the class has been defined, I'll try it out by (roughly) repeating the above experiment using the same method (basic Gauss-Seidel relaxation).

# <codecell>

g2d = Grid2D(25,25)
g2d.lrtbBoundaries(1,0,1,-2)
maxDiff, iterN = g2d.gaussSeidelSolver(tol=1e-5, maxIter=1000)
print "max diff:", maxDiff
print "number of iterations:", iterN
g2d.plotResults()

# <markdowncell>

# Now test the checkerboard method of updating.

# <codecell>

g2d = Grid2D(25,25)
g2d.lrtbBoundaries(1,0,1,-2)
maxDiff, iterN = g2d.checkerboardSolver(tol=1e-5, maxIter=1000)
print "max diff:", maxDiff
print "number of iterations:", iterN
g2d.plotResults()

# <markdowncell>

# So this single example turns out to converge in 2/3 the number of iterations as the first approach.
# 
# * **TODO**: plot convergence of each over a range of parameters.
# * **TODO**: compare actual results from one method to next

# <markdowncell>

# Now test the overrelaxation scheme, but using $w=1$ such that the above results *should* be replicated, to verify that basic behavior is OK.

# <codecell>

g2d = Grid2D(25,25)
g2d.lrtbBoundaries(1,0,1,-2)
maxDiff, iterN = g2d.overrelaxationSolver(w=1.9, tol=1e-5, maxIter=1000)
print "max diff:", maxDiff
print "number of iterations:", iterN
g2d.plotResults()

# <markdowncell>

# Okay, so now test for convergence rate for $1<w\le2$, for the same boundary conditions as used above.

# <codecell>

tol = 1e-5
wList = np.arange(1,2.0,0.005)
bcs = [(1,0,1,-2), (1,1,1,-2), (1,1,1,-200), (1,1,1,1)]
gridLen = 25
grids = []

f = figure()
ax = f.add_subplot(111)
for bc in bcs:
    iterations = []
    maxDiffs = []
    for w in wList:
        g2d = Grid2D(gridLen,gridLen)
        g2d.lrtbBoundaries(*bc)
        maxDiff, iterN = g2d.overrelaxationSolver(w=w, tol=tol, maxIter=1000)
        iterations.append(iterN)
        maxDiffs.append(maxDiff)
        grids.append(g2d)
    ax.semilogy(wList, iterations, markersize=3, label=str(bc))
ax.set_xticks(np.linspace(1,2,11))
ax.grid(b=True, which='both')
ax.set_xlabel(r"$w$")
ax.set_ylabel(r"Iterations to converge to " + str(tol))
ax.set_xticks(np.linspace(1,2,11))
ax.set_title(str(gridLen) + " row by " + str(gridLen) + " column grid")
ax.legend(loc='best');

# <codecell>

tol = 1e-5
wList = np.arange(1,2.0,0.01)
bcs = [(1,0,1,-2), (1,1,1,-2), (1,1,1,-200), (1,1,1,1)]
gridLen = 50
grids = []

f = figure()
ax = f.add_subplot(111)
for bc in bcs:
    iterations = []
    maxDiffs = []
    for w in wList:
        g2d = Grid2D(gridLen,gridLen)
        g2d.lrtbBoundaries(*bc)
        maxDiff, iterN = g2d.overrelaxationSolver(w=w, tol=tol, maxIter=10000)
        iterations.append(iterN)
        maxDiffs.append(maxDiff)
    grids.append(g2d)
    ax.semilogy(wList, iterations, markersize=3, label=str(bc));
ax.set_xticks(np.linspace(1,2,11))
ax.grid(b=True, which='both')
ax.set_xlabel(r"$w$")
ax.set_ylabel(r"Iterations to converge to " + str(tol))
ax.set_xticks(np.linspace(1,2,11))
ax.set_title(str(gridLen) + " row by " + str(gridLen) + " column grid")
ax.legend(loc='best');

# <markdowncell>

# The results for different boundary conditions causes the total number of iterations needed for convergence to increase or decrease across the board, and the optimal $w$ value shifts lef-and-right for fundamtally different boundary conditions.
# 
# Increasing the grid size caused convergence to take longer *and* for the minima to shift to higher $w$ values.

# <markdowncell>

# Visualizing the results of the different boundary conditions...

# <codecell>

for (g,bc) in zip(grids, bcs):
    g.plotResults();
    title(str(bc));

# <markdowncell>

# Note that the last result above (as well as all other results, but the effect might be most misleading here) has a colorscale which is scaled to the range of values present on the grid, so while it looks pretty non-uniform, note that the values range from just under 1 to just over 1, where the "correct" answer is 1 everywhere. Pretty close, despite its appearance.

# <markdowncell>

# Clearly it would be helpful to be able to easily iterate over various boundary conditions and find a range of "optimal" $w$ values for those initial conditions (methodize the procedure above), but that won't be implemented this time around.
# 
# Suffice it to say that setting $w$ larger than 1 and less than 1.9 seems to be a save way to speed up the algorithm. There are surely some pathalogical cases that might be slower especially the larger $w$ gets (it's like you're adding energy into the system, so it can go out of control; for $w=1$ there is no energy added to the system).

