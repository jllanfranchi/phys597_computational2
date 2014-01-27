#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numexpr
import numpy as np
from matplotlib.pyplot import *
import matplotlib as mpl
from scipy.weave import inline, converters

# <codecell>

nRows = 11
nCols = 11
grid = np.zeros((ny, nx))
metal = np.zeros((ny,nx))
print grid
code = """
for (int i=0; i<ny; i++) {
    for (int j=0; j<ny; j++) {
        grid(i,j) = i;
    }
}
"""
err = inline(code, ['nx', 'ny', 'grid', 'metal'], type_converters=converters.blitz)
print err
print grid

# <codecell>

class Grid2D:
    def __init__(self, nx, ny):
        self.grid = np.zeros((nx,ny))
        self.nx = nx
        self.ny = ny
        
    def solve(self, tol):
        pass

