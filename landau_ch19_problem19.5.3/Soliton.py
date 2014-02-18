""" From "A SURVEY OF COMPUTATIONAL PHYSICS", Python eBook Version
   by RH Landau, MJ Paez, and CC Bordeianu
   Copyright Princeton University Press, Princeton, 2011; Book  Copyright R Landau, 
   Oregon State Unv, MJ Paez, Univ Antioquia, C Bordeianu, Univ Bucharest, 2011.
   Support by National Science Foundation , Oregon State Univ, Microsoft Corp"""  

# Soliton.py: Solves Korteweg de Vries equation for a soliton.
from __future__ import division
from visual import *
import matplotlib.pylab as p;
from mpl_toolkits.mplot3d import Axes3D ;
import numpy
p.ion()

ds = 0.4  # default: 0.4
dt = 0.1 # default: 0.1
stopTime = 200
max = int(round(stopTime/dt))
mu = 0.1
eps = 0.2
mx = int(131*0.4/ds)
#mx = 131
#mx = 262

t_betw_plt = 1
t_steps_betw_plt = int(t_betw_plt/dt)
t_betw_plt = t_steps_betw_plt*dt
mt = int(stopTime/t_betw_plt)
u   = zeros( (mx, 3), float)
spl = zeros( (mx, 1+int(stopTime/dt/t_steps_betw_plt)), float)
m = 1
x_skip = 1

for  i in range(0, mx):                                   # initial wave
    u[i, 0] = 0.5*(1 -((math.exp(2*(0.2*ds*i-5.))-1)/(math.exp(2*(0.2*ds*i-5.))+1)))
u[0,1] = 1. 
u[0,2] = 1.
u[mx-1,1] = 0. 
u[mx-1,2] = 0.     # End points

for i in range (0, mx, x_skip):
    spl[i, 0] = u[i, 0]    # initial wave 2x step
fac = mu*dt/(ds**3)              
print("Working. Please hold breath and wait while I count to "+str(20))

for  i in range (1, mx-1):                              # First time step
    a1 = eps*dt*(u[i + 1, 0] + u[i, 0] + u[i - 1, 0])/(ds*6.)     
    if i > 1 and  i < mx-2: a2 = u[i+2,0]+2.*u[i-1,0]-2.*u[i+1,0]-u[i-2,0]
    else:  a2 = u[i-1, 0] - u[i+1, 0]
    a3 = u[i+1, 0] - u[i-1, 0] 
    u[i, 1] = u[i, 0] - a1*a3 - fac*a2/3.        



for j in range (1, max+1):                              # next time steps 
    for i in range(1, mx-2):
        a1 = eps*dt*(u[i + 1, 1]  +  u[i, 1]  +  u[i - 1, 1])/(3.*ds)
        if i > 1 and i < mx-2:
            a2 = u[i+2,1] + 2.*u[i-1,1] - 2.*u[i+1,1] - u[i-2,1]
        else:
            a2 = u[i-1, 1] - u[i+1, 1]  
        a3 = u[i+1, 1] - u[i-1, 1] 
        u[i, 2] = u[i,0] - a1*a3 - 2.*fac*a2/3.
    if j%t_steps_betw_plt ==  0:                # plot every 10 "seconds"
        for i in range (1, mx - 2): spl[i, m] = u[i, 2]
        print(m)  
        m = m + 1     
    for k in range(0, mx):                 # recycle array to save memory
        u[k, 0] = u[k, 1]                
        u[k, 1] = u[k, 2] 

x = list(range(0, mx, x_skip))                 # plot every spatial point
y = list(range(0, mt, t_steps_betw_plt))     # plot line every (?) t steps
X, Y = p.meshgrid(x, y)
def functz(spl):                           # Function returns temperature
    z = spl[X, Y]       
    return z
#s = surf(x, y, 20*spl)
#fig  = p.figure(1)                                         # create figure
#fig.clf()
#ax = Axes3D(fig)                                              # plot axes
#ax.plot_surface(X*ds*x_skip, Y*dt*t_steps_betw_plt, spl[X, Y],
#                cmap=p.cm.bone)#  color = 'r')                            # red wireframe
#ax.plot_surface(X*ds*x_skip, Y*dt*t_steps_betw_plt, spl[X, Y],
#                cmap=p.cm.bone)#  color = 'r')                            # red
#ax.set_xlabel('Positon')                                     # label axes
#ax.set_ylabel('Time')
#ax.set_zlabel('Disturbance')

f2=p.figure(2)
f2.clf()
ax2=f2.add_subplot(111)
ax2.imshow((numpy.transpose(spl)),interpolation='bicubic',cmap=p.cm.ocean)
p.xlabel(r"Position index, $x/\Delta x$")
p.ylabel(r"Time index, $t/\Delta t$")
p.axis('normal')
p.axis((0,numpy.max(x),0,numpy.max(y)))
p.tight_layout()
p.show()                                # Show figure, close Python shell
print("That's all folks!") 

