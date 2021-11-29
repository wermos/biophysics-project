from math import cos, pi, sqrt, sin
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sympy
from sympy import symbols, diff
import tqdm 
#enter number of proteins followed by x1, y1, a1, x2, y2, a2 and so on
def r(x, y, x_o, y_o): #radial distance between 2 proteins
    return ((x-x_o)**2 + (y-y_o)**2)**0.5

def v_x(x, y, a, x_o, y_o, a_o): #setting up dipole functions
    k = 1
    n_2D = 1
    v_1 =  -(k/(4*pi*n_2D*r(x, y, x_o, y_o)))*(1 - 2*(((x-x_o)/r(x, y, x_o, y_o))*cos(a_o) + ((y-y_o)/r(x, y, x_o, y_o))*sin(a_o))**2)*(x-x_o)/r(x, y, x_o, y_o) 
    return v_1

def v_y(x, y, a, x_o, y_o, a_o): #setting up dipole functions
    k = 1
    n_2D = 1
    v_2 = -(k/(4*pi*n_2D*r(x, y, x_o, y_o)))*(1 - 2*(((x-x_o)/r(x, y, x_o, y_o))*cos(a_o) + ((y-y_o)/r(x, y, x_o, y_o))*sin(a_o))**2)*(y-y_o)/r(x, y, x_o, y_o) 
    return v_2 

def v_a(x1, y1, a1, x2, y2, a2): #setting up dipole functions
    k = 1
    n_2D = 1
    x, y, a, x_o, y_o, a_o = symbols('x y a x_o y_o a_o', real=True)
    v_1 = -(k/(4*pi*n_2D*((x-x_o)**2 + (y-y_o)**2)**0.5))*(1 - 2*(((x-x_o)/((x-x_o)**2 + (y-y_o)**2)**0.5)*sympy.cos(a_o) + ((y-y_o)/((x-x_o)**2 + (y-y_o)**2)**0.5)*sympy.sin(a_o))**2)*(x-x_o)/((x-x_o)**2 + (y-y_o)**2)**0.5 
    v_2 = -(k/(4*pi*n_2D*((x-x_o)**2 + (y-y_o)**2)**0.5))*(1 - 2*(((x-x_o)/((x-x_o)**2 + (y-y_o)**2)**0.5)*sympy.cos(a_o) + ((y-y_o)/((x-x_o)**2 + (y-y_o)**2)**0.5)*sympy.sin(a_o))**2)*(y-y_o)/((x-x_o)**2 + (y-y_o)**2)**0.5 
    v_3 = diff(v_2, x).subs({x:x1, y:y1, a:a1, x_o:x2, y_o:y2, a_o:a2}) - diff(v_1, y).subs({x:x1, y:y1, a:a1, x_o:x2, y_o:y2, a_o:a2})
    return v_3

def F_i(x, y, a, i): #required summation of dipole forces along x direction for force on ith protein
    l = len(x)
    d = 0
    for j in range(l):
        if i != j:
            d = d + v_x(x[i], y[i], a[i], x[j], y[j], a[j])
    return d
    
def G_i(x, y, a, i): #required summation of dipole forces along y direction for force on ith protein
    l = len(x)
    d = 0
    for j in range(l):
        if i != j:
            d = d + v_y(x[i], y[i], a[i], x[j], y[j], a[j])
    return d

def H_i(x, y, a, i): #required summation of dipole forces along z direction for force on ith protein
    l = len(x)
    d = 0
    for j in range(l):
        if i != j:
            d = d + 0.5 * v_a(x[i], y[i], a[i], x[j], y[j], a[j])
    return d

def F(vec, t): #vector force on each dipole #F(vec) if using stepping up loop
    vec = vec.reshape(-1, 3)
    dot = []
    l = len(vec)
    p = []
    q = []
    s = []
    for i in range(l):
        p.append(vec[i][0])
        q.append(vec[i][1])
        s.append(vec[i][2])
    
    for i in range(l):
        dot.append([F_i(p, q, s, i), G_i(p, q, s, i), H_i(p, q, s, i)])
    dot_arr = np.array(dot,dtype='float64')
    new_dot = dot_arr.reshape(-1)
    return new_dot
x = []
y = []
a = []
n = int(input()) #take input of number of proteins from user
vec0 = []
t = np.linspace(0, 20, 50) #setting up t as a 1D vector to plot against, this is a simulation of 20 seconds with 50 data points on each plot
for i in range(n): #take input of initial x, y, alpha of each protein from user
    x.append(float(input()))
    y.append(float(input()))
    a.append(float(input()))
# x = [2,3] #sample initial conditions
# y = [1,4]
# a = [0,0]
# vec0 = [[2, 1, 0], [3, 4, 0]] #eg of vec for given initial conditions
for i in range(n):
    vec0.append([x[i], y[i], a[i]])
vec0_arr = np.array(vec0)
new_vec0 = vec0_arr.reshape(-1) 
#print(new_vec0) #used for debugging
#print(F(vec0,t)) #used for debugging

sol = odeint(F, new_vec0, t) #generates state of system at each protein as a matrix
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (12,10)) #sets up plots


ax1.plot(t, sol[:, 0])
ax1.plot(t, sol[:, 3])
ax1.title.set_text('x coordinates of proteins A and B')
ax1.set_xlabel('t')
ax1.set_ylabel('x')
ax1.legend(['protein A', 'protein B'], loc = 'lower right')

ax2.plot(t, sol[:, 1])
ax2.plot(t, sol[:, 4])
ax2.title.set_text('y coordinates of proteins A and B')
ax2.set_xlabel('t')
ax2.set_ylabel('y')
ax2.legend(['protein A', 'protein B'], loc = 'lower right')

ax3.plot(t, sol[:, 2])
ax3.plot(t, sol[:, 5])
ax3.title.set_text('Orientation of proteins A and B')
ax3.set_xlabel('t')
ax3.set_ylabel('alpha')
ax3.legend(['protein A', 'protein B'], loc = 'lower right')

ax4.plot(sol[:, 0], sol[:, 1])
ax4.plot(sol[:, 3], sol[:, 4])
ax4.title.set_text('trajectories of proteins A and B')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.legend(['protein A', 'protein B'], loc = 'lower right')
fig.tight_layout()
plt.show()