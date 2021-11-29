from math import cos, pi, sqrt, sin
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sympy
from sympy import symbols, diff
import tqdm 
#enter number of proteins followed by x1, y1, a1, x2, y2, a2 and so on
def r(x, y, x_o, y_o):
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
    v_3 = diff(v_2, x).subs({x:x1, y:y1, a:a1, x_o:x2, y_o:y2, a_o:a2}) - diff(v_1, y).subs({x:x1, y:y1, a:a1, x_o:x2, y_o:y2, a_o:a2}) #curl of (v1 i + v2 j).k
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

def F(vec): #vector force on each dipole #F(vec,t) if using odeint 
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
for i in range(n): #take input of initial x, y, alpha of each protein from user
     x.append(float(input()))
     y.append(float(input()))
     a.append(float(input()))

#x= [-0.4, -0.8] #sample initial conditions for debugging
#y = [0.3, -0.3]
#a = [0.5, 0]

for i in range(n):
    vec0.append([x[i], y[i], a[i]])

vec0 = np.array(vec0) #converts list to array for easier manipulation
sol = vec0.reshape(-1) #represents x1, y1, a1, x2, y2, a2 and so on as a 1D vector

dt = 0.4 #differential step time
tt = 40 #total time of simulation
Lx = 2 #x side of rectangle (-Lx, Lx)
Ly = 2 #y side of rectangle (-Ly, Ly)
nt = int(tt/dt) #number of step times
t = np.linspace(0, tt, nt) #setting up t as a 1D vector to plot against, tt is total time, nt is number of data points on each plot

final_sol = [sol] #final_sol stores the list of state of the system at each time as a matrix, with each row as state of the system at a particular time  #initially, final_sol stores only the initial condition
for i in tqdm.trange(nt-1): #stepping up the initial condition 99 times #tqdm allows us to track the progress of the code when running it 
        delta = F(sol) #time derivative of the state of the system
        dsol = delta*dt #differential change in state of the system
        new_sol = sol + dsol #stepped up state of the system
        for j in range(n): #incorporating boundary conditions of the rectangle
            if new_sol[3*j] > Lx/2:
                new_sol[3*j] -Lx/2
            if new_sol[3*j] < -Lx/2:
                new_sol[3*j] = Lx/2
        for j in range(n): #incorporating boundary conditions of the rectangle
            if new_sol[3*j + 1] > Ly/2:
                new_sol[3*j + 1] -Ly/2
            if new_sol[3*j + 1] < -Ly/2:
                new_sol[3*j + 1] = Ly/2
        final_sol = np.concatenate((final_sol, [new_sol])) #adds the stepped up state of the system as a row in final_sol
        sol = new_sol #updates the state of the system for the next iteration of the for loop





#sol = odeint(F, new_vec0, t) #this line is used if we don't want the phase space to be confined to the rectangle

#print(final_sol) #used for debugging

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (12,10)) #setting up plots


ax1.plot(t, final_sol[:, 0])
ax1.plot(t, final_sol[:, 3])
ax1.title.set_text('x coordinates of proteins A and B')
ax1.set_xlabel('t')
ax1.set_ylabel('x')
ax1.legend(['protein A', 'protein B'], loc = 'lower right')

ax2.plot(t, final_sol[:, 1])
ax2.plot(t, final_sol[:, 4])
ax2.title.set_text('y coordinates of proteins A and B')
ax2.set_xlabel('t')
ax2.set_ylabel('y')
ax2.legend(['protein A', 'protein B'], loc = 'lower right')

ax3.plot(t, final_sol[:, 2])
ax3.plot(t, final_sol[:, 5])
ax3.title.set_text('Orientation of proteins A and B')
ax3.set_xlabel('t')
ax3.set_ylabel('alpha')
ax3.legend(['protein A', 'protein B'], loc = 'lower right')

ax4.plot(final_sol[:, 0], final_sol[:, 1])
ax4.plot(final_sol[:, 3], final_sol[:, 4])
ax4.title.set_text('trajectories of proteins A and B')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.legend(['protein A', 'protein B'], loc = 'lower right')
fig.tight_layout()
plt.show()