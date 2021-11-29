from numpy import pi, sqrt, zeros, array
import math
from numpy.core.function_base import linspace
import matplotlib.pyplot as plt
from scipy.integrate import odeint
 

N=10000 # no. of steps
eta2D=1 # values of parameters
k=1    



def vx_dipole(x,y,alpha,x0,y0,alpha0):    # calculating the x-component of the dipole
    r= sqrt((x-x0)**2 +(y-y0)**2)
    b= -k/(4*pi*eta2D*r)
    a1 = (x-x0)/r
    a2=(y-y0)/r
    f=b*(1-2*((a1*math.cos(alpha0) )**2))*a1
    return f


def vy_dipole(x,y,alpha,x0,y0,alpha0):    # calculating the y-component of the dipole
    r= sqrt((x-x0)**2+(y-y0)**2 )
    b= -k/(4*pi*eta2D*r)
    a1 = (x-x0)/r
    a2=(y-y0)/r
    f=b*(1-2*(a1*math.cos(alpha0) + a2*math.sin(alpha0))**2)*a2
    return f

def planerMotion(y,time):                           # model function used to calculate the derivative of y
    
    g0=vx_dipole(y[0],y[1],y[2],y[3],y[4],y[5])
    g1=vy_dipole(y[0],y[1],y[2],y[3],y[4],y[5])
    g3=vx_dipole(y[3],y[4],y[5],y[0],y[1],y[2])
    g4=vy_dipole(y[3],y[4],y[5],y[0],y[1],y[2])
    g2=(1/2) *(y[0]-y[3])*(-y[0]+y[3])*math.cos(y[5])*math.sin(y[5])/((2*pi*eta2D)*((y[0]**2 -2*y[0]*y[3]+ y[3]**2)**2))
    g5=(1/2) *(y[3]-y[0])*(-y[3]+y[0])*math.cos(y[2])*math.sin(y[2])/((2*pi*eta2D)*((y[3]**2 -2*y[3]*y[0]+ y[0]**2)**2))
   
   
    return array([g0,g1,g2,g3,g4,g5])


y=zeros([6])   # y stores the initial values values of the variables.
x1=y1=alpha1=0 #initial value of particle 1
x2=0.4
y2=0
alpha2=pi/6

time =linspace(0,200,N)
y[0]=x1     # x coordinate of particle-1
y[1]=y1     # y coordinate of particle-1
y[2]=alpha1 # angle of rotation of  particle-1
y[3]=x2     # x coordinate of particle-2
y[4]=y2     # y coordinate of particle-2
y[5]=alpha2 # angle of rotation of  particle-2
 

answer = odeint(planerMotion,y,time)   # storing the answer using odeint method of scipy lib

xdata= answer[:,0]   #info and plots on x-coordinates
ydata =answer[:,3]

plt.subplot(1,3,1)
plt.plot(time,xdata,color='r',label='particle1')
plt.plot(time,ydata,color='g',label='particle2')

plt.xlabel('time')
plt.ylabel('x-coordinate')
plt.legend(['particle 1','particle 2'])

xdata= answer[:,1]   #info and plots on y-coordinates
ydata =answer[:,4]

plt.subplot(1,3,2)
plt.plot(time,xdata,color='r',label='particle1')
plt.plot(time,ydata,color='g',label='particle2')

plt.xlabel('time')
plt.ylabel('y-coordinate')
plt.legend(['particle 1','particle 2'])

xdata= answer[:,2]   #info and plots on aplpha
ydata =answer[:,5]

plt.subplot(1,3,3)
plt.plot(time,xdata,color='r',label='particle1')
plt.plot(time,ydata,color='g',label='particle2')

plt.xlabel('time')
plt.ylabel('Alpha')
plt.legend(['particle 1','particle 2'])



plt.show()


