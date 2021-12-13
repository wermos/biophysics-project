from math import cos, pi, sqrt, sin
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sympy
from sympy import symbols, diff, sympify, lambdify
import tqdm 
import math
import dill
from scipy.special import eval_legendre
#enter number of proteins followed by theta1, phi1, a1, theta2, phi2, a2 and so on 
#code returns a matrix with each row as a 1D vector of theta1, phi1, a1, theta2, phi2, a2 and so on at a particular time
#theta -> t, phi -> p, alpha -> a
R = 1
n2d = 1000
npv = 1 #n+
nnv = 1 #n-
k = 1
lp = n2d/npv
ln = n2d/nnv
t, p, a, t_o, p_o, a_o = symbols('t, p, a, t_o, p_o, a_o', real=True)
def v_d():
    t, p, a, t_o, p_o, a_o = symbols('t, p, a, t_o, p_o, a_o', real=True)

    T = np.array([[sympy.sin(a_o), sympy.cos(a_o)]])

    array = np.array([[0, -1], [1, 0]])
    M = np.array([[0, -sympy.cos(t_o)*sympy.cos(a_o)/(R*sympy.sin(t_o))], [sympy.cos(t_o)*sympy.cos(a_o)/(R*sympy.sin(t_o)), 0]])
    g = sympy.acos(sympy.sin(t)*sympy.sin(t_o)*sympy.cos(p - p_o) + sympy.cos(t)*sympy.cos(t_o))
    # def P(n): #defined a function to calculated the nth legendre polynomial, not used in this code
    #     if(n == 0):
    #         return 1 # P0 = 1
    #     elif(n == 1):
    #         return sympy.cos(g) # P1 = x
    #     elif(n == 2):
    #         return 1.5*(sympy.cos(g))**2 - 0.5
    #     elif(n == 3):
    #         return 2.5*(sympy.cos(g))**3 - 1.5*(sympy.cos(g))
    #     elif(n == 4):
    #         return (35/8)*(sympy.cos(g))**4 - (15/4)*(sympy.cos(g))**2 + 3/8
    #     elif(n == 5):
    #         return (63/8)*(sympy.cos(g))**5 - (35/4)*(sympy.cos(g))**3 + (15/8)*(sympy.cos(g))
    #     elif(n == 6):
    #         return (231/16)*(sympy.cos(g))**6 - (315/16)*(sympy.cos(g))**4 + (105/16)*(sympy.cos(g))**2 - 5/16
    #     elif(n == 7):
    #         return (429/16)*(sympy.cos(g))**7 - (693/16)*(sympy.cos(g))**5 + (315/16)*(sympy.cos(g))**3 - (35/16)*(sympy.cos(g))
    #     elif(n == 8):
    #         return (6435/128)*(sympy.cos(g))**8 - (12012/128)*(sympy.cos(g))**6 + (6930/128)*(sympy.cos(g))**4 - (1260/128)*(sympy.cos(g))**2 + (35/128)
    #     elif(n == 9):
    #          return (12155/128)*(sympy.cos(g))**9 - (25740/128)*(sympy.cos(g))**7 + (18018/128)*(sympy.cos(g))**5 - (4620/128)*(sympy.cos(g))**3 + (315/128)*(sympy.cos(g))
    #     elif(n == 10):
    #         return (46189/256)*(sympy.cos(g))**10 - (109395/256)*(sympy.cos(g))**8 + (90090/256)*(sympy.cos(g))**6 - (30030/256)*(sympy.cos(g))**4 + (3465/256)*(sympy.cos(g))**2 - 63/256
    #     else:
    #         return (((2 * n)-1)*sympy.cos(g) * P(n-1)-(n-1)*P(n-2))/float(n)
    
    def S(t, p, t_o, p_o):
        x = 0
        def s(q):
            return q*(q + 1) - 2 + (R/ln)*(q - 1) + (R/lp)*(q + 2)
        with open("legendre_poly.pkl", "rb") as fp:
            poly = dill.load(fp)
        for u in tqdm.trange(10): #numbers of polynomials we want to sum to, 10 should suffice for n2d = 1000
            w = u + 1
            x = x + poly[w]*(2*w + 1)/(s(w)*w*(w+1))#((2*i + 1)/(s(i)*i*(i+1)))*P(i)
        return x
    
    #print(S(1,2,3,4))
    d = np.array([[(1/sympy.sin(t_o))*diff(S(t, p, t_o, p_o), p_o)], [-1*diff(S(t, p, t_o, p_o), t_o)]])
    c = np.dot(T, M)
    f = np.dot(T, d)
    b = f.item()
    e = np.array([[diff(b, t_o)], [(1/sympy.sin(t_o))*diff(b, p_o)]])
    a = np.dot(T, e)
    v = a + np.dot(c, d)
    v1 = np.array([[((1/sympy.sin(t))*diff(v.item(), p))], [(-1*diff(v.item(), t))]])
    
    v2 = (k/(4*pi*n2d*R))*v1
    i = v2[0][0]
    j = v2[1][0]

    list_a = [i, j]
    
    return list_a

def evaluate_v_d(t1, p1, a1, t2, p2, a2):
    aa = sympify(v_d()[0])
    ab = sympify(v_d()[1])
    aa = float(aa.evalf(subs={t:t1, p:p1, a:a1, t_o:t2, p_o:p2, a_o:a2}))
    ab = float(ab.evalf(subs={t:t1, p:p1, a:a1, t_o:t2, p_o:p2, a_o:a2}))
    return [aa, ab]
    
#print(evaluate_v_d(1, 2, 3, 4, 5, 6))
def curl_vd(t1, p1, a1, t2, p2, a2):
    a = v_d()[0]
    b = v_d()[1]
    c = (1/(R*sympy.sin(t)))*(diff(b*sympy.sin(t), t) - diff(a, p))
    return c.evalf(subs = {t:t1, p:p1, a:a1, t_o:t2, p_o:p2, a_o:a2})

def F_i(theta, phi, alpha, i): #sum for theta component for i protein
    l = len(theta)
    d = 0
    for j in range(l):
        if i != j:
            d = d + (1/R)*float((evaluate_v_d(theta[i], phi[i], alpha[i], theta[j], phi[j], alpha[j])[0]))
    return d

def G_i(theta, phi, alpha, i): #sum for phi component for i protein
    l = len(theta)
    d = 0
    for j in range(l):
        if i != j:
            d = d + (1/(R*math.sin(theta[i])))*float(evaluate_v_d(theta[i], phi[i], alpha[i], theta[j], phi[j], alpha[j])[1])
    return d

def H_i(theta, phi, alpha, i): #sum for alpha component for i protein
    l = len(theta)
    d = 0
    for j in range(l):
        if i != j:
            d = d + (1/2)*float((curl_vd(theta[i], phi[i], alpha[i], theta[j], phi[j], alpha[j]))) + (math.cos(theta[i])/(R*math.sin(theta[i])))*float(evaluate_v_d(theta[i], phi[i], alpha[i], theta[j], phi[j], alpha[j])[1])
    return d

def F(vec, time): #vector force on each dipole #use F(vec) if using stepping up loop for finite rectangle
    vec = vec.reshape(-1, 3)
    dot = []
    l = len(vec)
    theta1 = []
    phi1 = []
    alpha1 = []
    for i in range(l):
        theta1.append(vec[i][0])
        phi1.append(vec[i][1])
        alpha1.append(vec[i][2])
    
    for i in range(l):
        dot.append([F_i(theta1, phi1, alpha1, i), G_i(theta1, phi1, alpha1, i), H_i(theta1, phi1, alpha1, i)])
    dot_arr = np.array(dot,dtype='float64')
    new_dot = dot_arr.reshape(-1)
    return new_dot


theta_inp = []
phi_inp = []
alpha_inp = []
n = int(input()) #take input of number of proteins from user
vec0 = []
time = np.linspace(0, 10, 2) #setting up t as a 1D vector to plot against, this is a simulation of 20 seconds with 50 data points on each plot
for i in range(n): #take input of initial x, y, alpha of each protein from user
    theta_inp.append(float(input()))
    phi_inp.append(float(input()))
    alpha_inp.append(float(input()))
for i in range(n):
    vec0.append([theta_inp[i], phi_inp[i], alpha_inp[i]])
vec0 = np.array(vec0)
vec0 = vec0.reshape(-1) 
sol = odeint(F, vec0, time)
print(sol) #prints the final matrix of different coordinates at different times. Use this for plotting.