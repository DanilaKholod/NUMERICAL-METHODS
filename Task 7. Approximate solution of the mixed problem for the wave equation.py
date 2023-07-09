#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import math
import matplotlib.pyplot as plt

# вариант 17
# точное решение

def solution (x, t):
    return 2 * (np.sin(1+t-x))**2

# параметры задачи

a = 0  # границы по пространственной переменной
b = 1

h = 0.05  # шаг по пространсвенной переменной
I = int((b-a)/h) # кол-во шагов

T = 1 # граница по времени
tau = 0.05  # шаг по времени
N = int(T/tau) # кол-во шагов

a_eq = np.sqrt(0.5) # в уравнении а

# из формулы для аппроксимации гр. условий
a_0 = 1
a_l = 1
b_0 = 0
b_l = 1

# неоднородность
 
def f(x, t):
    return 2 * np.cos(2+2*t-2*x)  # не 4, а 2, т.к. привели к норм. виду


# начальные условия

def phi(x):
    return 2 * (np.sin(1-x))**2

def psi(x):
    return 2 * np.sin(2-2*x)
    
# граничные условия

def gamma_0(t):
    return 2 * (np.sin(1+t))**2

def gamma_l(t):
    return 2 * ((np.sin(t))**2 - np.sin(2*t))


# определяем решения
# 1-ое - координата, 2-ое - время
u_1 = np.zeros((N+1,I+1)) # 1-ый порядок точности
u_2 = np.zeros((N+1,I+1)) # 2-ой порядок точности

# заполнение нулевого слоя по времени с помощью 1-го н.у. для 1-го и 2-го порядка точности

for i in range(I+1):
    u_1[0][i] = phi(i*h)

for i in range(I+1):
    u_2[0][i] = phi(i*h)

# заполнение 1-го слоя по времени с помощью 2-го н.у.
    
# первый порядок точности
for i in range(I+1):
    u_1[1][i] = u_1[0][i] + tau * psi(i*h)

# второй порядок точности

for i in range(1, I):
    u_2[1][i] = u_2[0][i] + tau * psi(i*h) + 0.5*((a_eq*tau/h)**2) *(u_2[0][i+1] - 2*u_2[0][i] +  u_2[0][i-1]) + (tau**2 / 2) * f(h*i,0)
# граничные условия для 2-го порядка точности
u_2[1][0] = (gamma_0(1*tau)-b_0/(2*h)*(4*u_2[1][1]-u_2[1][2]))/(a_0-3*b_0/(2*h))
u_2[1][I] = (gamma_l(1*tau)+b_l/(2*h)*(4*u_2[1][I-1]-u_2[1][I-2]))/(a_l+3*b_l/(2*h))   


# дозаполняем остальные слои по времени
# 1-ый порядок точности

for j in range(1, N):
    for i in range(1, I):
        u_1[j+1][i] = 2*u_1[j][i]-u_1[j-1][i]+(a_eq*tau/h)**2*(u_1[j][i+1]-2*u_1[j][i]+u_1[j][i-1])+tau**2*f(i*h,j*tau)
    u_1[j+1][0] = (gamma_0((j+1)*tau)-b_0/(h)*(u_1[j+1][1]))/(a_0-b_0/(h))
    u_1[j+1][I] = (gamma_l((j+1)*tau)+b_l/(h)*(u_1[j+1][I-1]))/(a_l+b_l/(h))

# 2-ой порядок точности 

for j in range(1, N):
    for i in range(1, I):
        u_2[j+1][i] = 2*u_2[j][i]-u_2[j-1][i]+(a_eq*tau/h)**2*(u_2[j][i+1]-2*u_2[j][i]+u_2[j][i-1])+tau**2*f(i*h,j*tau)
    u_2[j+1][0] = (gamma_0((j+1)*tau)-b_0/(2*h)*(4*u_2[j+1][1]-u_2[j+1][2]))/(a_0-3*b_0/(2*h))
    u_2[j+1][I] = (gamma_l((j+1)*tau)+b_l/(2*h)*(4*u_2[j+1][I-1]-u_2[j+1][I-2]))/(a_l+3*b_l/(2*h))

# графики

# вычисляем значения на сетке
x = np.zeros((I+1))
for i in range(I+1):
    x[i] =  h*i
t = np.zeros((N+1))
for i in range(N+1):
    t[i] = tau*i

# вычисляем теор. значения       
y_2 = np.zeros((I+1))
for i in range(I+1):
    y_2[i] = solution(h*i, T) 

# 1-ый порядок точности  
y_1 = np.zeros((I+1))
for i in range(I+1):
    y_1[i] = u_1[N][i]
    
# 2-ой порядок точности    
y_3 = np.zeros((I+1))
for i in range(I+1):
    y_3[i] = u_2[N][i]
    
# погрешность 1-го и 2-го порядка точности соответственно
err_1 = np.zeros((I+1))
err_2 = np.zeros((I+1))

for i in range(I+1):
    err_1[i] = abs(y_1[i]-y_2[i])
    err_2[i] = abs(y_3[i]-y_2[i])

fig_p = plt.figure()
ax_p = fig_p.add_subplot(1,1,1)
ax_p.plot(x, err_1, color = 'green', linewidth = 1, label = 'Погрешность, 1-ый порядок точности')
ax_p.legend()
plt.xlabel('x', fontsize=16)
plt.ylabel(r'$\Delta $u', fontsize=16)
plt.grid(which='major')

fig_pp = plt.figure()
ax_pp = fig_pp.add_subplot(1,1,1)
ax_pp.plot(x, err_2, color = 'green', linewidth = 1, label = 'Погрешность, 2-ой порядок точности')
ax_pp.legend()
plt.xlabel('x', fontsize=16)
plt.ylabel(r'$\Delta $u', fontsize=16)
plt.grid(which='major')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x, y_1, color = 'red', linewidth = 0.5, label = 'Численное, 1-ый порядок точности')
ax.plot(x, y_1, color = 'red', linewidth = 0.5)
ax.plot(x, y_2, color = 'blue', linewidth = 1, label = 'Теоретическое')
ax.scatter(x, y_3, color = 'green', linewidth = 0.5, label = 'Численное, 2-ой порядок точности')
ax.plot(x, y_3, color = 'green', linewidth = 0.5)


ax.legend()

plt.xlabel('x', fontsize=16)
plt.ylabel('u', fontsize=16)
plt.grid(which='major')

fig.set_figwidth(12)  
fig.set_figheight(6)


# In[ ]:




