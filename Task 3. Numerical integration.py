#!/usr/bin/env python
# coding: utf-8

# In[7]:


import matplotlib.pyplot as plt
import numpy as np

from scipy import misc


# In[8]:


# вычисление истинного значения через нахождение руками неопределенного интеграла

def integral_an(x):  # функция после интегрирования
    f = 5 * math.tan(x / 10)
    return f


def an(a, b):  # определенный интеграл
    result = integral_an(b) - integral_an(a)
    return result


def func(x):  # функция для интегрирования
    f = 1 / (1 + np.cos(x / 5))
    return f


# In[9]:


# вычисление интегралов с использованием численного интегрирования
# вариант 17 [a,b] = [-3,3]

def find_rectangle_left(a, b, n):  # левый прямоугольник
    n = int(n)
    X = [0] * n
    h = (b - a) / (n - 1)
    for i in range(n):
        X[i] = a + i * h
    f = 0
    for i in range(n - 1):
        f += h * func(X[i])
    return f


def find_rectangle_central(a, b, n):  # центральный прямоугольник
    n = int(n)
    X = [0] * n
    h = (b - a) / (n - 1)
    for i in range(n):
        X[i] = a + i * h
    f = 0
    for i in range(n - 1):
        f += h * func((X[i + 1] + X[i]) / 2)
    return f


def find_trapeze(a, b, n):  # трапеция
    n = int(n)
    X = [0] * n
    h = (b - a) / (n - 1)
    for i in range(n):
        X[i] = a + i * h
    f = 0
    for i in range(n - 1):
        f += h * (func(X[i + 1]) + func(X[i])) / 2
    return f


def find_Simpson(a, b, n):  # симпсон
    n = int(n)
    X = [0] * n
    h = (b - a) / (n - 1)
    for i in range(n):
        X[i] = a + i * h
    f = 0
    for i in range(n - 1):
        f += h * (func(X[i + 1]) + 4 * func(((X[i + 1] + X[i]) / 2)) + func(X[i])) / 6
    return f


a = -3  # lower limit of integration
b = 3  # upper limit of integration
n = 51  # number of segments

analytical = an(a, b)
print('Аналитическое значение:', analytical)
rectangle_left = find_rectangle_left(a, b, n)
print('Прямоугольник левый:', rectangle_left)
rectangle_central = find_rectangle_central(a, b, n)
print('Прямоугольник центральный:', rectangle_central)
trapeze = find_trapeze(a, b, n)
print('Трапеция:', trapeze)
Simpson = find_Simpson(a, b, n)
print('Симпсон:', Simpson)


# In[10]:


# здесь смотрим ошибку в зависимости от шага разбиения

def second_func(x):  # вторая производная от исходной функции
    cos = np.cos(x / 5)
    sin = np.sin(x / 5)
    return (2 * cos + 2 + cos * sin ** 2) / (25 * (1 + cos) ** 4)


def fourth_diff_4(a):  # четвертая производная от исходной функции
    return misc.derivative(func, a, n=4, order=5)


def find_second(a, b, n):  # максимум по вторым производным
    n = int(n)
    X = [0] * n
    h = (b - a) / (n - 1)
    for i in range(n):
        X[i] = a + i * h
    f = np.ones(len(X))
    for i in range(n):
        f = second_func(X[i])
    max_second = np.max(f)
    return max_second


def find_fourth(a, b, n):  # максимум по четвертым производным
    n = int(n)
    X = [0] * n
    h = (b - a) / (n - 1)
    for i in range(n):
        X[i] = a + i * h
    f = np.ones(len(X))
    for i in range(n):
        f = fourth_diff_4(X[i])
    max_second = np.max(f)
    return max_second


array_n = np.ones(5)  # разное кол-во прямоугольников
array_n[0] = 50
array_n[1] = 100
array_n[2] = 300
array_n[3] = 500
array_n[4] = 1000

step = np.ones(5)  # разный шаг сетки
for i in range(5):
    step[i] = (b - a) / (array_n[i] - 1)

# In[14]:


rectangle_central_arr_max = np.ones(5)  # центральные прямоугольники

for i in range(5):
    rectangle_central_arr_max[i] = find_second(a, b, array_n[i])

rectangle_central_error = np.ones(5)  # ошибка
for i in range(5):
    rectangle_central_error[i] = ((b - a) / 24) * rectangle_central_arr_max[i] * step[i] ** 2

print('rectangle_central_error', rectangle_central_error)

fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.plot(step, rectangle_central_error, label='Прямоугольники')

ax1.set_xlabel('Шаг сетки')
ax1.set_ylabel('Погрешность')

ax1.set_title(r'Погрешность от шага')
ax1.grid()
ax1.legend()

# In[15]:


trapeze_arr_max = np.ones(5)  # трапеции

for i in range(5):
    trapeze_arr_max[i] = find_second(a, b, array_n[i])

trapeze_error = np.ones(5)  # ошибка
for i in range(5):
    trapeze_error[i] = ((b - a) / 12) * trapeze_arr_max[i] * step[i] ** 2

print('trapeze_error', trapeze_error)

fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.plot(step, trapeze_error, label='Трапеция')

ax2.set_xlabel('Шаг сетки')
ax2.set_ylabel('Погрешность')
ax2.set_title(r'Погрешность от шага')
ax2.grid()
ax2.legend()

# In[16]:


Simpson_arr_max = np.ones(5)  # трапеции

for i in range(5):
    Simpson_arr_max[i] = find_fourth(a, b, array_n[i])

Simpson_error = np.ones(5)  # ошибка: max вторая производная * шаг^2
for i in range(5):
    Simpson_error[i] = ((b - a) / 180) * Simpson_arr_max[i] * step[i] ** 4

print('Simpson_error', Simpson_error)

fig3, ax3 = plt.subplots(figsize=(10, 8))
ax3.plot(step, Simpson_error, label='Симпсон')

ax3.set_xlabel('Шаг сетки')
ax3.set_ylabel('Погрешность')
ax3.set_title(r'Погрешность от шага')
ax3.grid()
ax3.legend()

# In[ ]:




