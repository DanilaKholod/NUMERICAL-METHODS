# библиотеки:

import numpy as np
import math
import matplotlib.pyplot as plt


# вариант 20:

# точное решение:

def solution(x, t):
    return 3 * (np.log(1 + x * t)) - x ** 2 / 3


# параметры задачи:>

# границы по пространственной переменной:

a = 0
b = 1

# шаг по пространсвенной переменной:

h = 0.05

# кол-во шагов:

I = int((b - a) / h)

# граница по времени:

T = float(input('T = '))

# шаг по времени:

tau = float(input('tau = '))

# кол-во шагов:

N = int(T / tau)

# задаем параметр sigma:

sigma = float(input('sigma = '))


# неоднородность:

def f(x, t):
    return 2 / 3 + 3 * (x + t ** 2 + x ** 2 * t) / (1 + x * t) ** 2


# начальные условия:

def phi(x):
    return -x ** 2 / 3


# граничные условия:

def gamma_0(t):
    return -3 * t


def gamma_1(t):
    return (3 * t / (1 + t)) - 2 / 3


# из формулы для аппроксимации гр. условий:

a_0 = 5
a_l = 0
b_0 = -1
b_l = 1

# определяем решение:

# 1-ое - координата, 2-ое - время:

u_1 = np.zeros((N + 1, I + 1))
A = np.zeros(I + 1)
B = np.zeros(I + 1)
C = np.zeros(I + 1)
d = np.zeros((N + 1, I + 1))

# прогоночные коэффициенты рекурентной формулы:

a = np.zeros(I)
b = np.zeros(I)

# аппроксимация Н.У:

for i in range(I + 1):
    u_1[0][i] = phi(i * h)

A[I] = - b_l / h
B[0] = a_0 - b_0 / h
B[I] = a_l + b_l / h
C[0] = b_0 / h

for i in range(1, I):
    A[i] = -(tau * sigma) / h ** 2
    B[i] = 1 + 2 * tau * sigma / h ** 2
    C[i] = -(tau * sigma) / h ** 2

# аппроксимация уравнения:

for n in range(1, N + 1):
    d[n][0] = gamma_0(n * tau)
    d[n][I] = gamma_1(n * tau)
    a[0] = -C[0] / B[0]
    b[0] = d[n][0] / B[0]
    # прямой ход прогонки:
    for i in range(1, I):
        d[n][i] = u_1[n - 1][i] + tau * (1 - sigma) / h ** 2 * (
                    u_1[n - 1][i + 1] - 2 * u_1[n - 1][i] + u_1[n - 1][i - 1]) + tau * f(i * h, n * tau)
        a[i] = -C[i] / (A[i] * a[i - 1] + B[i])
        b[i] = (d[n][i] - A[i] * b[i - 1]) / (A[i] * a[i - 1] + B[i])
    u_1[n][I] = (d[n][I] - A[I] * b[I - 1]) / (A[I] * a[I - 1] + B[I])
    # обратный ход прогонки:
    for i in range(I, 0, -1):
        u_1[n][i - 1] = a[i - 1] * u_1[n][i] + b[i - 1]

# графики:

# вычисляем значения на сетке:

x = np.zeros((I + 1))
for i in range(I + 1):
    x[i] = h * i
t = np.zeros((N + 1))
for i in range(N + 1):
    t[i] = tau * i

# вычисляем теор. значения:

y_2 = np.zeros((I + 1))
for i in range(I + 1):
    y_2[i] = solution(h * i, T)

# 1-ый порядок точности:

y_1 = np.zeros((I + 1))
for i in range(I + 1):
    y_1[i] = u_1[N][i]

# погрешность 1-го порядка точности:

err_1 = np.zeros((I + 1))

for i in range(I + 1):
    err_1[i] = abs(y_1[i] - y_2[i])

fig_p = plt.figure()
ax_p = fig_p.add_subplot(1, 1, 1)
ax_p.plot(x, err_1, color='green', linewidth=1, label='Погрешность, 1-ый порядок точности')
ax_p.legend()
plt.xlabel('x', fontsize=16)
plt.ylabel(r'$\Delta $u', fontsize=16)
plt.grid(which='major')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x, y_1, color='purple', linewidth=0.5, label='Численное, 1-ый порядок точности')
ax.plot(x, y_1, color='purple', linewidth=0.5)
ax.plot(x, y_2, color='yellow', linewidth=1, label='Теоретическое')

ax.legend()

plt.xlabel('x', fontsize=16)
plt.ylabel('u', fontsize=16)
plt.grid(which='major')

fig.set_figwidth(12)
fig.set_figheight(6)

plt.show()
