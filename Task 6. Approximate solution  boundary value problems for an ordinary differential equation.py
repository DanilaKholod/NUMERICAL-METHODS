# краевая задача для ОДУ
import math
import numpy as np
import matplotlib.pyplot as plt

# определяем точное решение задачи
def exact_sol(x):
    return (math.log(1+x)+2*math.exp(-x))

# задаем параметры задачи
a = 0.0                     # левый конец отрезка
b = 1.0                     # правый конец отрезка
h = 0.05                    # шаг сетки
b_1 = 0.0                   # граничное условие в a
a_1 = 1.0
y_1 = -1.0
b_2 = 6.0                   # граничное условие в b
a_2 = 1.0
y_2 = 8.3377
# N = int((b-a)/h)            # количество узлов в сетке

# оределяем коэффициенты уравнения
def p(x):                                   #коэффициент при первой производной    
    return 1/(1+x)  

def q(x):                                   #коэффициент при второй производной
    return -x/(1+x)
    
def f(x):                                   # свободный член
    return (-x*math.log(1+x))/(1+x)

# c0 b0 0 ..                          y0   d0
# a1 c1 b1 0  ...                     .    .
# 0  a2 c2 b2 0 ...                   .    .
#  ...                                .  = .
#           0  a[n-2] c[n-2] b[n-2]   .    .
#                  0  a[n-1] c[n-1]   yn   dn
def tridiagonal_matrix_algorithm(a, c, b, d):
    n = len(d)
    alpha = np.ndarray(shape=(n,), dtype=float)
    beta = np.ndarray(shape=(n,), dtype=float)
    alpha[1] = -b[0] / c[0]
    beta[1] = d[0] / c[0]
    for i in range(1, n - 1):
        alpha[i + 1] = -b[i] / (a[i] * alpha[i] + c[i])
        beta[i + 1] = (d[i] - a[i] * beta[i]) / (a[i] * alpha[i] + c[i])
    y = np.ndarray(shape=(n,))
    y[n - 1] = (d[n - 1] - a[n - 1] * beta[n - 1]) / (c[n - 1] + a[n - 1] * alpha[n - 1])
    for i in reversed(range(1, n)):
        y[i - 1] = alpha[i] * y[i] + beta[i]
    return y

x = np.arange(a, b + h, h) # узлы на равномерную сетку
n = len(x)
u_theory = np.zeros(n)
u_exper_1 = np.zeros(n) #1-ый порядок точности
u_exper_2 = np.zeros(n) #2-ой порядок точности

# теоретическое значение
for i in range (n):
    u_theory[i] = exact_sol(x[i])

p = [p(i) for i in x]
q = [q(i) for i in x]
f = [f(i) for i in x]

# вводим дополнительные переменные
a = np.zeros(n)
b = np.zeros(n)
c = np.zeros(n)
d = np.zeros(n)
    
# 1-ый порядок точности

c[0] = -a_1/h+b_1 
b[0] = a_1 / h        
d[0] = y_1          

a[n-1] = -a_2 / h
c[n-1] = a_2/h + b_2 
d[n-1] = y_2

for i in range(1, n-1):
    a[i] = 1/h**2 - p[i] / (2 * h)
    c[i] = -2/h**2 + q[i]
    b[i] = 1/h**2 + p[i] / (2*h)
    d[i] = f[i]

u_exper_1 = tridiagonal_matrix_algorithm(a, c, b, d)

# 2-ой порядок точности
    
c[0] = -2/h**2+2*b_1/(a_1*h)-p[0]*b_1/a_1
b[0] = 2/h**2
d[0] = f[0]+2*y_1/(h)-p[0]*y_1/(a_1)
a[n-1] = 2/h**2
c[n-1] = -2/h**2 -2*b_2/(h*a_2) - p[n-1]*b_2/a_2+q[n-1]
d[n-1] = f[n-1]-2*y_2/(h*a_2)-p[n-1]*y_2/a_2

u_exper_2 = tridiagonal_matrix_algorithm(a, c, b, d)

# погрешность 1-го и 2-го порядка точности соответственно
err_1 = np.zeros(n)
err_2 = np.zeros(n)

for i in range(n-1):
    err_1[i] = abs(u_exper_1[i]-u_theory[i])
    err_2[i] = abs(u_exper_2[i]-u_theory[i])
    
plt.title("1-ый порядок точности")
plt.grid()
plt.plot(x, u_exper_1, color = 'purple', label = "расчетное")
plt.plot(x, u_theory, color = 'orange', label = "теоретическое" )
plt.legend()
plt.show()

plt.title("2-oй порядок точности")
plt.grid()
plt.plot(x, u_exper_2, color = 'purple', label = "расчетное")
plt.plot(x, u_theory, color = 'orange', label = "теоретическое" )
plt.legend()
plt.show()

plt.title("погрешность")
plt.grid()
plt.plot(x, err_1, color = 'green')
plt.plot()
plt.show()

plt.title("погрешность")
plt.grid()
plt.plot(x, err_2, color = 'green')
plt.plot()
plt.show()