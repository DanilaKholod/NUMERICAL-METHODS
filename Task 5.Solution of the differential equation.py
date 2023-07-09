import math
import numpy as np
import matplotlib.pyplot as plt


# Определяем функцию для метода Эйлера
def euler(f, x0, y0, h, xn):
    # Создаем массивы для хранения координат x и y
    x = np.arange(x0, xn + h, h)
    y = np.zeros((len(x), len(y0)))
    # Устанавливаем начальное значение y
    y[0] = y0
    # Используем метод Эйлера для нахождения y на каждом шаге
    for i in range(len(x) - 1):
        y[i + 1] = y[i] + h * np.array(f(x[i], y[i]))
    return x, y


# Определяем функцию для метода Рунге-Кутты четвертого порядка точности
def runge_kutta(f, x0, y0, h, xn):
    # Создаем массивы для хранения координат x и y
    x = np.arange(x0, xn + h, h)
    y = np.zeros((len(x), len(y0)))
    # Устанавливаем начальное значение y
    y[0] = y0
    # Используем метод Рунге-Кутты четвертого порядка точности для нахождения y на каждом шаге
    for i in range(len(x) - 1):
        k1 = np.array(f(x[i], y[i]))
        k2 = np.array(f(x[i] + h / 2, y[i] + h / 2 * k1))
        k3 = np.array(f(x[i] + h / 2, y[i] + h / 2 * k2))
        k4 = np.array(f(x[i] + h, y[i] + h * k3))
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x, y


# Определяем правую часть дифференциального уравнения
def f(x, y):
    return [y[1], -1 / (1 + x) * y[1] - 2 / (np.cosh(x) ** 2) * y[0] + 1 / (np.cosh(x) ** 2) * (
            1 / (1 + x) + 2 * np.log(1 + x))]


# Точное решение
def U(x):
    return np.tanh(x) + np.log(1 + x)


y0 = [0, 2]
h = 0.05
xn = 1

x_euler, y_euler = euler(f, 0, y0, h, xn)
x_rk4, y_rk4 = runge_kutta(f, 0, y0, h, xn)

fig, ax = plt.subplots(figsize=(5, 5))
plt.plot(x_euler, y_euler[:, 0], label='Метод Эйлера')
plt.plot(x_rk4, y_rk4[:, 0], label='Метод Рунге-Кутты 4')
plt.plot(x_euler, U(x_euler), label='Точное решение')
plt.legend()
plt.show()

h1 = 0.1
x_rk4_h1, y_rk4_h1 = runge_kutta(f, 0, y0, h1, xn)
y_rk4_h1_half = y_rk4_h1[::2]  # значения на половине узлов грубой сетки
x_rk4_h1_half = x_rk4_h1[::2]

x_rk4_h05_half, y_rk4_h05_half = runge_kutta(f, 0, y0, h / 2, x_rk4_h1_half[-1])
y_rk4_h05_half = y_rk4_h05_half[::2]  # значения на половине узлов подробной сетки

R = abs(y_rk4_h05_half[::4, 0] - y_rk4_h1_half[:, 0]) / 15
plt.plot(x_rk4_h1_half, R)
plt.title('Погрешность метода Рунге-Кутты 4')
plt.show()
