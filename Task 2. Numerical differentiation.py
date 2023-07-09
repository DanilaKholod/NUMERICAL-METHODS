# numerical differentiation
import math
import numpy as np
import matplotlib.pyplot as plt


# определяем функцию
def func(x):
    return 2 ** (np.sin(x))


# 1-ая математическая производная
def diff_f_1(x):
    return 2 ** (np.sin(x)) * (math.log1p(1)) * (np.cos(x))


# 2-ая математическая производная
def diff_f_2(x):
    return 2 ** (np.sin(x)) * (math.log1p(1)) * ((math.log1p(1)) * (np.cos(x)) ** 2 - np.sin(x))


# первый порядок точности
def numerical_diff_1(f, x):
    n = len(x)
    df = [0.0] * (n)
    # определяем правую разность
    for i in range(n - 1):
        df[i] = (f[i + 1] - f[i]) / (x[i + 1] - x[i])
    # определяем левую разность для крайней правой точки
    df[n - 1] = (f[n - 1] - f[n - 2]) / (-x[n - 2] + x[n - 1])
    return df


# второй порядок точности для первой производной
def numerical_diff_2(f, x):
    n = len(x)
    df = [0.0] * n
    # определяем центральную разность
    for i in range(n - 1):
        df[i] = (f[i + 1] - f[i - 1]) / (2 * (x[i + 1] - x[i]))
    # несимметричные разности на границах
    df[0] = (-3 * f[0] + 4 * f[1] - f[2]) / (2 * (x[1] - x[0]))
    df[n - 1] = (f[n - 3] - 4 * f[n - 2] + 3 * f[n - 1]) / (2 * (x[n - 1] - x[n - 2]))
    return df


# второй порядок точности для второй производной
def numerical_diff_3(f, x):
    n = len(x)
    df = [0.0] * (n)
    # определяем центральную разность
    for i in range(n):
        if not (i == 0 or i == n - 1):
            df[i] = (f[i - 1] - 28 * f[i] + f[i + 1]) / ((x[i + 1] - x[i]) ** 2)
        else:
            # несимметричные разности на границах
            df[i] = (4 * f[i] - 9 * f[abs(i - 1)] + 6 * f[abs(i - 2)] - f[abs(i - 3)]) / (
                        3 * (x[abs(i - 1)] - x[i]) ** 2)

    return df


step = 0.1  # определяем шаг
x = np.arange(-1.5, 1.5 + step, step)  # узлы на равномерную сетку
f = func(x)
plt.title("Первая производная 1-го порядка точности")
plt.plot(x, diff_f_1(x), color='b')
plt.plot(x, numerical_diff_1(f, x), color='r')
plt.legend(['Теоретическое', 'Расчетное'])
plt.plot()
plt.show()

plt.title("Первая производная 2-го порядка точности")
plt.plot(x, diff_f_1(x), color='b')
plt.plot(x, numerical_diff_2(f, x), color='r')
plt.legend(['Теоретическое', 'Расчетное'])
plt.plot()
plt.show()

# производная 2-го порядка точности на меньшем интервале
x1 = np.arange(0.0, 1.0 + step, step)
f1 = func(x1)
plt.title("Первая производная 2-го порядка точности на меньшем интервале")
plt.plot(x1, diff_f_1(x1), color='b')
plt.plot(x1, numerical_diff_2(f1, x1), color='r')
plt.legend(['Теоретическое', 'Расчетное'])
plt.plot()
plt.show()

plt.title("Вторая производная 2-го порядка точности")
plt.plot(x, diff_f_2(x), color='b')
plt.plot(x, numerical_diff_3(f, x), color='r')
plt.legend(['Теоретическое', 'Расчетное'])
plt.plot()
plt.show()

plt.plot(x, abs(diff_f_1(x) - numerical_diff_1(f, x)), color='green')
plt.legend(['Ошибка первого порядка точности'])
plt.plot()
plt.show()

plt.plot(x, abs(diff_f_1(x) - numerical_diff_2(f, x)), color='blue')
plt.legend(['Ошибка второго порядка точности'])
plt.plot()
plt.show()


diff_values = [abs(diff_f_1(x) - numerical_diff_1(f, x)) for i in range(len(x))]



