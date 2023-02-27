import numpy as np
import matplotlib.pyplot as plt


# определяет функцию
def f(x):
    return x ** 3 - 2 * x ** 2 + x + 3


# функция для вычисления полинома Лагранжа в заданной точке x
def lagrange(x, points, values):
    result = 0
    n = len(points)
    for i in range(n):
        term = values[i]
        for j in range(n):
            if j != i:
                term = term * (x - points[j]) / (points[i] - points[j])
        result = result + term
    return result


start = int(input("Введите начальную точку отрезка: "))  # запрашиваем начальную точку
end = int(input("Введите конечную точку отрезка: "))  # запрашиваем конечную точку
nodes_type = input("Введите тип узлов интерполяции (Chebyshev / Step): ") # запрашиваем тип узлов интерполяции

if nodes_type.lower() == 'chebyshev':
    n = int(input("Введите количество узлов интерполяции: "))  # запрашиваем количество узлов
    # вычисляем узлы Чебышева
    nodes = [(start + end) / 2 + ((end - start) / 2) * np.cos((2 * k - 1) * np.pi / (2 * n)) for k in range(1, n + 1)]
if nodes_type.lower() == 'step':
    step = float(input("Введите шаг: "))  # запрашиваем шаг
    n = int((end - start) / step) + 1  # вычисляем количество узлов
    nodes = [start + step * i for i in range(n)] # вычисляем узлы с определенным шагом

values = []  # создаем пустой массив для значений в узлах
# цикл для вычисления значений функции в каждой точке
for nod in nodes:
    values.append(f(nod))

# интерполяция функции в точках - середины узлов интерполяции
midpoints = []  # создаем пустой массив для серединных точек
interpolated = []
# цикл для вычисления серединных точек на отрезке
for i in range(len(nodes) - 1):
    midpoint = (nodes[i] + nodes[i + 1]) / 2
    midpoints.append(midpoint)
# цикл для вычисления интерполированных значений функции в серединных точках
for x in midpoints:
    interpolated.append(lagrange(x, nodes, values))
# создаем графики с функцией и интерполированными значениями функции в серединных точках
plt.plot(nodes, values, label='Функция')
plt.plot(midpoints, interpolated, 'x', label='Интерполированные точки')
plt.legend()
plt.show()
