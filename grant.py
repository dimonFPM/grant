import cmath
import datetime
import math as m
import numpy as np
import numba as nb
from statistics import mean
import matplotlib.pyplot as plt

# x = 1
# y = 1
xx = []
yy = []
rez = []
rez_all = []


# @nb.njit(nopython=True, parallel=True)
def f2(nu: tuple = (0.1, 0.2, 0.3, 0.4)) -> None:
    '''функция для проверки заданных нулей функции, которые мне скинул Бабешко
    ныжно только для этого. Из конечной программы это надо удалить'''
    x, y, u = 10, 10, 0.1
    print("подстановка в формулы при u=x+iy и x=", x, "  y=", y, "  nu=", u, ": ")
    print("начальная формула = ",
          2 * (3 - 4 * u) * cmath.cosh(2 * complex(x, y)) + 4 * complex(x, y) ** 2 + 1 + (1 - u) ** 2)
    print("разложение        = ",
          (6 + 12 * m.sinh(x) * m.sinh(x) * m.cos(y) * m.cos(y) - 12 * m.cosh(x) * m.cosh(x) * m.sin(y) * m.sin(
              y) - 8 * u - 16 * u * m.sinh(x) * m.sinh(x) * m.cos(y) * m.cos(y) + 16 * u * m.cosh(x) * m.cosh(
              x) * m.sin(y) * m.sin(y) + 4 * x * x - 4 * y * y + 2 - 2 * u + u * u) + complex(0, 1) * (
                  24 * m.cosh(x) * m.sinh(x) * m.sin(y) * m.cos(y) - 32 * u * m.cosh(x) * m.sinh(x) * m.sin(
              y) * m.cos(y) + 8 * x * y))
    print("\n")
    for u in nu:
        print("nu=", u)
        print("x            y            sqrt(r1 ** 2 + r2 ** 2)")
        xxx = (0.0000, 1.2865, 2.1051, 1.6175, 3.6613, 5.7917, 8.8385)
        yyy = (0.7456, 2.6688, 5.9380, 6.5978, 6.5976, 6.6412, 5.7926)
        for i in range(len(xxx)):
            x = xxx[i]
            y = yyy[i]

            # region формула два
            r1 = (6 + 12 * m.sinh(x) * m.sinh(x) * m.cos(y) * m.cos(y) - 12 * m.cosh(x) * m.cosh(x) * m.sin(y) * m.sin(
                y) - 8 * u - 16 * u * m.sinh(x) * m.sinh(x) * m.cos(y) * m.cos(y) + 16 * u * m.cosh(x) * m.cosh(
                x) * m.sin(y) * m.sin(y) + 4 * x * x - 4 * y * y + 2 - 2 * u + u * u)
            r2 = (24 * m.cosh(x) * m.sinh(x) * m.sin(y) * m.cos(y) - 32 * u * m.cosh(x) * m.sinh(x) * m.sin(
                y) * m.cos(y) + 8 * x * y)
            # endregion
            r = m.sqrt(r1 ** 2 + r2 ** 2)
            print(x, "       ", y, " = ", r)
        print("\n")
        break
    print("----------------------------------------------------------------------------------------------" * 4)
    return None


@nb.njit(nopython=True, parallel=True)
def f1(shag=0.1) -> (list, list, list, list):
    xx = []
    yy = []
    rez = []
    rez_all = []
    for x in np.arange(0, 11 + shag, shag):
        if x % 1 == 0:
            print(x)
        for y in np.arange(0, 11 + shag, shag):

            # region формула один
            # r1 = (12 * m.sinh(x) * m.cosh(x) * (m.cos(y) * m.cos(y) - m.sin(y) * m.sin(y)) - 16 * 0.1 * m.sinh(
            #     x) * m.cosh(x) * (m.cos(y) * m.cos(y) - m.sin(y) * m.sin(y)) - 4 * x)
            # r2 = (12 * m.cos(y) * m.sin(y) * (m.sinh(x) * m.sinh(x) + m.cosh(x) * m.cosh(x)) - 16 * 0.1 * m.cos(
            #     y) * m.sin(y) * (m.sinh(x) * m.sinh(x) + m.cosh(x) * m.cosh(x)) - 4 * y)
            # # endregion

            # region формула два
            r1 = (6 + 12 * m.sinh(x) * m.sinh(x) * m.cos(y) * m.cos(y) - 12 * m.cosh(x) * m.cosh(x) * m.sin(y) * m.sin(
                y) - 8 * 0.1 - 16 * 0.1 * m.sinh(x) * m.sinh(x) * m.cos(y) * m.cos(y) + 16 * 0.1 * m.cosh(x) * m.cosh(
                x) * m.sin(y) * m.sin(y) + 4 * x * x - 4 * y * y + 2 - 2 * 0.1 + 0.1 * 0.1)
            r2 = (24 * m.cosh(x) * m.sinh(x) * m.sin(y) * m.cos(y) - 32 * 0.1 * m.cosh(x) * m.sinh(x) * m.sin(
                y) * m.cos(
                y) + 8 * x * y)
            # endregion
            r = m.sqrt(r1 ** 2 + r2 ** 2)
            if -0.1 <= r <= 0.1:
                xx.append(x)
                yy.append(y)
                rez.append(r)
            rez_all.append(r)
    print("----------------------------------------------------------------------------------------------" * 4)
    return xx, yy, rez, rez_all


def file_w(xx, yy, rez) -> None:
    with open("rezult.txt", "w") as file:
        for i in range(len(xx)):
            file.write(f"x={xx[i]},  y={yy[i]},  sqrt(r1 ** 2 + r2 ** 2)={rez[i]}\n")
    return None


def stat(rez_all: list) -> None:
    print("----------stat----------")
    rez_all = np.array(rez_all)
    print("количество результатов=", rez_all.shape[0])
    print(f"max={rez_all.max()}")
    print(f"min={rez_all.min()}")
    print(f"avr={mean(rez_all)}")
    col_00 = 0
    col_0 = 0
    col_01 = 0
    col_005 = 0
    col_1 = 0
    for i in rez_all:
        if i < 0:
            col_00 += 1
        elif i == 0:
            col_0 += 1
        elif 0 < i <= 0.05:
            col_005 += 1
            col_01 += 1
        elif 0 < i <= 0.1:
            col_01 += 1
        elif 0 < i <= 1:
            col_1 += 1
    print(
        f'меньше 0 = {col_00}\nравно 0 = {col_0}\nот 0 до 0.05 = {col_005}\nот 0 до 0.1 = {col_01}\nот 0 до 1 = {col_1}\n'
        f'остальных = {rez_all.shape[0] - sum((col_00, col_0, col_01, col_005, col_1))}')

    # xx = [i for i in range(len(rez_all))]
    # x = rez_all.real
    # y = rez_all.imag
    # fig = plt.figure()
    # plt.subplot(121)
    # plt.title("real")
    # plt.grid()
    # plt.plot(xx, x)
    # plt.subplot(122)

    #     # plt.title("img")
    #     # plt.grid()
    #     # plt.plot(xx, y)
    #     # plt.show()
    return None


d = datetime.datetime.now()
# f2()
xx, yy, rez, rez_all = f1()
file_w(xx, yy, rez)
print("len x=", len(xx), "  x=", xx)
print("len y=", len(yy), "  y=", yy)
print("len rez=", len(rez), "  rez=", rez)
stat(rez_all)
d = datetime.datetime.now() - d

# import requests as req
# token = "5226592225:AAGuyEtD_FOotorITU45tTNOLWhEcR2htVA"
# chat_id = 488216212
# r = req.get('https://api.telegram.org/bot' + token + '/sendMessage',
#             params={"chat_id": chat_id, "text": f"закончено вычисление\nзатрачено {d}"})
# print(r)
# print(r.status_code)
