import cmath
import datetime
import math as m
import numpy as np
import numba as nb
from statistics import mean
import matplotlib.pyplot as plt
import requests as req
from decimal import Decimal

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
def f1(u=0.1, shag=0.01) -> (list, list, list, list):
    xx = []
    yy = []
    rez = []
    rez_all = []
    for y in np.arange(0, 10 + shag, shag):
        # y=Decimal(y)
        if y % 1 == 0:
            print(y)
        for x in np.arange(0, 10 + shag, shag):
            # x = Decimal(x)
            # region формула один
            # r1 = (12 * m.sinh(x) * m.cosh(x) * (m.cos(y) * m.cos(y) - m.sin(y) * m.sin(y)) - 16 * 0.1 * m.sinh(
            #     x) * m.cosh(x) * (m.cos(y) * m.cos(y) - m.sin(y) * m.sin(y)) - 4 * x)
            # r2 = (12 * m.cos(y) * m.sin(y) * (m.sinh(x) * m.sinh(x) + m.cosh(x) * m.cosh(x)) - 16 * 0.1 * m.cos(
            #     y) * m.sin(y) * (m.sinh(x) * m.sinh(x) + m.cosh(x) * m.cosh(x)) - 4 * y)
            # # endregion

            # region формула два
            r1 = (6 + 12 * m.sinh(x) * m.sinh(x) * m.cos(y) * m.cos(y) - 12 * m.cosh(x) * m.cosh(x) * m.sin(y) * m.sin(
                y) - 8 * u - 16 * u * m.sinh(x) * m.sinh(x) * m.cos(y) * m.cos(y) + 16 * u * m.cosh(x) * m.cosh(
                x) * m.sin(y) * m.sin(y) + 4 * x * x - 4 * y * y + 2 - 2 * u + u * u)
            r2 = (24 * m.cosh(x) * m.sinh(x) * m.sin(y) * m.cos(y) - 32 * u * m.cosh(x) * m.sinh(x) * m.sin(
                y) * m.cos(y) + 8 * x * y)
            # endregion

            # r1 = (6 + 12 * Decimal(m.sinh(x)) * Decimal(m.sinh(x)) * Decimal(m.cos(y)) * Decimal(m.cos(y)) - 12 * Decimal(m.cosh(x)) * Decimal(m.cosh(x)) * Decimal(m.sin(y)) * Decimal(m.sin(
            #     y)) - 8 * u - 16 * u * Decimal(m.sinh(x)) * Decimal(m.sinh(x)) * Decimal(m.cos(y)) * Decimal(m.cos(y)) + 16 * u * Decimal(m.cosh(x)) * Decimal(m.cosh(
            #     x)) * Decimal(m.sin(y)) * Decimal(m.sin(y)) + 4 * x * x - 4 * y * y + 2 - 2 * u + u * u)
            # r2 = (24 * Decimal(m.cosh(x)) * Decimal(m.sinh(x)) * Decimal(m.sin(y)) * Decimal(m.cos(y)) - 32 * u * Decimal(m.cosh(x)) * Decimal(m.sinh(x)) * Decimal(m.sin(
            #     y)) * Decimal(m.cos(y)) + 8 * x * y)

            r = m.sqrt(r1 ** 2 + r2 ** 2)
            if -1 < r < 0.5:
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


def stat(rez_all: list, grath: bool = False, time=None, otc_check: bool = False) -> None:
    rez_all = np.array(rez_all)
    col_00 = 0
    col_0 = 0
    col_01 = 0
    col_005 = 0
    col_1 = 0
    col_5 = 0
    col_10 = 0
    for i in rez_all:
        if i < 0:
            col_00 += 1
        elif i == 0:
            col_0 += 1
        elif 0 < i <= 0.05:
            col_005 += 1
        elif 0.05 < i <= 0.1:
            col_01 += 1
        elif 0.1 < i <= 1:
            col_1 += 1
        elif 1 < i <= 5:
            col_5 += 1
        elif 5 < i <= 10:
            col_10 += 1
    print("----------stat----------")
    print("количество результатов=", rez_all.shape[0])
    print(f"max={rez_all.max()}")
    print(f"min={rez_all.min()}")
    # print(f"avr={mean(rez_all)}")
    print(
        f'меньше 0 = {col_00}\n'
        f'равно 0 = {col_0}\n'
        f'от 0 до 0.05 = {col_005}\n'
        f'от 0.05 до 0.1 = {col_01}\n'
        f'от 0.1 до 1 = {col_1}\n'
        f'от 1 до 5 = {col_5}\n'
        f'от 5 до 10 = {col_10}\n'
        f'остальные = {rez_all.shape[0] - sum((col_00, col_0, col_01, col_005, col_1, col_5, col_10))}'
    )
    if grath == True:
        maxx = max((col_00, col_0, col_01, col_005, col_1, col_5, col_10))
        fig_hist = plt.figure("Распределение")
        plt.ylim((0, maxx + maxx * 0.1))
        plt.minorticks_on()
        plt.grid(which="major", linewidth=1)
        plt.grid(which="minor", linestyle=":", color="gray")
        plt.bar(("<0", "0", "0-0.05", "0.05-0.1", "0.1-1", "1-5", "5-10"),
                (col_00, col_0, col_01, col_005, col_1, col_5, col_10))

    if otc_check == True:
        str_otc = "Программа закончена\n" + "----------stat----------\n" + "количество результатов=" + str(
            rez_all.shape[0]) + "\n" + \
                  f"max={rez_all.max()}\n" + f"min={rez_all.min()}\n" + \
                  f'меньше 0 = {col_00}\n' + f'равно 0 = {col_0}\nот 0 до 0.05 = {col_005}\nот 0.05 до 0.1 = {col_01}\n' \
                                             f'от 0.1 до 1 = {col_1}\nот 1 до 5 = {col_5}\nот 5 до 10 = {col_10}\n' \
                                             f'остальные = {rez_all.shape[0] - sum((col_00, col_0, col_01, col_005, col_1, col_5, col_10))}'
        if time != None:
            str_otc += f"\nВремя выполнения {time}"
        try:
            token = "5226592225:AAGuyEtD_FOotorITU45tTNOLWhEcR2htVA"
            r = req.get("https://api.telegram.org/bot" + token + "/sendMessage",
                        params={"chat_id": 488216212, "text": str_otc})
            if r.status_code != 200:
                print(r.status_code + "  Отчёт не отправлен")
            else:
                print("Отчёт отправлен")
        except:
            print("Ошибка при отправке")
    if grath == True:
        plt.show()

    return None


t = datetime.datetime.now()
# f2()
xx, yy, rez, rez_all = f1()
t = datetime.datetime.now() - t
print("time=", t)
file_w(xx, yy, rez)
print("len x=", len(xx), "  x=", xx)
print("len y=", len(yy), "  y=", yy)
print("len rez=", len(rez), "  rez=", rez)
d = datetime.datetime.now()
stat(rez_all, time=t)
stat(rez, True)
d = datetime.datetime.now() - d
print("time=", d)
