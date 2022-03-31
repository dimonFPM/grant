import math
import matplotlib.pyplot as plt
import numpy as np
import numba as nb
from loguru import logger
import warnings
import os
import datetime

warnings.filterwarnings("ignore")


# log1 = logger.add("log.log", level="DEBUG")


@nb.njit(nopython=True, parallel=True)
def skobka(x: float, xi2: float) -> float or None:
    return (x ** 2 - 0.5 * xi2) ** 2


@nb.njit(nopython=True, parallel=True)
def f_xi2(nu: float, xi1: float) -> float or None:
    if nu * 2 != 1:
        return ((2 * (1 - nu)) / (1 - (2 * nu))) * xi1
    else:
        return None


@nb.njit(nopython=True, parallel=True)
def sigma(x: float, xi: float):
    # if type(x) == float and type(xi) == float:
    if x ** 2 > xi:
        return math.sqrt(x ** 2 - xi)
    elif xi > x ** 2:
        return math.sqrt(xi - x ** 2)
    else:
        return 0


@nb.njit(nopython=True, parallel=True)
def finder(x, xi1, xi2, ch):
    res = skobka(x, xi2)
    s1 = sigma(x, xi1)
    s2 = sigma(x, xi2)
    if ch == 1:  # без мнимых единиц
        return res * s1 * s2 * math.cosh(s1) * math.sinh(s2) - x ** 2 * math.sinh(s1) * math.cosh(s2)
    elif ch == 2:  # с мнимой единицей
        # return res * s1 * s2 * math.cosh(s1) * math.sinh(s2) + x ** 2 * math.sinh(s1) * math.cosh(s2)
        return res * (-s1) * s2 * math.cosh(s1) * math.sin(s2) - x ** 2 * math.sinh(s1) * math.cos(s2)


def validate_nu():
    q = input("Введите nu через пробел (все nu должны быть вещественными числами)\nМожно использовать значения по умолчанию (0.1, 0.2, 0.3, 0.4). Для этого введите 'NO':\n")
    if "," in q:
        print("Введите nu через пробел, а не через запятые.\n")
        return None
    q = q.split()
    if len(q) == 1 and q[0].upper() == "NO":
        u: tuple = (0.1, 0.2, 0.3, 0.4)
        return u
    else:
        for i in range(len(q)):
            try:
                q[i] = float(q[i])
            except ValueError:
                print(f"{q[i]} не вещественное число.\n")
                return None
        return tuple(q)


def main(u: tuple = (0.1, 0.2, 0.3, 0.4), xi1_list: tuple = (0.1, 0.5, 1, 2, 3, 4, 5), maxX: int = 10,
         shag: float = 0.01):

    while True:
        u = validate_nu()
        if type(u) == tuple:
            break
    print(*u, sep="\n")
    exit()

    name = datetime.datetime.now().strftime('%d-%m-%Y_%H.%M.%S')
    os.mkdir(f"result_{name}")

    for i in xi1_list:
        xi1 = i
        for j in u:
            nu = j
            xi2 = f_xi2(nu, xi1)
            if xi2 == None:
                logger.debug("Недопустимое значение xi1, возникает деление на  ноль.")
                return False
            col = 0
            col_mnim = 0
            res_x = []
            res_list = []  # список вещественных значений уравнения
            for x in np.arange(0, maxX + shag, shag):
                logger.debug(f"x={x}")
                logger.debug(f"x**2={x ** 2}")
                logger.debug(f"nu={nu}")
                logger.debug(f"xi1={xi1}")
                logger.debug(f"xi2={xi2}")
                if (x ** 2 > xi1 and x ** 2 > xi2):
                    res = finder(x, xi1, xi2, 1)
                    res_x.append(x)
                    res_list.append(res)
                    logger.debug(f"res={res}\n")
                elif (x ** 2 > xi1 and x ** 2 < xi2):
                    res = finder(x, xi1, xi2, 2)
                    res_x.append(x)
                    res_list.append(res)
                    logger.debug(f"res={res}\n")
                else:
                    logger.debug("мнимый корень\n")
                    col_mnim += 1

            col = 0
            for i in res_list:
                if i.imag != 0:
                    col += 1
            logger.debug(f"col-vo imag in res_list={col}")
            logger.debug(f"len res_list={len(res_list)}")
            logger.debug(f"col_mnim={col_mnim}")

            with open(f"result_{name}/resalt_xi1({xi1})_nu({nu})_shag({shag})_maxX({maxX}).txt", "w") as file:
                if len(res_list) == len(res_x):
                    for i in range(len(res_list)):
                        # if -1 < res_list[i] < 1:
                        if -0.5 < res_list[i] < 0.5:
                            # if -0.1 < res_list[i] < 0.1:
                            # if True:
                            file.write(f"x={'%.2f' % res_x[i]}    f(u)={'%.5f' % res_list[i]}\n")
                else:
                    print("списки разной длины")

            # fig = plt.figure(f"график nu={nu} xi1={xi1}")
            # xx = []
            # yy = []
            # for i in range(len(res_list)):
            #     if -1 < res_list[i] < 1:
            #         xx.append(res_x[i])
            #         yy.append(res_list[i])
            # plt.grid()
            # plt.scatter(res_x, res_list, color="r", linewidths=0.5)
            # plt.scatter(xx, yy, color="g", linewidths=0.1)
    return None


if __name__ == '__main__':
    logger.remove()
    logger.debug("старт")
    while True:
        ret = main()
        if ret == None:
            break
        elif ret == False:
            print(
                "Ошибка. Недопустимое значение xi1, возникает деление на  ноль. Попробуйте ещё раз, используя другое значение.")
    plt.show()
