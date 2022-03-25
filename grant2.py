import math
import numpy as np
import numba as nb
from loguru import logger
import warnings
import os
import datetime

warnings.filterwarnings("ignore")
# log1 = logger.add("log.log", level="DEBUG")
logger.remove()
logger.debug("старт")


@nb.njit(nopython=True, parallel=True)
def skobka(x: float, xi2: float) -> float or None:
    return (x ** 2 - 0.5 * xi2) ** 2


@nb.njit(nopython=True, parallel=True)
def f_xi2(nu: float, xi1: float) -> float or None:
    if nu * 2 != 1:
        return ((2 * (1 - nu)) / (1 - (2 * nu))) * xi1
    else:
        return None


# @nb.njit(nopython=True, parallel=True)
def sigma(x: float, xi: float):
    # if type(x) == float and type(xi) == float:
    if x ** 2 > xi:
        logger.debug("ветка 1")
        return math.sqrt(x ** 2 - xi)
    elif xi > x ** 2:
        logger.debug("ветка 2")
        return math.sqrt(xi - x ** 2)
    else:
        logger.debug("ветка 3")
        return 0

# @nb.njit(nopython=True, parallel=True)
def finder(x, xi1, xi2):
    res = skobka(x, xi2)
    s1 = sigma(x, xi1)
    s2 = sigma(x, xi2)
    return res * s1 * s2 * math.cosh(s1) * math.sinh(s2) - x ** 2 * math.sinh(s1) * math.cosh(
        s2)  # пренести в отдельную функцию


def main():
    name = datetime.datetime.now().strftime('%d-%m-%Y_%H.%M.%S')
    os.mkdir(f"result_{name}")

    maxX = 10
    shag = 0.01
    u = (0.1, 0.2, 0.3, 0.4)
    xi1_list = (0.1, 0.5, 1, 2, 3, 4, 5)
    for i in xi1_list:
        xi1 = i
        for j in u:
            nu = j
            xi2 = f_xi2(nu, xi1)
            if xi2 == None:
                logger.debug("error: деление на ноль")
                exit(500)
            col = 0
            col_mnim = 0
            res_x = []
            res_list = []  # список вещественных значений уравнения
            for x in np.arange(0, maxX + shag, shag):
                logger.debug(f"xi2={xi2}")
                logger.debug(f"x={x}")
                if (x ** 2 > xi1 and x ** 2 > xi2) or (x ** 2 < xi1 and x ** 2 < xi2):  # *1
                    rez = finder(x, xi1, xi2)
                    res_x.append(x)
                    res_list.append(rez)
                    logger.debug(f"x**2={x ** 2}    {rez}\n")
                else:
                    logger.debug("мнимый корень\n")
                    col_mnim += 1

            col = 0
            for i in res_list:
                if i.imag != 0:
                    col += 1
            logger.debug(f"col-vo imag={col}")
            logger.debug(len(res_list))
            logger.debug(f"col_mnim={col_mnim}")

            with open(f"result_{name}/resalt_xi1({xi1})_nu({nu})_shag({shag})_maxX({maxX}).txt", "w") as file:
                if len(res_list) == len(res_x):
                    for i in range(len(res_list)):
                        if -1 < res_list[i] < 1:
                            file.write(f"x={'%.2f' % res_x[i]}    f(u)={'%.5f' % res_list[i]}\n")
                else:
                    print("списки разной длины")


if __name__ == '__main__':
    main()
