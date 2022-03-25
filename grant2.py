import cmath
import math
import numpy as np
import numba as nb
from loguru import logger
import warnings

warnings.filterwarnings("ignore")
# log1 = logger.add("log.log", level="DEBUG")
# logger.remove()
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


def finder(x, xi1, xi2):
    rez = skobka(x, xi2)
    s1 = sigma(x, xi1)
    s2 = sigma(x, xi2)
    return rez * s1 * s2 * math.cosh(s1) * math.sinh(s2) - x ** 2 * math.sinh(s1) * math.cosh(
        s2)  # пренести в отдельную функцию


def main():
    nu = 0.1
    xi1 = 0.1
    shag = 0.01
    xi2 = f_xi2(nu, xi1)
    if xi2 == None:
        logger.debug("error: деление на ноль")
        exit(500)
    col = 0
    col_mnim = 0
    rez_x = []
    rez_list = []  # список вещественных значений уравнения
    for x in np.arange(0, 10 + shag, shag):
        logger.debug(f"xi2={xi2}")
        logger.debug(f"x={x}")
        if (x ** 2 > xi1 and x ** 2 > xi2) or (x ** 2 < xi1 and x ** 2 < xi2):  # *1
            rez = finder(x, xi1, xi2)
            rez_x.append(x)
            rez_list.append(rez)
            logger.debug(f"x**2={x ** 2}    {rez}\n")
        else:
            logger.debug("мнимый корень\n")
            col_mnim += 1



    col = 0
    for i in rez_list:
        if i.imag != 0:
            col += 1
    logger.debug(f"col-vo imag={col}")
    logger.debug(len(rez_list))
    logger.debug(f"col_mnim={col_mnim}")
    with open(f"rezalt_nu({nu})_xi1({xi1}).txt", "w") as file:
        if len(rez_list) == len(rez_x):
            for i in range(len(rez_list)):
                if rez_list[i] < 1:
                    file.write(f"x={'%.2f' % rez_x[i]}    f(u)={'%.5f' % rez_list[i]}\n")
        else:
            print("списки разной длины")


if __name__ == '__main__':
    main()
