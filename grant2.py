import math
import cmath
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import numba as nb
from loguru import logger
import warnings
import os
import datetime

warnings.filterwarnings("ignore")

sigma_list = []


@nb.njit(nopython=True, parallel=True)
def skobka(x: float, xi2: float) -> float or None:
    return (x - 0.5 * xi2) ** 2


@nb.njit(nopython=True, parallel=True)
def f_xi2(nu: float, xi1: float) -> float or None:
    if nu * 2 != 1:
        return ((2 * (1 - nu)) / (1 - (2 * nu))) * xi1
    else:
        return None


# @nb.njit(nopython=True, parallel=True)
def sigma(x: float, xi: float):
    if x > xi:
        logger.debug("ветка 1")
        return math.sqrt(x - xi)
    elif xi > x:
        logger.debug("ветка 2")
        return complex(0, 1) * math.sqrt(xi - x)
    else:
        logger.debug("ветка 3")
        return 0


# # @nb.njit(nopython=True, parallel=True)
# def finder(x, xi1, xi2, ch):
#     global sigma_list
#     res = skobka(x, xi2)
#     logger.debug(f"skobka={(x - 0.5 * xi2) ** 2}")
#     s1 = sigma(x, xi1)
#     logger.debug(f"s1={s1}")
#     s2 = sigma(x, xi2)
#     sigma_list.append(("s1", s1))
#     sigma_list.append(("s2", s2))
#     logger.debug(f"s2={s2}")
#     logger.debug(f"t1={s1 * s2 * math.cosh(s1) * math.sinh(s2)}")
#     logger.debug((f"t2={x ** 2 * math.sinh(s1) * math.cosh(s2)}"))
#     if ch == 1:  # без мнимых единиц
#         return res * s1 * s2 * math.cosh(s1) * math.sinh(s2) - x ** 2 * math.sinh(s1) * math.cosh(s2)
#     elif ch == 2:  # с мнимой единицей
#         # return res * s1 * s2 * math.cosh(s1) * math.sinh(s2) + x ** 2 * math.sinh(s1) * math.cosh(s2)
#         return res * (-s1) * s2 * math.cosh(s1) * math.sin(s2) - x ** 2 * math.sinh(s1) * math.cos(s2)

# @nb.njit(nopython=True, parallel=True)
def finder2(x, xi1, xi2):
    global sigma_list
    res = skobka(x, xi2)

    logger.debug(f"skobka={skobka(x, xi2)}")
    s1 = sigma(x, xi1)
    logger.debug(f"s1={s1}")
    s2 = sigma(x, xi2)

    #######################
    sigma_list.append(("s1", s1))
    sigma_list.append(("s2", s2))
    #######################

    logger.debug(f"s2={s2}")

    return res * s1 * s2 * cmath.cosh(s1) * cmath.sinh(s2) - x * cmath.sinh(s1) * cmath.cosh(s2)


def validate_nu():
    q = input(
        "Введите nu через пробел (все nu должны быть вещественными числами)\nМожно использовать значения по умолчанию (0.1, 0.2, 0.3, 0.4). Для этого введите 'NO':\n")
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


def group(res_x: list, res_list: list, change=0.01):
    l_list = []
    if len(res_list) == 0:
        print("Нет корней, список пустой")
        return None
    for i in range(len(res_x)):
        if len(l_list) == 0:
            l_list.append([])
            l_list[-1].append([res_x[i], complex(abs(res_list[i].real), abs(res_list[i].imag))])
        else:
            if float("%.5f" % (res_x[i] - l_list[-1][-1][0])) <= change:
                l_list[-1].append([res_x[i], complex(abs(res_list[i].real), abs(res_list[i].imag))])
            else:
                l_list.append([])
                l_list[-1].append([res_x[i], complex(abs(res_list[i].real), abs(res_list[i].imag))])
    return l_list


def otchet_writer(name, shag, maxX, okr):
    name_list = os.listdir(f"{name}/group_data")
    otc_text = f"Нули находились при: x=u^2, x от 0 до {maxX} с шагом {shag}. За ноль берутся значения функции от -{okr} до {okr}.\n"
    for i in name_list:
        xi1, nu = list(map(float, i.replace("resalt_xi1(", "").replace(")_nu(", " ").replace(")_shag(", " ").
                           replace(")_maxX(", " ").replace(").txt", "").split()))[0:2]
        otc_text = otc_text + f"При xi1={xi1} nu={nu}:\n"
        # print(otc_text)
        with open(f"{name}/group_data/{i}", "r") as file:
            for j in file:
                u2, fu2 = j.split()
                # otc_text = otc_text + f"    {u2}  u={math.sqrt(float(u2.split('=')[1]))}  {fu2}\n"
                otc_text = otc_text + f"    {u2}  u={math.sqrt(float(u2.split('=')[1]))}\n"
    with open(f"{name}/otchet.txt", "w") as file:
        file.write(otc_text)


def main():
    # while True:
    #     u = validate_nu()
    #     if type(u) == tuple:
    #         break

    name = datetime.datetime.now().strftime('%d-%m-%Y_%H.%M.%S')
    os.mkdir(f"result_{name}")
    os.mkdir(f"result_{name}/raw_data")
    os.mkdir(f"result_{name}/group_data")

    okr = 0.5
    maxX = 10
    shag = 0.001
    # u = (0.3,)
    # xi1_list = (0.1,)
    u = (0.1, 0.2, 0.3, 0.4)
    xi1_list = (0.1, 0.5, 1, 2, 3, 4, 5)
    for i in tqdm(xi1_list):
        xi1 = i
        for j in u:
            nu = j
            xi2 = f_xi2(nu, xi1)
            if xi2 == None:
                logger.debug("error: деление на ноль")
                return False
            col = 0
            col_mnim = 0
            res_x = []
            res_list = []  # список вещественных значений уравнения
            for x in np.arange(0, maxX + shag, shag):
                # for x in (6.34,):
                logger.debug(f"x={x}")
                logger.debug(f"x**2={x ** 2}")
                logger.debug(f"nu={nu}")
                logger.debug(f"xi1={xi1}")
                logger.debug(f"xi2={xi2}")

                res = finder2(x, xi1, xi2)
                res_x.append(x)
                res_list.append(res)
                logger.debug(f"res={res}\n")

            col = 0
            for i in res_list:
                if i.imag != 0:
                    col += 1
            logger.debug(f"col-vo imag in res_list={col}")
            logger.debug(f"len res_list={len(res_list)}")
            logger.debug(f"col_mnim={col_mnim}")

            xxx_list = []
            yyy_list = []

            with open(f"result_{name}/raw_data/resalt_raw_xi1({xi1})_nu({nu})_shag({shag})_maxX({maxX}).txt",
                      "w") as file:
                if len(res_list) == len(res_x):
                    for i in range(len(res_list)):
                        # if True:
                        #     file.write(f"x={res_x[i]}    f(u)={res_list[i]}\n")
                        if (-okr < res_list[i].imag < okr) and (-okr < res_list[i].real < okr):
                            file.write(f"u^2={res_x[i]}    f(u^2)={res_list[i]}\n")
                            xxx_list.append(res_x[i])
                            yyy_list.append(res_list[i])
                else:
                    print("списки разной длины")

            l = group(xxx_list, yyy_list, shag)
            l1 = []
            # print("------------------------")
            for ii in range(len(l)):
                min_real = [100, complex(10, 10)]
                min_imag = [100, complex(10, 10)]
                for jj in range(len(l[ii])):
                    if l[ii][jj][1] == complex(0, 0):
                        min_imag[0] = l[ii][jj][0]
                        min_imag[1] = l[ii][jj][1]
                        min_real[0] = l[ii][jj][0]
                        min_real[1] = l[ii][jj][1]
                        break
                    elif l[ii][jj][1].real == 0 and l[ii][jj][1].imag < min_imag[1].imag:
                        min_imag[0] = l[ii][jj][0]
                        min_imag[1] = l[ii][jj][1]
                    elif l[ii][jj][1].imag == 0 and l[ii][jj][1].real < min_real[1].real:
                        min_real[0] = l[ii][jj][0]
                        min_real[1] = l[ii][jj][1]
                if min_real[0] != 100 and min_imag[0] != 100:
                    l1.append(min_real)
                elif min_real[0] != 100:
                    l1.append(min_real)
                elif min_imag[0] != 100:
                    l1.append(min_imag)
            # print(l1)

            with open(f"result_{name}/group_data/resalt_xi1({xi1})_nu({nu})_shag({shag})_maxX({maxX}).txt",
                      "w") as file:
                for i in range(len(l1)):
                    file.write(f"u^2={'%.5f' % l1[i][0]}   f(u^2)={l1[i][1]}\n")
    otchet_writer(f"result_{name}", shag, maxX, okr)
    return None


if __name__ == '__main__':

    logger.remove()
    # log = logger.add("log.log", level="DEBUG", )
    # logger.remove()

    logger.debug("старт")

    while True:
        ret = main()
        if ret == None:
            break
        elif ret == False:
            print(
                "Ошибка. Недопустимое значение nu, возникает деление на  ноль. Попробуйте ещё раз, используя другое значение.")
