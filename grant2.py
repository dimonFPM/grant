import cmath
import math
import numpy as np
import numba as nb

import warnings

warnings.filterwarnings("ignore")


@nb.njit(nopython=True, parallel=True)
def skobka(x: float, xi2: float) -> float or None:
    # if type(x) == float and type(xi2) == float:
    return (x ** 2 - 0.5 * xi2) ** 2
    # else:
    #     return None


@nb.njit(nopython=True, parallel=True)
def f_xi2(nu: float, xi1: float) -> float or None:
    # if type(nu) == float and type(xi1) == float:
    if nu * 2 != 1:
        return ((2 * (1 - nu)) / (1 - (2 * nu))) * xi1
    else:
        print("error: деление на ноль")
        return None
    # else:
    #     return None


@nb.njit(nopython=True, parallel=True)
def sigma(x: float, xi: float):
    # if type(x) == float and type(xi) == float:
    if x ** 2 > xi:
        print("ветка 1")
        return math.sqrt(x ** 2 - xi)
    elif xi > x ** 2:
        print("ветка 2")
        return math.sqrt(xi - x ** 2)
    else:
        print("ветка 3")
        print("Равно")

    # else:
    #     return None


def main():
    nu = 0.1
    xi1 = 0.1
    shag = 0.01
    xi2 = f_xi2(nu, xi1)
    col = 0
    for x in np.arange(0, 10 + shag, shag):
        s1 = sigma(x, xi1)
        s2 = sigma(x, xi2)
        # rez = skobka(x, xi2)
        rez = 1
        if (x ** 2 > xi1 and x ** 2 > xi2) or (x ** 2 < xi1 and x ** 2 < xi2):  # *1
            rez = rez * s1 * s2 * math.cosh(s1) * math.sinh(s2)
        elif (x ** 2 > xi1 and x ** 2 < xi2) or (x ** 2 < xi1 and x ** 2 > xi2):  # *-1
            rez = rez * s1 * s2 * math.cosh(s1) * math.sinh(s2) * (-1)

        if rez.imag != 0:
            col += 1
        print(x)
        print(f"x={x ** 2}    {rez}\n")
    print(col)


if __name__ == '__main__':
    main()
    # # print(skobka(10.0,10.0))
    # # print(f_xi2(10.0,10.0))
    # # print(sigma(0.1,5))
    # nu = 0.1
    # xi1 = 0.1
    # shag = 0.01
    # rez = []
    # col = 0
    # for x in np.arange(0, 10 + shag, shag):
    #     # z = sigma(x, xi1) * sigma(x, f_xi2(nu, xi1)) * cmath.cosh(sigma(x, xi1)) * cmath.sinh(sigma(x, f_xi2(nu, xi1)))
    #
    #     # z = sigma(x, f_xi2(nu, xi1)) * cmath.sinh(sigma(x, f_xi2(nu, xi1)))
    #     z = sigma(x, xi1) * cmath.cosh(sigma(x, xi1))
    #     # z = cmath.cosh(sigma(x, xi1)) * cmath.sinh(sigma(x, f_xi2(nu, xi1)))
    #
    #     # z = sigma(x, xi1)
    #     # z = sigma(x, f_xi2(nu, xi1))
    #     # z = cmath.cosh(sigma(x, xi1))
    #     # z = cmath.sinh(sigma(x, f_xi2(nu, xi1)))
    #     print(f"xi={f_xi2(nu, xi1)}")
    #
    #     if z.imag != 0:
    #         col += 1
    #     print(x)
    #     print(f"x={x ** 2}    {z}\n")
    #
    #     rez.append(z)
    # print(col)
    # # rez.sort(key=lambda i: i.imag)
    # # print(*rez, sep="\n")
