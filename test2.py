import cmath

import numpy as np
import matplotlib.pyplot as plt
import math
import grant2 as gr
from loguru import logger

logger.remove()


def x2_or_skobka(xi1=0.1):
    ''' xi1 = (0.1, 0.5, 1, 2, 3, 4, 5) '''
    u = (0.1, 0.2, 0.3, 0.4)
    for nu in u:
        xi2 = (2 * (1 - nu)) / (1 - 2 * nu) * xi1
        x_list = []
        y_list1 = []
        y_list2 = []
        xx_list = []
        yy_list = []
        sigma1_list = []
        sigma2_list = []
        for x in np.arange(0, 10.01, 0.01):
            if (x ** 2 > xi1 and x ** 2 > xi2) or (x ** 2 < xi1 and x ** 2 < xi2):  # !!!!!!!!!!!!!!!!!!
                x2 = x ** 2
                s1 = gr.sigma(x, xi1)
                s2 = gr.sigma(x, xi2)
                skob = (x2 - 0.5 * xi2) ** 2
                y_list1.append(skob * s1 * s2)
                # y_list1.append(skob)
                y_list2.append(x2)
                x_list.append(x)
                if x2 > skob:
                    xx_list.append(x)
                    yy_list.append(-1)

        fig = plt.figure(f"сравнение nu={nu} xi1={xi1} xi2={xi2}")

        # plt.minorticks_on()
        # plt.grid(which="minor", color="gray",)

        plt.grid(which="major")
        # plt.xlim((0, 2.2))
        # plt.plot(x_list, y_list1, "r", label="(x**2 - 0.5 * xi2) ** 2")
        # plt.plot(x_list, y_list2, "b", label="x**2")

        plt.scatter(x_list, y_list1, color="r", label="(x**2 - 0.5 * xi2) ** 2", linewidths=0.001)
        plt.scatter(x_list, y_list2, color="b", label="x**2", linewidths=0.001)

        plt.scatter(xx_list, yy_list, linewidths=0.001, color="green")
        plt.legend()
        print(f"количество x2 > skob={len(xx_list)}")
    plt.show()


def sinh_cosh_grath():
    x_list = tuple(np.arange(-5, 10.01, 0.01))
    y_sinh_list = []
    y_cosh_list = []
    change_list = []
    for x in x_list:
        y_cosh_list.append(math.cosh(x))
        y_sinh_list.append(math.sinh(x))
        if math.cosh(x) == math.sinh(x):
            change_list.append(x)
    fig = plt.figure()
    plt.grid()
    plt.plot(x_list, y_cosh_list, "r", label="cosh")
    plt.plot(x_list, y_sinh_list, "b", label="sinh")
    plt.scatter(change_list, np.zeros_like(change_list), color="g", linewidths=0.5)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # x2_or_skobka()
    # sinh_cosh_grath()
    x = 10
    x2 = 0.0001
    nu = 0.1
    xi1 = 0.1
    xi2 = 0.225
    s1 = gr.sigma(x, xi1)
    s2 = gr.sigma(x, xi2)
    res = ((x ** 2 - 0.5 * xi2) ** 2) * complex(0, s1) * complex(0, s2) * cmath.cosh(complex(0, s1)) * cmath.sinh(
        complex(0, s2)) - x ** 2 * cmath.sinh(complex(0, s1)) * cmath.cosh(complex(0, s2))

    print("%.7f" % res.real)
    print(res.imag)
