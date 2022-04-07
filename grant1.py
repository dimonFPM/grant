import datetime
import math as m
import os

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import requests as req


@nb.njit(nopython=True, parallel=True)
def finder(u=0.1, shag=0.001, start_y=0) -> (list, list, list, list):
    p = "Поиск нулей"
    print(p.center(40, "_"))
    xx = []
    yy = []
    rez = []
    for y in np.arange(start_y, start_y + 30 + shag, shag):
        if y % 1 == 0:
            print("y = ", y)
        for x in np.arange(0, 30 + shag, shag):
            # region формула один
            # r1 = (12 * m.sinh(x) * m.cosh(x) * (m.cos(y) * m.cos(y) - m.sin(y) * m.sin(y)) - 16 * u * m.sinh(
            #     x) * m.cosh(x) * (m.cos(y) * m.cos(y) - m.sin(y) * m.sin(y)) - 4 * x)
            # r2 = (12 * m.cos(y) * m.sin(y) * (m.sinh(x) * m.sinh(x) + m.cosh(x) * m.cosh(x)) - 16 * u * m.cos(
            #     y) * m.sin(y) * (m.sinh(x) * m.sinh(x) + m.cosh(x) * m.cosh(x)) - 4 * y)
            # # endregion

            # region формула два
            r1 = (6 + 12 * m.sinh(x) * m.sinh(x) * m.cos(y) * m.cos(y) - 12 * m.cosh(x) * m.cosh(x) * m.sin(y) * m.sin(
                y) - 8 * u - 16 * u * m.sinh(x) * m.sinh(x) * m.cos(y) * m.cos(y) + 16 * u * m.cosh(x) * m.cosh(
                x) * m.sin(y) * m.sin(y) + 4 * x * x - 4 * y * y + 2 - 2 * u + u * u)
            r2 = (24 * m.cosh(x) * m.sinh(x) * m.sin(y) * m.cos(y) - 32 * u * m.cosh(x) * m.sinh(x) * m.sin(
                y) * m.cos(y) + 8 * x * y)
            # endregion

            r = m.sqrt(r1 ** 2 + r2 ** 2)
            if -1 < r < 1:
                xx.append(x)
                yy.append(y)
                rez.append(r)
    print("-" * 50)
    return xx, yy, rez


def file_writer_raw_rezalt(xx, yy, rez, name_tuple: tuple) -> str:
    file_name = f"{name_tuple[0]}/raw_data/result_{name_tuple[1]}.txt"
    with open(file_name, "w") as file:
        for i in range(len(xx)):
            file.write(f"x={xx[i]},  y={yy[i]},  sqrt(r1 ** 2 + r2 ** 2)={rez[i]}\n")
    return file_name


def file_writer_group_rezalt(res, name_tuple: tuple):
    group_nomer = 1
    for group in res:
        # print("group")
        # print(group)
        # exit()
        file_name = f"{name_tuple[0]}/group_data/result_{name_tuple[1]}_group_{group_nomer}.txt"
        with open(file_name, "w") as file:
            for i in group:
                # for i in sorted(group, key=lambda x: x[2]):
                file.write(f"{i}\n")
            # file.write(f"x={xx[group_nomer]},  y={yy[group_nomer]},  sqrt(r1 ** 2 + r2 ** 2)={res[group_nomer]}\n")
        group_nomer += 1


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
        # f'от 1 до 5 = {col_5}\n'
        # f'от 5 до 10 = {col_10}\n'
        # f'остальные = {rez_all.shape[0] - sum((col_00, col_0, col_01, col_005, col_1, col_5, col_10))}'
        f''
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


def group(name_file: str, changes: float):
    col_vo_znakov = int(len(str(changes).split(".")[1]))
    l = []
    ones = []
    with open(name_file, "r") as file:
        for i in file:
            l.append(tuple(
                map(float,
                    (i.removesuffix("\n").replace("x=", "").replace("y=", "").replace("sqrt(r1 ** 2 + r2 ** 2)=",
                                                                                      "").replace(" ", "").split(
                        ",")))))
    l = tuple(l)
    for i in range(len(l)):
        if len(ones) == 0:
            ones.append([l[i]])
        else:
            # if abs(float(f"%.{col_vo_znakov}f" % (l[i][0] - ones[-1][-1][0]))) <= changes and float(
            #         f"%.{col_vo_znakov}f" % (l[i][1] - ones[-1][-1][1])) <= changes:
            if float(f"%.{col_vo_znakov}f" % (l[i][1] - ones[-1][-1][1])) <= changes:
                ones[-1].append(l[i])
            else:
                ones.append([l[i]])
    print(f"Выделено нулей - {len(ones)}.")

    ones_group = ones[:]

    ones = [sorted(i, key=lambda l: l[2])[0] for i in ones]
    print("Нули функции:")
    for i in ones:
        print(f"x = {'%.4f' % i[0]}   y = {'%.4f' % i[1]}   rez={'%.4f' % i[2]}")

    return ones_group


def main():
    directory_name = f"result_{datetime.datetime.now().strftime('%d.%m.%Y_%H-%M-%S')}"
    os.mkdir(directory_name)
    os.mkdir(f"{directory_name}/raw_data")
    os.mkdir(f"{directory_name}/group_data")
    # exit()
    shag = 0.001
    # nu = (0.1, 0.2, 0.3, 0.4)
    nu = (0.1,)
    for u in nu:
        t = datetime.datetime.now()
        xx, yy, rez = finder(shag=shag, u=u)
        t = datetime.datetime.now() - t
        print("Время поиска = ", t)
        rezalt_file_name = file_writer_raw_rezalt(xx, yy, rez, (directory_name, u))
        stat(rez)

        group_result_list = group(rezalt_file_name, shag)
        file_writer_group_rezalt(group_result_list, (directory_name, u))

        # break


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        er = "Ошибка: " + str(error) + "  " + str(error.__traceback__.tb_frame)
        print(er)
