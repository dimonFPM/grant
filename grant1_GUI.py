import tkinter as tk
import grant



class Calk:
    def __init__(self):
        self.app = tk.Tk()
        self.app.title("Поиск корней")
        self.app.geometry("300x200+100+100")
        # app.resizable(True,True)

        tk.Label(self.app, text="Выберете нужную формулу:", width=30, bg="orange").pack(side=tk.TOP)
        self.l = tk.Label(self.app, text="Параметр группировки: ", width=30, bg="orange")
        self.e = tk.Entry(self.app, justify=tk.CENTER)

        self.r = tk.IntVar()
        self.r.set(0)
        self.r_but1 = tk.Radiobutton(self.app, text="формула 1", variable=self.r, value=0)
        self.r_but2 = tk.Radiobutton(self.app, text="формула 2", variable=self.r, value=1)

        self.b_action = tk.Button(self.app, text="Найти", bg="orange", command=self.action)

        # self.check_stop=tk.BooleanVar()
        # self.check_stop.set(False)
        # self.b_stop=tk.Button(self.app,text="Стоп",command=lambda )

        self.pack_config(self.l, self.e, self.r_but1, self.r_but2, self.b_action)

        self.app.mainloop()

    def pack_config(self, *l: tuple):
        for i in l:
            i.pack(side=tk.TOP)

    def action(self):  ###################?
        try:
            grant.main()
        except Exception as error:
            er = "Ошибка: " + str(error) + "  " + str(error.__traceback__.tb_frame)
            print(er)


if __name__ == '__main__':
    x = Calk()
