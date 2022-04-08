import matplotlib.pyplot as plt

l = []
with open("", "r") as file:
    s = file.read()

l = s.replace("x=", "").replace("y=", "").replace("sqrt(r1 ** 2 + r2 ** 2)=", "").replace(",", "").split("\n")
l = [list(map(float, i.split())) for i in l]
l.pop()
l = sorted(l, key=lambda x: x[2])
# print(l)
print(*l, sep="\n")
