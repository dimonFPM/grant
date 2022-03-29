import math
from tqdm import tqdm


i = 100000000
l = []
for j in tqdm(range(i)):
    l.append(i * 12 * math.sqrt(100))
