import numpy as np
import matplotlib.pyplot as plt
normal_dist = []
count = 0
with open("my_dist_sp3.txt") as f:
    for line in f:
        count = count+1
        if count <= 13226:
            normal_dist.append(int(line))

plt.hist(normal_dist)
plt.show()

#my_dist = open("my_dist.txt","r")


