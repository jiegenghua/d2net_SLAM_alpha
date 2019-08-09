import numpy as np
import matplotlib.pyplot as plt
normal_dist = []
count = 0

with open("normal_chi2.txt") as f:
    for line in f:
        count = count+1
        if count <= 14983:
            normal_dist.append(float(line))
            #print(float(line))
x = np.arange(14983)
print(max(normal_dist))
plt.scatter(x,normal_dist)
plt.show()

#my_dist = open("my_dist.txt","r")


