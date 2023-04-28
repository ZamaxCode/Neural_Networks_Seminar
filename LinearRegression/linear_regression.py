import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5])
t = np.array([1,2,3,4,5])

w = 0
b = 0
eta = 0.01
max_it = 1000

errors = list()
gradientsW = list()
gradientsB = list()
E = list()

for epoch in range(max_it):
    for k in range(len(x)):
        y= x[k]*w+b
        e = t[k]-y
        errors.append(e)
        if e!=0:
            gw = e * x[k] * eta
            gb = e * eta
            w = w + gw
            b = b + gb
            gradientsW.append(gw)
            gradientsB.append(gb)
    mean = np.array(errors).mean()
    E.append(mean)
    errors.clear()

print('w:',w)
print('b:',b)

j = np.array([6,7,8,9,10])
y = j*w+b
print('y:',y)

plt.plot(list(range(len(E))), E)
#plt.plot(list(range(len(gradientsW))), gradientsW)
#plt.plot(list(range(len(gradientsB))), gradientsB)

plt.show()