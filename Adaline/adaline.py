import numpy as np
import matplotlib.pyplot as plt

x = np.array([
    [1, 0.5, 0.9],
    [1, 0.3, 0.8],
    [1, 0.2, 0.95],
    [1, 0.7, 0.1],
    [1, 0.9, 0.2],
    [1, 0.8, 0.3],
])

t = np.array([
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
])


_, C = x.shape
w = np.ones((1, C))
eta = 0.01


def step(inputs):
    if inputs >= 0:
        return 1
    return 0

def activation_func(x, der):
    if der:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))

epoch = 0
max_epoch = 1000
e_min = 0.001
error = 1

while error > e_min and epoch < max_epoch:
    E=0
    for k in range(len(x)):
        v = np.dot(w, x[k, :])
        y = activation_func(v, 0)
        y_der = activation_func(y, 1)
        e = t[k] - y
        w = w + ( x[k, :] * e * y_der * eta )
        E = E + e**2
    error = E/len(x)
    epoch = epoch + 1


x1 = x[...,1]
x2 = x[...,2]

plt.scatter(x=x1, y=x2)

plt.xlim(-2,2)
plt.ylim(-2,2)

for i in range(len(w)):
    a=w[i,1]
    b=w[i,2]
    c=w[i,0]
    y1 = (-c-(a*-1.5))/b
    y2 = (-c-(a*1.5))/b
    plt.plot([-1.5,1.5], [y1,y2])

plt.show()