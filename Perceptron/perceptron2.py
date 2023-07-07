import numpy as np
import matplotlib.pyplot as plt

x = np.array([
    [0.5, 0.9, 1],
    [0.3, 0.8, 1],
    [0.2, 0.95, 1],
    [0.7, 0.1, 1],
    [0.9, 0.2, 1],
    [0.8, 0.3, 1],
    [-0.7, 0.1, 1],
    [-0.9, 0.2, 1],
    [-0.8, 0.3, 1],
    [-0.7, -0.1, 1],
    [-0.9, -0.2, 1],
    [-0.8, -0.3, 1]
])

t = np.array([
    [0,0],
    [0,0],
    [0,0],
    [0,1],
    [0,1],
    [0,1],
    [1,0],
    [1,0],
    [1,0],
    [1,1],
    [1,1],
    [1,1]
])


_, C = x.shape
w = np.ones((2, C))
eta = 0.01


def step(inputs):
    if inputs >= 0:
        return 1
    return 0

f = False
s = 0
max_itr = 1000

while f is False and s < max_itr:
    f = True
    for k in range(len(x)):
        v = np.dot(x[k, :], w.T)
        y1 = step(v[0])
        y2 = step(v[1])
        y=[y1,y2]
        e = t[k] - y

        if t[k,0] != y[0] or t[k,1] != y[1]:
            w = w + (np.dot(np.array([x[k, :]]).T, np.array([e])) * eta).T
            f = False
    s = s + 1

print(s)
print(w)

x1 = x[...,0]
x2 = x[...,1]

plt.xlim(-2,2)
plt.ylim(-2,2)
plt.scatter(x=x1, y=x2)

for i in range(len(w)):
    #ax, by, c = 0
    a=w[i,0]
    b=w[i,1]
    c=w[i,2]
    y1 = (-c-(a*-1.5))/b
    y2 = (-c-(a*1.5))/b
    plt.plot([-1.5,1.5], [y1,y2])

plt.show()

'''
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

f = False
s = 0
max_itr = 1000

while f is False and s < max_itr:
    f = True
    for k in range(len(x)):
        v = np.dot(w, x[k, :])
        y = step(v)
        e = t[k] - y
        if t[k] != y:
            w = w + (x[k, :] * e * eta)
            f = False
    s = s + 1

print(x[0, :])


x1 = x[...,1]
x2 = x[...,2]


plt.scatter(x=x1, y=x2)

plt.xlim(-2,2)
plt.ylim(-2,2)

for i in range(len(w)):
    a=w[i,0]
    b=w[i,1]
    c=w[i,2]
    y1 = (-c-(a*-1.5))/b
    y2 = (-c-(a*1.5))/b
    plt.plot([-1.5,1.5], [y1,y2])

plt.show()
'''