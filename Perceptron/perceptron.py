import numpy as np

# Peso [0-1] 0 = peso min, 1=peso max
# Color [0-1] 0 = menos rojo, 1= muy rojo
# Forma [0-1] 0 = circulo, 1 = no circulo

x = np.array([
    [0.5, 0.9, 0.1],
    [0.3, 0.8, 0.2],
    [0.2, 0.95, 0.01],
    [0.7, 0.1, 0.8],
    [0.9, 0.2, 0.7],
    [0.8, 0.3, 0.85]
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
b = np.ones((1, 1))
eta = 0.01


def step(inputs):
    if inputs >= 0:
        return 1
    return 0

f = False
s = 0
max_itr = 1000

print('Antes de entrenar:')
v = np.dot(x, w.T) + b # 6x3 1x3 -> 6x1
u = np.vectorize(step)
y = u(v)
print(y)

while f is False and s < max_itr:
    f = True
    for k in range(len(x)):
        v = np.dot(x[k, :], w.T)
        y = step(v)
        e = t[k] - y
        if t[k] != y:
            w = w + (x[k, :] * e * eta)
            b = b + (e * eta)
            f = False
    s = s + 1

print(f'Epocas: {s}')
print(f'w: {w}')
print(f'b: {b}')

print('Despues de entrenar:')
v = np.dot(x, w.T) + b
u = np.vectorize(step)
y = u(v)
print(y)

# Peso [0-1] 0 = peso min, 1=peso max
# Color [0-1] 0 = menos rojo, 1= muy rojo
# Forma [0-1] 0 = circulo, 1 = no circulo

x_test = np.array([
    [0.3, 0.87, 0.2],
    [0.1, 0.8, 0.1],
    [0.35, 0.15, 0.77],
    [0.5, 0.5, 0.5]
])

print('Nuevas')
v = np.dot(x_test, w.T) + b
u = np.vectorize(step)
y = u(v)
# Esperado [[0]
#           [0]
#           [1]
#           [?]]
print(y)
