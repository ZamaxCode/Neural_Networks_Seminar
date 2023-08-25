import numpy as np

def sigmoid(v):
    return 1/(1+np.exp(-v))

def dsigmoid(v):
    return v*(1-v)

def MLP(x, t, eta=0.05, max_it=100000, min_err=0.01, c=3):
    s = 0
    error = 1
    np.random.seed(9001)
    wo = np.random.rand(c, np.size(x[1,:]))
    ws = np.random.rand(1, c)
    n = len(x[:,0])
    while s < max_it and error > min_err:
        E = 0
        for k in range(0, n):
            yo = sigmoid(np.dot(x[k,:], wo.T))
            ys = sigmoid(np.dot(yo, ws.T))
            e = t[k] - ys
            es = dsigmoid(ys) * e
            eo = dsigmoid(yo) * es * ws
            wo = wo + (eo.T * x[k,:] * eta)
            ws = ws + (es * yo * eta)
            E = E + e**2
        error = E / n
        s = s + 1
    return wo, ws


def main():
    x = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    t = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    A = np.array([
        [0,0,0,1,0,0,0],
        [0,0,1,0,1,0,0],
        [0,1,0,0,0,1,0],
        [0,1,1,1,1,1,0],
        [0,1,0,0,0,1,0],
        [0,1,0,0,0,1,0]]).flatten()
    
    E = np.array([
        [0,1,1,1,1,1,0],
        [0,1,0,0,0,0,0],
        [0,1,0,0,0,0,0],
        [0,1,1,1,1,0,0],
        [0,1,0,0,0,0,0],
        [0,1,1,1,1,1,0]]).flatten()
    
    I = np.array([
        [0,1,1,1,1,1,0],
        [0,0,0,1,0,0,0],
        [0,0,0,1,0,0,0],
        [0,0,0,1,0,0,0],
        [0,0,0,1,0,0,0],
        [0,1,1,1,1,1,0]]).flatten()
    
    O = np.array([
        [0,0,1,1,1,0,0],
        [0,1,0,0,0,1,0],
        [0,1,0,0,0,1,0],
        [0,1,0,0,0,1,0],
        [0,1,0,0,0,1,0],
        [0,0,1,1,1,0,0]]).flatten()
    
    U = np.array([
        [0,1,0,0,0,1,0],
        [0,1,0,0,0,1,0],
        [0,1,0,0,0,1,0],
        [0,1,0,0,0,1,0],
        [0,1,0,0,0,1,0],
        [0,0,1,1,1,0,0]]).flatten()

    x2 = np.array([A,E,I,O,U])
    t2 = np.array([0,
                   0.2,
                   0.5,
                   0.7,
                   1])

    wo, ws = MLP(x2,t2)
    yo = sigmoid(np.dot(E, wo.T))
    ys = sigmoid(np.dot(yo, ws.T))
    print(ys)

main()
