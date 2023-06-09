import torch
import matplotlib.pyplot as plt

x = torch.tensor([
    [0.5, 0.9],
    [0.3, 0.8],
    [0.2, 0.95],
    [0.7, 0.1],
    [0.9, 0.2],
    [0.8, 0.3]
    ],dtype=torch.float)

t = torch.tensor([
    [0],
    [0],
    [0],
    [1],
    [1],
    [1]
    ],dtype=torch.float)

class Perceptron(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2,1)
    
    def forward(self, tensor):
        y = self.linear(tensor)
        return y

perceptron = Perceptron()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(params=perceptron.parameters(), lr=0.01)

for epoch in range(1000):
    y = perceptron(x)
    e = loss_fn(y,t)
    optimizer.zero_grad()
    e.backward()
    optimizer.step()

x1 = x[...,0]
x2 = x[...,1]

plt.scatter(x=x1, y=x2)

plt.xlim(-2,2)
plt.ylim(-2,2)

weights=torch.tensor(perceptron.linear.weight)
bias=torch.tensor(perceptron.linear.bias)

a=weights[0,0]
b=weights[0,1]
c=bias[0]
y1 = (-c-(a*-1.5))/b
y2 = (-c-(a*1.5))/b
plt.plot([-1.5,1.5], [y1,y2])

plt.show()