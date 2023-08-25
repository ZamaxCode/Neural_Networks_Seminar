import torch

x = torch.tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
    ],dtype=torch.float)

t = torch.tensor([
    [0],
    [1],
    [1],
    [0]
    ],dtype=torch.float)

class Adaline(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2,3),
            torch.nn.Sigmoid(),
            torch.nn.Linear(3,1),
            torch.nn.Sigmoid(),
        )
    
    def forward(self, tensor):
        return self.model(tensor)

adaline = Adaline()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(params=adaline.parameters(), lr=0.3)

for epoch in range(10000):
    y = adaline(x)
    e = loss_fn(y,t)
    optimizer.zero_grad()
    e.backward()
    optimizer.step()

y = adaline(x)
print(y)