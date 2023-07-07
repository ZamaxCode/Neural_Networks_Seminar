import torch

x = torch.tensor([[1],[2],[3],[4],[5]],dtype=torch.float)

t = torch.tensor([[1.],[2.],[3.],[4.],[5.]])

class LinearRegression(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features=1, out_features=1)
    
    def forward(self, tensor):
        y = self.linear(tensor)
        return y

linReg = LinearRegression()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(params=linReg.parameters(), lr=0.01)

test = linReg(torch.tensor([[6.]]))
print(test)

for epoch in range(100000):
    y = linReg(x)
    e = loss_fn(y,t)
    optimizer.zero_grad()
    e.backward()
    optimizer.step()

test = linReg(torch.tensor([[6.]]))
print(test)