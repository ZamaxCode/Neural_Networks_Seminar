import torch
import os

LETTERS = True
XOR = False

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_neurons) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size,hidden_neurons),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_neurons,1),
            torch.nn.Sigmoid(),
        )
    
    def forward(self, tensor):
        return self.model(tensor)
    
def train(x, t, model, learning_rate, epochs):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        y = model(x)
        e = loss_fn(y,t)
        optimizer.zero_grad()
        e.backward()
        optimizer.step()

def test(x, model, test_mode):
    y = model(x)
    if test_mode == LETTERS:
        if y>=0 and y<0.1:
            return 'A'
        elif y>=0.1 and y<0.35:
            return 'E'
        elif y>=0.35 and y<=0.6:
            return 'I'
        elif y>=0.6 and y<0.85:
            return 'O'
        else:
            return 'U'
    else:
        return y

def letters():
    A = torch.tensor([
        [0,0,0,1,0,0,0],
        [0,0,1,0,1,0,0],
        [0,1,0,0,0,1,0],
        [0,1,1,1,1,1,0],
        [0,1,0,0,0,1,0],
        [0,1,0,0,0,1,0]],dtype=torch.float)
    A=torch.reshape(A, (1,42))
    
    E = torch.tensor([
        [0,1,1,1,1,1,0],
        [0,1,0,0,0,0,0],
        [0,1,0,0,0,0,0],
        [0,1,1,1,1,0,0],
        [0,1,0,0,0,0,0],
        [0,1,1,1,1,1,0]],dtype=torch.float)
    E=torch.reshape(E, (1,42))
    
    I = torch.tensor([
        [0,1,1,1,1,1,0],
        [0,0,0,1,0,0,0],
        [0,0,0,1,0,0,0],
        [0,0,0,1,0,0,0],
        [0,0,0,1,0,0,0],
        [0,1,1,1,1,1,0]],dtype=torch.float)
    I=torch.reshape(I, (1,42))

    O = torch.tensor([
        [0,0,1,1,1,0,0],
        [0,1,0,0,0,1,0],
        [0,1,0,0,0,1,0],
        [0,1,0,0,0,1,0],
        [0,1,0,0,0,1,0],
        [0,0,1,1,1,0,0]],dtype=torch.float)
    O=torch.reshape(O, (1,42))

    U = torch.tensor([
        [0,1,0,0,0,1,0],
        [0,1,0,0,0,1,0],
        [0,1,0,0,0,1,0],
        [0,1,0,0,0,1,0],
        [0,1,0,0,0,1,0],
        [0,0,1,1,1,0,0]],dtype=torch.float)
    U=torch.reshape(U, (1,42))

    noise = torch.tensor([
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1]],dtype=torch.float)
    noise=torch.reshape(noise, (1,42))

    mlp = MLP(42,5)

    x = torch.cat([A,E,I,O,U])
    t = torch.tensor([[0], [0.2], [0.5], [0.7], [1]],dtype=torch.float)

    print("Starting train...")
    train(x, t, mlp, 0.1, 10000)

    print("Prediccion de letra A:", test(A,mlp,LETTERS))
    print("Prediccion de letra E:", test(E,mlp,LETTERS))
    print("Prediccion de letra I:", test(I,mlp,LETTERS))
    print("Prediccion de letra O:", test(O,mlp,LETTERS))
    print("Prediccion de letra U:", test(U,mlp,LETTERS))
    print("Prediccion de letra con ruido:", test(noise,mlp,LETTERS))

def xor():
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

    mlp = MLP(2,3)
    print("Starting train...")
    train(x, t, mlp, 0.3, 10000)
    print("Final prediction of Xor")
    print(test(x, mlp, XOR))

def main():
    input_user = '0'
    wrong_select = False
    while input_user!='1' and input_user!='2' and input_user!='3':
        os.system('clear')
        print("[==========================]")
        print("||   Ejemplo a ejecutar   ||")
        print("[==========================]")
        print("|| 1.-Xor                 ||")
        print("|| 2.-Letras              ||")
        print("|| 3.-Salir               ||")
        print("[==========================]")
        if wrong_select:
            print(f"Opcion '{input_user}' invalida - Vuelva a intentar.")
        input_user = (input("Seleccion: "))

        if input_user == '1':
            os.system('clear')
            print("[===========================]")
            print("||      Compuerta Xor      ||")
            print("[===========================]")
            xor()

        elif input_user == '2':
            os.system('clear')
            print("[==========================]")
            print("||   Dectecto de letras   ||")
            print("[==========================]")
            letters()
            
        elif input_user == '3':
            print("Hasta pronto!")
        else:
            wrong_select = True
main()
