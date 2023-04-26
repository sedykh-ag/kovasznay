import torch
import pennylane as qml
import numpy as np
from torch.utils.data import TensorDataset
from torch import nn

def get_circuit():
    n_qubits = 4
    n_depth = 4
    n_layers = 1
    in_dim = n_qubits * n_depth
    out_dim = n_qubits
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, w):
        qml.StronglyEntanglingLayers(w[0], wires=range(n_qubits))
        
        for i in range(n_depth):
            qml.AngleEmbedding(features=inputs[i*n_qubits : (i+1)*n_qubits],
                            wires=range(n_qubits),
                            rotation="X")
            qml.StronglyEntanglingLayers(w[i+1], wires=range(n_qubits))
            
        return [qml.expval(qml.PauliZ(k)) for k in range(n_qubits)]

    weight_shapes = {"w": (n_depth+1, n_layers, n_qubits, 3)}

    return circuit, weight_shapes, in_dim, out_dim

class EncNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.circuit, self.weight_shapes, self.in_dim, self.out_dim = get_circuit()
        self.qlayer = qml.qnn.TorchLayer(self.circuit, self.weight_shapes)
        self.lin_out = nn.Linear(self.out_dim, 1)
    def forward(self, X):
        X = X.view(-1, 1)
        X = X.expand(-1, self.in_dim)

        X = self.qlayer(X)
        X = self.lin_out(X)

        return X

def get_dataset():
    f = lambda x: 0.15 * torch.sin(x)
    x = torch.linspace(-torch.pi, torch.pi, 100)
    return TensorDataset(x, f(x).view(-1, 1))


