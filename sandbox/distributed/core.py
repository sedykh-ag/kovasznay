import pennylane as qml
import numpy as np

def get_dataloader():


def get_quantum_circuit():
    n_qubits = 4
    n_depth = 1
    n_layers = 1
    in_dim = n_qubits * n_depth
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, w):
        qml.BasicEntanglerLayers(w[0], wires=range(n_qubits), rotation=qml.RX)
        
        for i in range(n_depth):
            qml.AngleEmbedding(features=inputs[i*n_qubits : (i+1)*n_qubits],
                            wires=range(n_qubits),
                            rotation="X")
            qml.BasicEntanglerLayers(w[i+1],
                                    wires=range(n_qubits),
                                    rotation=qml.RX)
            
        return [qml.expval(qml.PauliZ(k)) for k in range(1)]

    weight_shapes = {"w": (n_depth+1, n_layers, n_qubits)}

def main():
    circuit, weight_shapes = get_quantum_circuit()
    

