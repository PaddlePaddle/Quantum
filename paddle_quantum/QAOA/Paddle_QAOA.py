# Copyright (c) 2021 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Paddle_QAOA: To learn more about the functions and properties of this application,
you could check the corresponding Jupyter notebook under the Tutorial folder.
"""

import paddle
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.utils import pauli_str_to_matrix
from paddle_quantum.QAOA.QAOA_Prefunc import Generate_H_D, Generate_default_graph
from paddle_quantum.QAOA.QAOA_Prefunc import Draw_benchmark
from paddle_quantum.QAOA.QAOA_Prefunc import Draw_cut_graph, Draw_original_graph

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# Random seed for optimizer
SEED = 1024

__all__ = [
    "circuit_QAOA",
    "Net",
    "Paddle_QAOA",
]

def circuit_QAOA(E, V, n, p, gamma, beta):
    """
    This function constructs the parameterized QAOA circuit which is composed of P layers of two blocks:
    one block is based on the problem Hamiltonian H which encodes the classical problem,
    and the other is constructed from the driving Hamiltonian describing the rotation around Pauli X
    acting on each qubit. It outputs the final state of the QAOA circuit.
    Args:
        E: edges of the graph
        V: vertices of the graph
        n: number of qubits in th QAOA circuit
        p: number of layers of two blocks in the QAOA circuit
        gamma: parameter to be optimized in the QAOA circuit, parameter for the first block
        beta: parameter to be optimized in the QAOA circui, parameter for the second block
    Returns:
        the QAOA circuit
    """
    cir = UAnsatz(n)
    cir.superposition_layer()
    for layer in range(p):
        for (u, v) in E:
            cir.cnot([u, v])
            cir.rz(gamma[layer], v)
            cir.cnot([u, v])
        for v in V:
            cir.rx(beta[layer], v)
    return cir

class Net(paddle.nn.Layer):
    """
    It constructs the net for QAOA which combines the  QAOA circuit with the classical optimizer which sets rules
    to update parameters described by theta introduced in the QAOA circuit.
    """
    def __init__(
        self,
        p,
        dtype="float64",
    ):
        super(Net, self).__init__()

        self.p = p
        self.gamma = self.create_parameter(shape = [self.p], 
                                           default_initializer = paddle.nn.initializer.Uniform(
                                           low = 0.0, 
                                           high = 2 * np.pi
                                           ),
                                           dtype = dtype, 
                                           is_bias = False)
        self.beta = self.create_parameter(shape = [self.p], 
                                          default_initializer = paddle.nn.initializer.Uniform(
                                          low = 0.0, 
                                          high = 2 * np.pi
                                          ),
                                          dtype = dtype, is_bias = False)


    def forward(self, n, E, V, H_D_list):
        cir = circuit_QAOA(E, V, n, self.p, self.gamma, self.beta)
        cir.run_state_vector()
        loss = -cir.expecval(H_D_list)
        return loss, cir

def Paddle_QAOA(n, p, E, V, H_D_list, ITR, LR):
    """
    This is the core function to run QAOA.
     Args:
         n: number of qubits (default value N=4)
         E: edges of the graph
         V: vertices of the graph
         p: number of layers of blocks in the QAOA circuit (default value p=4)
         ITR: number of iteration steps for QAOA (default value ITR=120)
         LR: learning rate for the gradient-based optimization method (default value LR=0.1)
     Returns:
         the optimized QAOA circuit
         summary_iter
         summary_loss
    """
    net = Net(p)
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())

    summary_iter, summary_loss = [], []
    for itr in range(1, ITR + 1):
        loss, cir = net(n, E, V, H_D_list)
        loss.backward()
        opt.minimize(loss)
        opt.clear_grad()
        if itr % 10 == 0:
            print("iter:", itr, "  loss:", "%.4f" % loss.numpy())
        summary_loss.append(loss[0][0].numpy())
        summary_iter.append(itr)

    gamma_opt = net.gamma.numpy()
    print("优化后的参数 gamma:\n", gamma_opt)
    beta_opt = net.beta.numpy()
    print("优化后的参数 beta:\n", beta_opt)

    return cir, summary_iter, summary_loss

def main(n = 4, E = None):
    paddle.seed(SEED)
    
    p = 4 # number of layers in the circuit  
    ITR = 120  #number of iterations
    LR = 0.1    #learning rate
    
    if E is None:
        G, V, E = Generate_default_graph(n)
    else:
        G = nx.Graph()
        V = range(n)
        G.add_nodes_from(V)
        G.add_edges_from(E)
    
    Draw_original_graph(G)
    
    #construct the Hamiltonia
    H_D_list, H_D_matrix = Generate_H_D(E, n)
    H_D_diag = np.diag(H_D_matrix).real
    H_max = np.max(H_D_diag)
    H_min = -H_max

    print(H_D_diag)
    print('H_max:', H_max, '  H_min:', H_min)   

    cir, summary_iter, summary_loss = Paddle_QAOA(n, p, E, V, H_D_list, ITR, LR)
    
    H_min = np.ones([len(summary_iter)]) * H_min
    Draw_benchmark(summary_iter, summary_loss, H_min)
    
    prob_measure = cir.measure(plot=True)
    cut_bitstring = max(prob_measure, key=prob_measure.get)
    print("找到的割的比特串形式：", cut_bitstring)

    Draw_cut_graph(V, E, G, cut_bitstring)

if __name__ == "__main__":
    n = int(input("Please input the number of vertices: "))
    user_input_edge_flag = int(input("Please choose if you want to input edges yourself (0 for yes, 1 for no): "))
    if user_input_edge_flag == 1:
        main(n)
    else:
        E = []
        prompt = "Please input tuples indicating edges (e.g., (0, 1)), input 'z' if finished: "
        while True:
            edge = input(prompt)
            if edge == 'z':
                main(n, E)
                break
            else:
                edge = eval(edge)
                E.append(edge)
