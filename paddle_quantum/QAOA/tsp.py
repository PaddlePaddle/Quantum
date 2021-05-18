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
Travelling Salesman Problem (TSP): To learn more about the functions and properties of this application,
you could check the corresponding Jupyter notebook under the Tutorial folder.
"""


from itertools import permutations
import numpy as np
import networkx as nx
import paddle
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.utils import pauli_str_to_matrix



__all__ = [
    "tsp_hamiltonian",
    "solve_tsp",
    "solve_tsp_brute_force"
]


def tsp_hamiltonian(g, A, n):
    """
    This is to construct Hamiltonia H_C
     Args:
        G: the graph to solve
     Returns:
        Hamiltonian list
        Hamiltonian H_C
    """
    H_C_list1 = []
    for i in range(n - 1):
        for j in range(n - 1):
            if i != j:
                w_ij = g[i][j]['weight']
                for t in range(n - 2):
                    H_C_list1.append([w_ij / 4, 'i1'])
                    H_C_list1.append([w_ij / 4, 'z' + str(i * (n - 1) + t) + ',z' + str(j * (n - 1) + t + 1)])
                    H_C_list1.append([-w_ij / 4, 'z' + str(i * (n - 1) + t)])
                    H_C_list1.append([-w_ij / 4, 'z' + str(j * (n - 1) + t + 1)])
        H_C_list1.append([g[n - 1][i]['weight'] / 2, 'i1'])
        H_C_list1.append([-g[n - 1][i]['weight'] / 2, 'z' + str(i * (n - 1) + (n - 2))])
        H_C_list1.append([g[i][n - 1]['weight'] / 2, 'i1'])
        H_C_list1.append([-g[i][n - 1]['weight'] / 2, 'z' + str(i * (n - 1))])

    H_C_list2 = []
    for i in range(n - 1):
        H_C_list2.append([1, 'i1'])
        for t in range(n-1):
            H_C_list2.append([-2 * 1/2, 'i1'])
            H_C_list2.append([2 * 1/2, 'z' + str(i * (n - 1) + t)])
            H_C_list2.append([2/4, 'i1'])
            H_C_list2.append([-2/4, 'z' + str(i * (n - 1) + t)])
            for tt in range(t):
                H_C_list2.append([2/4, 'i1'])
                H_C_list2.append([2/4, 'z' + str(i * (n - 1) + t) + ',z' + str(i * (n - 1) + tt)])
                H_C_list2.append([-2/4, 'z' + str(i * (n - 1) + t)])
                H_C_list2.append([-2/4, 'z' + str(i * (n - 1) + tt)])
    H_C_list2 = [[A * c, s] for (c, s) in H_C_list2]

    H_C_list3 = []
    for t in range(n - 1):
        H_C_list3.append([1, 'i1'])
        for i in range(n-1):
            H_C_list3.append([-2 * 1/2, 'i1'])
            H_C_list3.append([2 * 1/2, 'z' + str(i * (n - 1) + t)])
            H_C_list3.append([2/4, 'i1'])
            H_C_list3.append([-2/4, 'z' + str(i * (n - 1) + t)])
            for ii in range(i):
                H_C_list3.append([2/4, 'i1'])
                H_C_list3.append([2/4, 'z' + str(i * (n - 1) + t) + ',z' + str(ii * (n - 1) + t)])
                H_C_list3.append([-2/4,'z'+str(i * (n - 1) + t)])
                H_C_list3.append([-2/4,'z'+str(ii * (n - 1) + t)])
    H_C_list3 = [[A * c, s] for (c, s) in H_C_list3]

    H_C_list = H_C_list1 + H_C_list2 + H_C_list3

    return H_C_list


class _TSPNet(paddle.nn.Layer):
    """
    It constructs the net for TSP which combines the complex entangled circuit with the classical optimizer that sets rules
    to update parameters described by theta introduced in the circuit.
    """
    def __init__(self, n, p, dtype="float64"):
        super(_TSPNet, self).__init__()

        self.p = p
        self.num_qubits = (n - 1) ** 2
        self.theta = self.create_parameter(shape=[self.p, self.num_qubits, 3],
            default_initializer=paddle.nn.initializer.Uniform(low=0.0, high=2 * np.pi),
            dtype=dtype, is_bias=False)

    def forward(self, H_C_ls):
        """
        Forward propagation
        """
        cir = UAnsatz(self.num_qubits)
        cir.complex_entangled_layer(self.theta, self.p)
        cir.run_state_vector()
        loss = cir.expecval(H_C_ls)

        return loss, cir


def solve_tsp(g, A, p=2, ITR=120, LR=0.4, print_loss=False, shots=0):
    """
    This is the core function to solve the TSP.

    Args:
        g: the graph to solve
        A: the penality parameter
        p: number of layers of blocks in the complex entangled circuit (default value p=2)
        ITR: number of iteration steps for the complex entangled circuit (default value ITR=120)
        LR: learning rate for the gradient-based optimization method (default value LR=0.4)
    Returns:
        string representation for the optimized walk for the salesman
    """
    e = list(g.edges(data = True))
    v = list(g.nodes)
    n = len(v)

    H_C_list = tsp_hamiltonian(g, A, n)

    net = _TSPNet(n, p)
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())

    for itr in range(1, ITR + 1):
        loss, cir = net(H_C_list)
        loss.backward()
        opt.minimize(loss)
        opt.clear_grad()
        if print_loss and itr % 10 == 0:
            print("iter:", itr, "loss:", "%.4f"% loss.numpy())

    prob_measure = cir.measure(shots=shots)
    reduced_salesman_walk = max(prob_measure, key=prob_measure.get)
    str_by_vertex = [reduced_salesman_walk[i:i + n - 1] for i in range(0, len(reduced_salesman_walk) + 1, n - 1)]
    salesman_walk = '0'.join(str_by_vertex) + '0' * (n - 1) + '1'

    return salesman_walk


def solve_tsp_brute_force(g):
    """
    This is the brute-force algorithm to solve the TSP.

    Args:
        g: the graph to solve
    Returns:
        the list of the optimized walk in the visiting order and the optimal distance
    """
    n = len(g.nodes)
    all_routes = list(permutations(range(n)))
    best_distance = 1e10
    best_route = (0, 0, 0, 0)

    for route in all_routes:
        distance = 0
        for i in range(n):
            u = route[i]
            v = route[(i + 1) % n]
            distance += g[u][v]['weight']
        if distance < best_distance:
            best_distance = distance
            best_route = route

    return list(best_route), best_distance
