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

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import paddle
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.utils import pauli_str_to_matrix
from paddle_quantum.QAOA.QAOA_Prefunc import generate_graph, H_generator

# Random seed for optimizer
SEED = 1024

__all__ = [
    "circuit_QAOA",
    "circuit_extend_QAOA",
    "Net",
    "Paddle_QAOA",
]


def circuit_QAOA(theta, adjacency_matrix, N, P):
    """
    This function constructs the parameterized QAOA circuit which is composed of P layers of two blocks:
    one block is based on the problem Hamiltonian H which encodes the classical problem,
    and the other is constructed from the driving Hamiltonian describing the rotation around Pauli X
    acting on each qubit. It outputs the final state of the QAOA circuit.

    Args:
        theta: parameters to be optimized in the QAOA circuit
        adjacency_matrix:  the adjacency matrix of the graph encoding the classical problem
        N: number of qubits, or equivalently, the number of parameters in the original classical problem
        P: number of layers of two blocks in the QAOA circuit
    Returns:
        the QAOA circuit
    """

    cir = UAnsatz(N)

    # prepare the input state in the uniform superposition of 2^N bit-strings in the computational basis
    cir.superposition_layer()
    # This loop defines the QAOA circuit with P layers of two blocks
    for layer in range(P):
        # The second and third loops construct the first block which involves two-qubit operation
        #  e^{-i\gamma Z_iZ_j} acting on a pair of qubits or nodes i and j in the circuit in each layer.
        for row in range(N):
            for col in range(N):
                if adjacency_matrix[row, col] and row < col:
                    cir.cnot([row, col])
                    cir.rz(theta[layer][0], col)
                    cir.cnot([row, col])
        # This loop constructs the second block only involving the single-qubit operation e^{-i\beta X}.
        for i in range(N):
            cir.rx(theta[layer][1], i)

    return cir


def circuit_extend_QAOA(theta, adjacency_matrix, N, P):
    """
    This is an extended version of the QAOA circuit, and the main difference is the block constructed
    from the driving Hamiltonian describing the rotation around an arbitrary direction on each qubit.

    Args:
        theta: parameters to be optimized in the QAOA circuit
        input_state: input state of the QAOA circuit which usually is the uniform superposition of 2^N bit-strings
                     in the computational basis
        adjacency_matrix:  the adjacency matrix of the problem graph encoding the original problem
        N: number of qubits, or equivalently, the number of parameters in the original classical problem
        P: number of layers of two blocks in the QAOA circuit
    Returns:
        the extended QAOA circuit

    Note:
        If this circuit_extend_QAOA function is used to construct QAOA circuit, then we need to change the parameter layer
        in the Net function defined below from the Net(shape=[D, 2]) for circuit_QAOA function to Net(shape=[D, 4])
        because the number of parameters doubles in each layer in this QAOA circuit.
    """
    cir = UAnsatz(N)

    # prepare the input state in the uniform superposition of 2^N bit-strings in the computational basis
    cir.superposition_layer()
    for layer in range(P):
        for row in range(N):
            for col in range(N):
                if adjacency_matrix[row, col] and row < col:
                    cir.cnot([row, col])
                    cir.rz(theta[layer][0], col)
                    cir.cnot([row, col])

        for i in range(N):
            cir.u3(*theta[layer][1:], i)

    return cir


class Net(paddle.nn.Layer):
    """
    It constructs the net for QAOA which combines the  QAOA circuit with the classical optimizer which sets rules
    to update parameters described by theta introduced in the QAOA circuit.

    """

    def __init__(
            self,
            shape,
            param_attr=paddle.nn.initializer.Uniform(low=0.0, high=np.pi),
            dtype="float64",
    ):
        super(Net, self).__init__()

        self.theta = self.create_parameter(
            shape=shape, attr=param_attr, dtype=dtype, is_bias=False
        )

    def forward(self, adjacency_matrix, N, P, METHOD):
        """
        This function constructs the loss function for the QAOA circuit.

        Args:
            adjacency_matrix: the adjacency matrix generated from the graph encoding the classical problem
            N: number of qubits
            P: number of layers
            METHOD: which version of QAOA is chosen to solve the problem, i.e., standard version labeled by 1 or
            extended version by 2.
        Returns:
            the loss function for the parameterized QAOA circuit and the circuit itself
        """

        # Generate the problem_based quantum Hamiltonian H_problem based on the classical problem in paddle
        H_problem = H_generator(N, adjacency_matrix)

        # The standard QAOA circuit: the function circuit_QAOA is used to construct the circuit, indexed by METHOD 1.
        if METHOD == 1:
            cir = circuit_QAOA(self.theta, adjacency_matrix, N, P)
        # The extended QAOA circuit: the function circuit_extend_QAOA is used to construct the net, indexed by METHOD 2.
        elif METHOD == 2:
            cir = circuit_extend_QAOA(self.theta, adjacency_matrix, N, P)
        else:
            raise ValueError("Wrong method called!")

        cir.run_state_vector()
        loss = cir.expecval(H_problem)

        return loss, cir


def Paddle_QAOA(classical_graph_adjacency, N, P, METHOD, ITR, LR):
    """
    This is the core function to run QAOA.

     Args:
         classical_graph_adjacency: adjacency matrix to describe the graph which encodes the classical problem
         N: number of qubits (default value N=4)
         P: number of layers of blocks in the QAOA circuit (default value P=4)
         METHOD: which version of the QAOA circuit is used: 1, standard circuit (default); 2, extended circuit
         ITR: number of iteration steps for QAOA (default value ITR=120)
         LR: learning rate for the gradient-based optimization method (default value LR=0.1)
     Returns:
         the optimized QAOA circuit
    """
    # Construct the net or QAOA circuits based on the standard modules
    if METHOD == 1:
        net = Net(shape=[P, 2])
    # Construct the net or QAOA circuits based on the extended modules
    elif METHOD == 2:
        net = Net(shape=[P, 4])
    else:
        raise ValueError("Wrong method called!")

    # Classical optimizer
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())

    # Gradient descent loop
    summary_iter, summary_loss = [], []
    for itr in range(1, ITR + 1):
        loss, cir = net(
            classical_graph_adjacency, N, P, METHOD
        )
        loss.backward()
        opt.minimize(loss)
        opt.clear_grad()

        if itr % 10 == 0:
            print("iter:", itr, "  loss:", "%.4f" % loss.numpy())
        summary_loss.append(loss[0][0].numpy())
        summary_iter.append(itr)

        theta_opt = net.parameters()[0].numpy()
        # print("Optimized parameters theta:\n", theta_opt)

        os.makedirs("output", exist_ok=True)
        np.savez("./output/summary_data", iter=summary_iter, energy=summary_loss)

    return cir


def main(N=4):
    paddle.seed(SEED)
    # number of qubits or number of nodes in the graph
    N = 4
    classical_graph, classical_graph_adjacency = generate_graph(N, GRAPHMETHOD=1)
    print(classical_graph_adjacency)

    # Convert the Hamiltonian's list form to matrix form
    H_matrix = pauli_str_to_matrix(H_generator(N, classical_graph_adjacency), N)

    H_diag = np.diag(H_matrix).real
    H_max = np.max(H_diag)
    H_min = np.min(H_diag)

    print(H_diag)
    print('H_max:', H_max, '  H_min:', H_min)

    pos = nx.circular_layout(classical_graph)
    nx.draw(classical_graph, pos, width=4, with_labels=True, font_weight='bold')
    plt.show()

    classical_graph, classical_graph_adjacency = generate_graph(N, 1)

    opt_cir = Paddle_QAOA(classical_graph_adjacency, N=4, P=4, METHOD=1, ITR=120, LR=0.1)

    # Load the data of QAOA
    x1 = np.load('./output/summary_data.npz')

    H_min = np.ones([len(x1['iter'])]) * H_min

    # Plot loss
    loss_QAOA, = plt.plot(x1['iter'], x1['energy'], alpha=0.7, marker='', linestyle="--", linewidth=2, color='m')
    benchmark, = plt.plot(x1['iter'], H_min, alpha=0.7, marker='', linestyle=":", linewidth=2, color='b')
    plt.xlabel('Number of iteration')
    plt.ylabel('Performance of the loss function for QAOA')

    plt.legend(handles=[
        loss_QAOA,
        benchmark
    ],
        labels=[
            r'Loss function $\left\langle {\psi \left( {\bf{\theta }} \right)} '
            r'\right|H\left| {\psi \left( {\bf{\theta }} \right)} \right\rangle $',
            'The benchmark result',
        ], loc='best')

    # Show the plot
    plt.show()

    # Measure the output state of the QAOA circuit for 1024 shots by default
    prob_measure = opt_cir.measure(plot=True)

    # Find the max value in measured probability of bitstrings
    max_prob = max(prob_measure.values())
    # Find the bitstring with max probability
    solution_list = [result[0] for result in prob_measure.items() if result[1] == max_prob]
    print("The output bitstring:", solution_list)

    # Draw the graph representing the first bitstring in the solution_list to the MaxCut-like problem
    head_bitstring = solution_list[0]

    node_cut = ["blue" if head_bitstring[node] == "1" else "red" for node in classical_graph]

    edge_cut = [
        "solid" if head_bitstring[node_row] == head_bitstring[node_col] else "dashed"
        for node_row, node_col in classical_graph.edges()
    ]
    nx.draw(
        classical_graph,
        pos,
        node_color=node_cut,
        style=edge_cut,
        width=4,
        with_labels=True,
        font_weight="bold",
    )
    plt.show()


if __name__ == "__main__":
    main()
