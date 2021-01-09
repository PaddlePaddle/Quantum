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
main
"""

from paddle import fluid

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from paddle_quantum.utils import pauli_str_to_matrix
from paddle_quantum.QAOA.Paddle_QAOA import Paddle_QAOA
from paddle_quantum.QAOA.QAOA_Prefunc import generate_graph, H_generator


def main(N=4):
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

    with fluid.dygraph.guard():
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
