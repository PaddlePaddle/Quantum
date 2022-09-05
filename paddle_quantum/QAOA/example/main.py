# !/usr/bin/env python3
# Copyright (c) 2020 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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

r"""
main
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import paddle
from paddle_quantum.QAOA.maxcut import maxcut_hamiltonian, find_cut
from paddle_quantum.qinfo import pauli_str_to_matrix

SEED = 1024
options = {
    "with_labels": True,
    "font_size": 20,
    "font_weight": "bold",
    "font_color": "white",
    "node_size": 2000,
    "width": 2
}


if __name__ == "__main__":
    n = 4
    paddle.seed(SEED)
    
    p = 4  # number of layers in the circuit
    ITR = 120  # number of iterations
    LR = 0.1    # learning rate

    G = nx.cycle_graph(4)
    V = list(G.nodes())
    E = list(G.edges())
    # Draw the original graph
    pos = nx.circular_layout(G)
    nx.draw_networkx(G, pos, **options)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()

    # construct the Hamiltonian
    H_D_list = maxcut_hamiltonian(E)
    H_D_matrix = pauli_str_to_matrix(H_D_list, n)
    H_D_diag = np.diag(H_D_matrix).real
    H_max = np.max(H_D_diag)

    print(H_D_diag)
    print('H_max:', H_max)   

    cut_bitstring, _ = find_cut(G, p, ITR, LR, print_loss=True, plot=True)
    print("The bit string form of the cut found:", cut_bitstring)

    node_cut = ["blue" if cut_bitstring[v] == "1" else "red" for v in V]
    edge_cut = ["solid" if cut_bitstring[u] == cut_bitstring[v] else "dashed" for (u, v) in G.edges()]

    nx.draw(G, pos, node_color=node_cut, style=edge_cut, **options)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()
