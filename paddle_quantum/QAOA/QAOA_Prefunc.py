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
Aid func
"""
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from paddle_quantum.utils import pauli_str_to_matrix


def Draw_original_graph(G):
    """
    This is to draw the original graph
     Args:
        G: the constructed graph
     Returns:
        Null
    """
    pos = nx.circular_layout(G)
    options = {
        "with_labels": True,
        "font_size": 20,
        "font_weight": "bold",
        "font_color": "white",
        "node_size": 2000,
        "width": 2
    }
    nx.draw_networkx(G, pos, **options)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()
    
    return


def Draw_cut_graph(V, E, G, cut_bitstring):
    """
    This is to draw the graph after cutting
    Args:
       V: vertices in the graph
       E: edges in the graph
       cut_bitstring: bit string indicate whether vertices belongs to the first group or the second group
    Returns:
       Null
    """
    node_cut = ["blue" if cut_bitstring[v] == "1" else "red" for v in V]

    edge_cut = [
        "solid" if cut_bitstring[u] == cut_bitstring[v] else "dashed"
        for (u, v) in G.edges()
        ]
    
    pos = nx.circular_layout(G)
    
    options = {
        "with_labels": True,
        "font_size": 20,
        "font_weight": "bold",
        "font_color": "white",
        "node_size": 2000,
        "width": 2
    }

    nx.draw(
            G,
            pos,
            node_color = node_cut,
            style = edge_cut,
            **options
    )
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()
    
    return


def Generate_default_graph(n):
    """
    This is to generate a default graph if no input
     Args:
        n: number of vertices
     Returns:
        G: the graph
        E: edges list
        V: vertices list
    """
    G = nx.Graph()
    V = range(n)
    G.add_nodes_from(V)
    E = []
    for i in range(n - 1):
        E.append((i, i + 1))
    E.append((0, n - 1))
        
    G.add_edges_from(E)
    
    return G, V, E


def Generate_H_D(E, n):
    """
    This is to construct Hamiltonia H_D
     Args:
        E: edges of the graph
     Returns:
        Hamiltonia list
        Hamiltonia H_D
    """
    H_D_list = []
    for (u, v) in E:
        H_D_list.append([-1.0, 'z' + str(u) + ',z' + str(v)])
    print(H_D_list)
    H_D_matrix = pauli_str_to_matrix(H_D_list, n)

    return H_D_list, H_D_matrix


def Draw_benchmark(summary_iter, summary_loss, H_min):
    """
    This is draw the learning tendency, and difference bwtween it and the benchmark
    Args:
        summary_iter: indicate which iteration
        summary_loss: indicate the energy of that iteration
        H_min: benchmark value H_min
    Returns:
        NULL
    """
    plt.figure(1)
    loss_QAOA, = plt.plot(
        summary_iter, 
        summary_loss,
        alpha=0.7, 
        marker='', 
        linestyle="--", 
        linewidth=2, 
        color='m')
    benchmark, = plt.plot(
        summary_iter,
        H_min,
        alpha=0.7,
        marker='',
        linestyle=":",
        linewidth=2,
        color='b')
        
    plt.xlabel('Number of iteration')
    plt.ylabel('Performance of the loss function for QAOA')

    plt.legend(
        handles=[loss_QAOA, benchmark],
        labels=[
            r'Loss function $\left\langle {\psi \left( {\bf{\theta }} \right)} '
            r'\right|H\left| {\psi \left( {\bf{\theta }} \right)} \right\rangle $',
            'The benchmark result',
        ],
        loc='best')

    # Show the picture
    plt.show()
    
    return 


def main():
    # number of qubits or number of nodes in the graph
    n = 4
    G, V, E = Generate_default_graph(n)
    Draw_original_graph(G)


if __name__ == "__main__":
    main()
