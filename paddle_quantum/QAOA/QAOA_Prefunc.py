# Copyright (c) 2020 Paddle Quantum Authors. All Rights Reserved.
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
from matplotlib import pyplot
from numpy import abs, array, binary_repr, diag, kron, max, ones, real, where, zeros
import networkx

__all__ = [
    "plot_graph",
    "generate_graph",
    "H_generator",
]


def plot_graph(measure_prob_distribution, graph, N):
    """
    This function plots the graph encoding the combinatorial problem such as Max-Cut and the final graph encoding the
    approximate solution obtained from QAOA

    Args:
        measure_prob_distribution: the measurement probability distribution which is sampled from the output state
                                   of optimized QAOA circuit.
        graph: graph encoding the topology of the classical combinatorial problem, such as Max-Cut problem
        N: number of qubits, or number of nodes in the graph
    Return:
        three graphs: the first displays the graph topology of the classical problem;
                      the second is the bar graph for the measurement distribution for each bitstring in the output state
                      the third plots the graph corresponding to the bitstring with maximal measurement probability
    """

    # Find the position of max value in the measure_prob_distribution
    max_prob_pos_list = where(
        measure_prob_distribution == max(measure_prob_distribution))
    # Store the max value from ndarray to list
    max_prob_list = max_prob_pos_list[0].tolist()
    # Store it in the  binary format
    solution_list = [binary_repr(index, width=N) for index in max_prob_list]
    print("The output bitstring:", solution_list)

    # Draw the graph representing the first bitstring in the solution_list to the MaxCut-like problem
    head_bitstring = solution_list[0]

    node_cut = [
        "blue" if head_bitstring[node] == "1" else "red" for node in graph
    ]

    edge_cut = [
        "solid"
        if head_bitstring[node_row] == head_bitstring[node_col] else "dashed"
        for node_row, node_col in graph.edges()
    ]

    pos = networkx.circular_layout(graph)

    pyplot.figure(0)
    networkx.draw(graph, pos, width=4, with_labels=True, font_weight="bold")

    # when N is large, it is not suggested to plot this figure
    pyplot.figure(1)
    name_list = [binary_repr(index, width=N) for index in range(0, 2**N)]
    pyplot.bar(
        range(len(real(measure_prob_distribution))),
        real(measure_prob_distribution),
        width=0.7,
        tick_label=name_list, )
    pyplot.xticks(rotation=90)

    pyplot.figure(2)
    networkx.draw(
        graph,
        pos,
        node_color=node_cut,
        style=edge_cut,
        width=4,
        with_labels=True,
        font_weight="bold", )
    pyplot.show()


def generate_graph(N, GRAPHMETHOD):
    """
    This function offers two methods to generate a graph.

    Args:
        N: number of nodes (vertices) in the graph, which is also the number of qubits
        GRAPHMETHOD: which method to generate a graph
    Return:
        graph description and its adjacency matrix
    """
    # Method 1 generates a graph by self-definition. Note that the node label starts from 0 to N-1, while the edges
    # could be attributed to weights additionally. If no specific rules are given, then all weights are set to 1.
    if GRAPHMETHOD == 1:
        print(
            "Method 1 generates the graph from self-definition using EDGE description"
        )
        graph = networkx.Graph()
        graph_nodelist = range(N)
        graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
        graph_adjacency = networkx.to_numpy_matrix(
            graph, nodelist=graph_nodelist)
    # Method 2 generates a graph by using its adjacency matrix directly
    elif GRAPHMETHOD == 2:
        print(
            "Method 2 generates the graph from networks using adjacency matrix")
        graph_adjacency = array(
            [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
        graph = networkx.Graph(graph_adjacency)
    else:
        print("Method doesn't exist ")

    output_graph = graph
    output_graph_adjacency = graph_adjacency

    return output_graph, output_graph_adjacency


def H_generator(N, adjacency_matrix):
    """
    This function generates the problem-based Hamiltonian, given the graph with its adjacency matrix description.

    Args:
        N: number of qubits, or number of nodes in the graph, or number of parameters in the classical problem
        adjacency_matrix:  the adjacency matrix generated from the graph encoding the classical problem
    Return:
        H_graph: the problem-based Hamiltonian H generated from the graph_adjacency matrix for the given graph
        H_graph_diag: the real part of the problem-based Hamiltonian H_graph
    """

    sigma_Z = array([[1, 0], [0, -1]])
    H = zeros([2**N, 2**N])

    for row in range(N):
        for col in range(N):
            if abs(adjacency_matrix[N - row - 1, N - col - 1]) and row < col:
                identity_1 = diag(ones([2**row]))
                identity_2 = diag(ones([2**(col - row - 1)]))
                identity_3 = diag(ones([2**(N - col - 1)]))
                H += adjacency_matrix[N - row - 1, N - col - 1] * kron(
                    kron(
                        kron(kron(identity_1, sigma_Z), identity_2), sigma_Z),
                    identity_3, )

    H_graph = H.astype("complex64")
    H_graph_diag = diag(H_graph).real

    return H_graph, H_graph_diag
