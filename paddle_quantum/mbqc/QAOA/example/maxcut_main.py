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
MaxCut main
"""

from networkx.generators.random_graphs import gnp_random_graph
from paddle_quantum.mbqc.QAOA.maxcut import mbqc_maxcut, is_graph_valid, circuit_maxcut


def main():
    r"""MaxCut 主函数。

    """
    # Generate a random graph
    qubit_number = 5  # Number of qubits
    probability_to_generate_edge = 0.7  # The probability to generate an edge randomly
    rand_graph = gnp_random_graph(qubit_number, probability_to_generate_edge)
    V = list(rand_graph.nodes)
    E = list(rand_graph.edges)

    # Make the vertex labels start from 1
    V = [v + 1 for v in V]
    E = [(e[0] + 1, e[1] + 1) for e in E]
    G = [V, E]

    # Note: As the graph is generated randomly,
    # some invalid conditions might EXIST when there is at least one isolated vertex
    # So before our MaxCut solution, we need to check the validity of the generated graph
    print("Input graph is: \n", G)
    is_graph_valid(G)

    # MaxCut under MBQC
    mbqc_result = mbqc_maxcut(
        GRAPH=G,  # Input graph
        DEPTH=4,  # Depth
        SEED=1024,  # Plant Seed
        LR=0.1,  # Learning Rate
        ITR=120,  # Training Iters
        EPOCH=1,  # Epoch Times
        SHOTS=1024  # Shots to get the bit string
    )

    # Print the result from MBQC model
    print("Optimal solution by MBQC: ", mbqc_result[0])
    print("Optimal value by MBQC: ", mbqc_result[1])

    # MaxCut under circuit model
    circuit_result = circuit_maxcut(
        GRAPH=G,  # Input graph, G = [V, E]
        DEPTH=4,  # Depth
        SEED=1024,  # Plant Seed
        LR=0.1,  # Learning Rate
        ITR=120,  # Training Iters
        EPOCH=1,  # Epoch Times
        SHOTS=1024  # Shots to get the bit string
    )

    # Print the result from circuit model
    print("Optimal solution by circuit: ", circuit_result)


if __name__ == '__main__':
    main()
