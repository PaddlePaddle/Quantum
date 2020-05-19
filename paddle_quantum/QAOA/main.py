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
main
"""

from paddle_quantum.QAOA.Paddle_QAOA import Paddle_QAOA
from paddle_quantum.QAOA.QAOA_Prefunc import plot_graph, generate_graph

# Random seed for optimizer
SEED = 1


def main(N=4):
    """
    QAOA Main
    """

    classical_graph, classical_graph_adjacency = generate_graph(N, 1)
    print(classical_graph_adjacency)
    prob_measure = Paddle_QAOA(classical_graph_adjacency)

    # Flatten array[[]] to []
    prob_measure = prob_measure.flatten()
    # Plot it!
    plot_graph(prob_measure, classical_graph, N)


if __name__ == '__main__':
    main()
