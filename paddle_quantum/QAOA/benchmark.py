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
Benchmark
"""
from matplotlib import pyplot
from numpy import max, min, load, ones
from paddle_quantum.QAOA.QAOA_Prefunc import generate_graph, H_generator


def benchmark_QAOA(classical_graph_adjacency=None, N=None):
    """
     This function benchmarks the performance of QAOA. Indeed, it compares its approximate solution obtained
     from QAOA with predetermined parameters, such as iteration step = 120 and learning rate = 0.1, to the exact solution
     to the classical problem.

    """
    # Generate the graph and its adjacency matrix from the classical problem, such as the Max-Cut problem
    if all(var is None for var in (classical_graph_adjacency, N)):
        N = 4
        _, classical_graph_adjacency = generate_graph(N, 1)

    # Compute the exact solution of the original problem to benchmark the performance of QAOA
    _, H_problem_diag = H_generator(N, classical_graph_adjacency)

    H_graph_max = max(H_problem_diag)
    H_graph_min = min(H_problem_diag)
    print('H_max:', H_graph_max, '  H_min:', H_graph_min)

    # Load the data of QAOA
    x1 = load('./output/summary_data.npz')

    H_min = ones([len(x1['iter'])]) * H_graph_min

    # Plot it
    pyplot.figure(1)
    loss_QAOA, = pyplot.plot(x1['iter'], x1['energy'], \
                                        alpha=0.7, marker='', linestyle="--", linewidth=2, color='m')
    benchmark, = pyplot.plot(
        x1['iter'],
        H_min,
        alpha=0.7,
        marker='',
        linestyle=":",
        linewidth=2,
        color='b')
    pyplot.xlabel('Number of iteration')
    pyplot.ylabel('Performance of the loss function for QAOA')

    pyplot.legend(
        handles=[loss_QAOA, benchmark],
        labels=[
            r'Loss function $\left\langle {\psi \left( {\bf{\theta }} \right)} '
            r'\right|H\left| {\psi \left( {\bf{\theta }} \right)} \right\rangle $',
            'The benchmark result',
        ],
        loc='best')

    # Show the picture
    pyplot.show()


def main():
    """
    main
    """

    benchmark_QAOA()


if __name__ == '__main__':
    main()
