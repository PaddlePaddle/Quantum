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
Use MBQC-QAOA algorithm to solve MaxCut problem and compare the solution with the circuit model
"""

from time import perf_counter
from sympy import symbols
from networkx import draw, Graph, circular_layout
import matplotlib.pyplot as plt
from paddle import seed, optimizer
from paddle_quantum.mbqc.QAOA.qaoa import get_solution_string, MBQC_QAOA_Net, Circuit_QAOA_Net

__all__ = [
    "is_graph_valid",
    "plot_graph",
    "graph_to_poly",
    "plot_solution",
    "mbqc_maxcut"
]


def is_graph_valid(graph):
    r"""检查输入的图是否符合规范。

    为了规范输入标准，我们约定输入的图没有孤立节点，即每个节点至少有边与之相连。图的节点编号为连续的自然数 ``"[1, ..., n]"``。

    Args:
       graph (list): 输入的图 ``"G = [V, E]"``，其中，``"V"`` 为节点集合，``"E"`` 为边集合

   代码示例:

    .. code-block:: python

        from paddle_quantum.mbqc.QAOA.maxcut import is_graph_valid
        V = [1,2,3,4]
        E = [(1,2), (2, 3)]
        G = [V, E]
        is_graph_valid(G)

    ::

        ValueError: input graph is not valid!
        The input graph is not valid: there is at least one isolated vertex.
   """
    # Obtain flat list
    V = graph[0]
    E = graph[1]
    flat_list = []
    for edge in E:
        flat_list += list(edge)
    # Check if there are isolated vertices
    if len(set(V).difference(flat_list)) != 0:
        print("The input graph is not valid: there is at least one isolated vertex.")
        raise ValueError("input graph is not valid!")
    else:
        print("The input graph is valid.")


def plot_graph(graph, title, pos=None, node_color=None, style=None):
    r"""定义画图函数。

    Args:
        graph (list): 输入的图 ``G = [V, E]``，其中，``V`` 为节点集合，``E`` 为边集合
        title (str): 画图对应的标题
        pos (dict): 图中各个节点的坐标，以字典的形式输入，例如：{'G': (0, 1)}
        node_color (list): 节点的颜色，与节点的标签的顺序一一对应，例如：['blue', 'red'] 对应节点 ['G1', 'G2']
        style (list): 边的样式，与边的标签的顺序一一对应，例如：['solid', 'solid'] 对应边[('G1', 'G2'), ('G3', 'G4')]

    代码示例:

    .. code-block:: python

        from paddle_quantum.mbqc.QAOA.maxcut import plot_graph
        V = ['0', '1', '2', '3']
        E = [('0', '1'), ('1', '2'), ('2', '3'), ('3', '0')]
        G = [V, E]
        node_color = ['blue', 'yellow', 'red', 'black']
        style = ['-', '--', ':', 'solid']
        pos = {'0': (0, 1),'1': (1, 1), '2': (1, 0), '3': (0, 0)}
        title = 'A demo of "plot_graph"'
        plot_graph(G, title, pos, node_color, style)
    """
    # Obtain graph
    V = graph[0]
    E = graph[1]
    qubit_num = len(V)
    G = Graph()
    G.add_nodes_from(V)
    G.add_edges_from(E)

    # Open the plot figure
    plt.figure()
    plt.ion()
    plt.cla()
    plt.title(title)
    # Set parameters for nodes
    pos = circular_layout(G) if pos is None else pos
    node_color = ["blue" for _ in list(range(qubit_num))] if node_color is None else node_color
    style = ["solid" for (_, _) in E] if style is None else style
    options = {
        "with_labels": True,
        "font_size": 20,
        "font_weight": "bold",
        "font_color": "white",
        "node_size": 2000,
        "width": 2
    }

    # Draw the graph
    draw(G=G, pos=pos, node_color=node_color, style=style, **options)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.pause(1)
    # plt.ioff()
    # plt.show()


def graph_to_poly(graph):
    r"""将图转化为对应 PUBO 问题的目标多项式。

    为了代码规范，我们要求输入的图的节点编号为 ``"[1,...,n]"``。

    Args:
        graph (list): 用户输入的图，图的形式为 ``"[V, E]"``， ``"V" `` 为点集合， ``"E"`` 为边集合

    Returns:
        list: 列表第一个元素为多项式的变量个数，第二个元素为符号化的目标多项式

    代码示例：

    .. code-block:: python

        from paddle_quantum.mbqc.QAOA.maxcut import graph_to_poly
        graph = [[1, 2, 3, 4], [(1, 2), (2, 3), (3, 4), (4, 1)]]
        polynomial = graph_to_poly(graph)
        print("Corresponding objective polynomial of the graph is: \r\n", polynomial)

    ::

        Corresponding objective polynomial of the graph is:
        [4, -2*x_1*x_2 - 2*x_1*x_4 + 2*x_1 - 2*x_2*x_3 + 2*x_2 - 2*x_3*x_4 + 2*x_3 + 2*x_4]
    """
    # Get the vertices and edges
    V = graph[0]
    E = graph[1]
    qubit_num = len(V)

    # Set symbol variables
    vars_x = {i: symbols('x_' + str(i)) for i in range(1, qubit_num + 1)}

    # Get the objective polynomial
    poly_symbol = 0
    for edge in E:
        poly_symbol += vars_x[edge[0]] + vars_x[edge[1]] - 2 * vars_x[edge[0]] * vars_x[edge[1]]
    # Set the polynomial
    polynomial = [qubit_num, poly_symbol]
    return polynomial


def plot_solution(graph, string):
    r"""画出对应最大割问题的解。

    Args:
        graph (list): 用户输入的图，图的形式为 ``"[V, E]"``， ``"V"`` 为点集合， ``"E"`` 为边集合
        string (str): 得到的最终解对应的比特串，列如：``"010101"`` ...

    代码示例:

    .. code-block:: python

        from paddle_quantum.mbqc.QAOA.maxcut import plot_solution
        V = [0, 1, 2, 3]
        E = [(0, 1), (1, 2), (2, 3), (3, 0)]
        G = [V, E]
        plot_solution(G, '1010')
    """
    # Plot the figure of bitstring using matplotlib and networkx
    V = graph[0]
    E = graph[1]
    n = len(V)
    title = "MaxCut solution"
    node_cut = ["blue" if string[v - 1] == "1" else "red" for v in list(range(1, n + 1))]
    edge_cut = ["solid" if string[u - 1] == string[v - 1] else "dashed" for (u, v) in E]
    plot_graph(graph=graph, title=title, node_color=node_cut, style=edge_cut)


def mbqc_maxcut(GRAPH, DEPTH, SEED, LR, ITR, EPOCH, SHOTS=1024):
    r"""定义 MBQC 模型下的 MaxCut 主函数。

    Args:
        GRAPH (list): 输入的图 ``G = [V, E]``，其中，``V`` 为节点集合，``E`` 为边集合
        DEPTH (int): QAOA 算法的深度
        SEED (int): paddle 的训练种子
        LR (float): 学习率
        ITR (int): 单次轮回的迭代次数
        EPOCH (int): 轮回数
        SHOTS (int): 获得最终比特串时设定的测量次数

    Returns:
        list: 列表第一个元素为求得的最优解，第二个元素为对应的目标函数的值

    代码示例：

    .. code-block:: python

        from paddle_quantum.mbqc.QAOA.maxcut import mbqc_maxcut
        V = [1, 2, 3, 4]
        E = [(1, 2), (2, 3), (3, 4), (4, 1)]
        GRAPH = [V, E]
        mbqc_opt = mbqc_maxcut(GRAPH=GRAPH, P=2, SEED=1024, LR=0.1, ITR=120, EPOCH=1, shots=1024)
        print("Optimal solution from MBQC: ", mbqc_opt[0])
        print("Optimal value from MBQC", mbqc_opt[1])

    ::

        Corresponding polynomial is:
        -2*x_1*x_2 - 2*x_1*x_4 + 2*x_1 - 2*x_2*x_3 + 2*x_2 - 2*x_3*x_4 + 2*x_3 + 2*x_4
        iter: 10   loss_MBQC: -3.3919
        iter: 20   loss_MBQC: -3.8858
        iter: 30   loss_MBQC: -3.9810
        iter: 40   loss_MBQC: -3.9582
        iter: 50   loss_MBQC: -3.9967
        iter: 60   loss_MBQC: -3.9972
        iter: 70   loss_MBQC: -3.9999
        iter: 80   loss_MBQC: -3.9997
        iter: 90   loss_MBQC: -3.9999
        iter: 100   loss_MBQC: -4.0000
        iter: 110   loss_MBQC: -4.0000
        iter: 120   loss_MBQC: -4.0000
        训练得到的最优参数 gamma:  [1.57244132 0.78389509]
        训练得到的最优参数 beta:  [5.105461   0.78446647]
        MBQC 模型下训练用时为： 8.373932838439941
        MBQC 模型得到的最优解为： 1010
        MBQC 模型得到的最优值为： 4
    """
    plot_graph(graph=GRAPH, title="Graph to be cut")

    # Obtain the objective polynomial
    polynomial = graph_to_poly(GRAPH)
    print("Corresponding polynomial is:\r\n", polynomial[1])

    start_time = perf_counter()
    seed(SEED)
    mbqc_net = MBQC_QAOA_Net(DEPTH)
    # Choose Adams optimizer (or SGD optimizer)
    opt = optimizer.Adam(learning_rate=LR, parameters=mbqc_net.parameters())
    # opt = optimizer.SGD(learning_rate = LR, parameters = mbqc_net.parameters())

    # Start training
    for epoch in range(EPOCH):
        for itr in range(1, ITR + 1):
            loss, state_out = mbqc_net(poly=polynomial)
            loss.backward()
            opt.minimize(loss)
            opt.clear_grad()
            if itr % 10 == 0:
                print("iter:", itr, "  loss_MBQC:", "%.4f" % loss.numpy())
    print("Optimal parameter gamma: ", mbqc_net.gamma.numpy())
    print("Optimal parameter beta: ", mbqc_net.beta.numpy())
    end_time = perf_counter()
    print("MBQC running time: ", end_time - start_time)

    # Decode the MaxCut solution from the final state
    mbqc_solution = get_solution_string(state_out, SHOTS)
    plot_solution(GRAPH, mbqc_solution)

    # Evaluate the number of cuts
    var_num, poly_symbol = polynomial
    relation = {symbols('x_' + str(j + 1)): int(mbqc_solution[j]) for j in range(var_num)}
    mbqc_value = int(poly_symbol.evalf(subs=relation))
    mbqc_opt = [mbqc_solution, mbqc_value]

    return mbqc_opt


def circuit_maxcut(SEED, GRAPH, DEPTH, LR, ITR, EPOCH, SHOTS):
    r"""定义电路模型下的 MaxCut 主函数。

    Args:
        SEED (int): paddle 的训练种子
        GRAPH (list): 输入的图 ``G = [V, E]``，其中，``V`` 为节点集合，``E`` 为边集合
        DEPTH (int): QAOA 算法的深度
        LR (float): 学习率
        ITR (int): 单次轮回的迭代次数
        EPOCH (int): 轮回数
        SHOTS (int): 获得最终比特串时设定的测量次数

    Returns:
        str: 最大割问题最优解对应的比特串
    """
    plot_graph(graph=GRAPH, title="Graph to be cut")
    start_time_PQ = perf_counter()
    # Set cost Hamiltonian
    H_D_list = [[-0.5, 'z' + str(u - 1) + ',z' + str(v - 1)] for (u, v) in GRAPH[1]]

    # Initialize
    seed(SEED)
    pq_net = Circuit_QAOA_Net(DEPTH, GRAPH, H_D_list)
    opt = optimizer.Adam(learning_rate=LR, parameters=pq_net.parameters())
    for epoch in range(EPOCH):
        for itr in range(1, ITR + 1):
            loss, state = pq_net()
            loss.backward()
            opt.minimize(loss)
            opt.clear_grad()
            if itr % 10 == 0:
                print("iter:", itr, "  loss_cir:", "%.4f" % loss.numpy())
    gamma = pq_net.parameters()[0]
    beta = pq_net.parameters()[1]
    print("Optimal parameter gamma: ", gamma.numpy())
    print("Optimal parameter beta: ", beta.numpy())
    end_time_PQ = perf_counter()
    print("Circuit running time: ", end_time_PQ - start_time_PQ)

    # Obtain the bit string
    prob_measure = state.measure(shots=SHOTS, plot=False)
    cut_bitstring = max(prob_measure, key=prob_measure.get)
    plot_solution(GRAPH, cut_bitstring)
    return cut_bitstring
