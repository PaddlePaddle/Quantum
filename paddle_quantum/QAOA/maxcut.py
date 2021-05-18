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
To learn more about the functions and properties of this application,
you could check the corresponding Jupyter notebook under the Tutorial folder.
"""

import paddle
from paddle_quantum.circuit import UAnsatz

import numpy as np
import networkx as nx


__all__ = [
    "maxcut_hamiltonian",
    "circuit_maxcut",
    "find_cut",
]


def maxcut_hamiltonian(E):
    r"""生成最大割问题对应的哈密顿量。

    Args:
        E (list): 图的边

    Returns:
        list: 生成的哈密顿量的列表形式
    """
    H_D_list = []
    for (u, v) in E:
        H_D_list.append([-1.0, 'z' + str(u) + ',z' + str(v)])

    return H_D_list


def circuit_maxcut(E, V, p, gamma, beta):
    r"""构建用于最大割问题的 QAOA 参数化电路。

    Args:
        E: 图的边
        V: 图的顶点
        p: QAOA 电路的层数
        gamma: 与最大割问题哈密顿量相关的电路参数
        beta: 与混合哈密顿量相关的电路参数

    Returns:
        UAnsatz: 构建好的 QAOA 电路
    """
    # Number of qubits needed
    n = len(V)
    cir = UAnsatz(n)
    cir.superposition_layer()
    for layer in range(p):
        for (u, v) in E:
            cir.cnot([u, v])
            cir.rz(gamma[layer], v)
            cir.cnot([u, v])
        for i in V:
            cir.rx(beta[layer], i)

    return cir


class _MaxcutNet(paddle.nn.Layer):
    """
    It constructs the net for maxcut which combines the QAOA circuit with the classical optimizer that sets rules
    to update parameters described by theta introduced in the QAOA circuit.
    """
    def __init__(
        self,
        p,
        dtype="float64",
    ):
        super(_MaxcutNet, self).__init__()

        self.p = p
        self.gamma = self.create_parameter(shape=[self.p],
                                           default_initializer=paddle.nn.initializer.Uniform(low=0.0, high=2 * np.pi),
                                           dtype=dtype, is_bias=False)
        self.beta = self.create_parameter(shape=[self.p],
                                          default_initializer=paddle.nn.initializer.Uniform(low=0.0, high=2 * np.pi),
                                          dtype=dtype, is_bias=False)

    def forward(self, E, V, H_D_list):
        """
        Forward propagation
        """
        cir = circuit_maxcut(E, V, self.p, self.gamma, self.beta)
        cir.run_state_vector()
        loss = -cir.expecval(H_D_list)

        return loss, cir


def find_cut(G, p, ITR, LR, print_loss=False, shots=0, plot=False):
    r"""运行 QAOA 寻找最大割问题的近似解。

    Args:
        G (NetworkX graph): 图
        p (int): QAOA 电路的层数
        ITR (int): 梯度下降优化参数的迭代次数
        LR (float): Adam 优化器的学习率
        print_loss (bool, optional): 优化过程中是否输出损失函数的值，默认为 ``False``，即不输出
        shots (int, optional): QAOA 电路最终输出的量子态的测量次数，默认 0，则返回测量结果的精确概率分布
        plot (bool, optional): 是否绘制测量结果图，默认为 ``False`` ，即不绘制

    Returns:
        tuple: tuple containing:

            string: 寻找到的近似解
            dict: 所有测量结果和其对应的出现次数
    """
    V = list(G.nodes())
    # Map nodes' labels to integers from 0 to |V|-1
    # node_mapping = {V[i]:i for i in range(len(V))}
    # G_mapped = nx.relabel_nodes(G, node_mapping)
    G_mapped = nx.convert_node_labels_to_integers(G)
    V = list(G_mapped.nodes())
    E = list(G_mapped.edges())
    n = len(V)
    H_D_list = maxcut_hamiltonian(E)
    net = _MaxcutNet(p)
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())

    for itr in range(1, ITR + 1):
        loss, cir = net(E, V, H_D_list)
        loss.backward()
        opt.minimize(loss)
        opt.clear_grad()
        if print_loss and itr % 10 == 0:
            print("iter:", itr, "  loss:", "%.4f" % loss.numpy())

    prob_measure = cir.measure(shots=shots, plot=plot)
    cut_bitstring = max(prob_measure, key=prob_measure.get)

    return cut_bitstring, prob_measure
