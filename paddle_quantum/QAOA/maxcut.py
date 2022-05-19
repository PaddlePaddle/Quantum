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
To learn more about the functions and properties of this application,
you could check the corresponding Jupyter notebook under the Tutorial folder.
"""

import numpy as np
import networkx as nx
import paddle
import paddle_quantum
from paddle_quantum.ansatz import Circuit
from paddle_quantum.loss import ExpecVal
from paddle_quantum import Hamiltonian

__all__ = [
    "maxcut_hamiltonian",
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
    net = Circuit(n)
    net.superposition_layer()
    net.qaoa_layer(E, V, p)
    hamiltonian = Hamiltonian(H_D_list)
    loss_func = ExpecVal(hamiltonian)
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())

    for itr in range(1, ITR + 1):
        state = net()
        loss = -loss_func(state)
        loss.backward()
        opt.minimize(loss)
        opt.clear_grad()
        if print_loss and itr % 10 == 0:
            print("iter:", itr, "  loss:", "%.4f" % loss.numpy())

    state = net()
    prob_measure = state.measure(shots=shots, plot=plot)
    cut_bitstring = max(prob_measure, key=prob_measure.get)

    return cut_bitstring, prob_measure
