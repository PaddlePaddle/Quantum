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
Perform the QAOA algorithm under MBQC model and Circuit model
"""

from collections import Counter
from sympy import symbols, expand
from numpy import pi, random, log2
from paddle import to_tensor, real, abs, zeros, t, conj, matmul, multiply
from paddle import nn
import paddle_quantum
from paddle_quantum.mbqc.utils import kron, basis, permute_systems
from paddle_quantum.mbqc.simulator import MBQC

__all__ = [
    "get_all_indices",
    "var_substitute",
    "symbol_to_list",
    "preprocess",
    "adaptive_angle",
    "byproduct_power",
    "get_cost_hamiltonian",
    "expecval",
    "get_solution_string",
    "mbqc_qaoa",
    "MBQC_QAOA_Net",
    "circuit_qaoa",
    "Circuit_QAOA_Net"
]


def get_all_indices(lst, item):
    r"""查找一个列表中对应元素的全部索引。

    Args:
        lst (list): 输入需要查询的列表
        item (any type): 列表中被索引的某个元素，类型与输入的列表中元素类型对应

    Returns:
        list: 列表中对应元素的全部索引构成的列表

    代码示例:

    .. code-block:: python

        from mbqc_qaoa import get_all_indices
        print(get_all_indices(['1', 0, (2,3), '4', 0, '4'], '4'))

    ::

        [3, 5]
    """
    return [idx for (idx, val) in enumerate(lst) if val == item]


def var_substitute(poly_x):
    r"""多项式变量替换。

    输入为以 x 为自变量的目标函数，输出为以 z 为自变量的目标函数。变换关系为 ``x = (1 - z) / 2``。

    Args:
        poly_x (list): 输入为以 ``x`` 为自变量的符号化的目标函数，列表第一个元素为变量个数，第二个元素为符号化的多项式

    Returns:
        list: 输出为以 ``z`` 为自变量的目标函数，列表第一个元素为变量个数，第二个元素为符号化的多项式

    代码示例:

    .. code-block:: python

        from mbqc_qaoa import var_substitute
        from sympy import symbols
        x_1, x_2, x_3, x_4 = symbols(['x_1', 'x_2', 'x_3', 'x_4'])
        poly_x_symbol = x_1 * x_2 + x_3 * x_4
        poly_x = [4, poly_x_symbol]
        print("The original polynomial is: ", poly_x)
        poly_z = var_substitute(poly_x)
        print("The polynomial after variable substitution is: ", poly_z)

    ::

        The original polynomial is:  [4, x_1*x_2 + x_3*x_4]
        The polynomial after variable substitution is:  [4, z_1*z_2/4 - z_1/4 - z_2/4 + z_3*z_4/4 - z_3/4 - z_4/4 + 1/2]
    """
    # Do variable replacement x = (1 - z) / 2
    var_num, poly_x_symbol = poly_x
    # Set the relations between two variables
    var_relation = {symbols('x_' + str(i)): (1 - symbols('z_' + str(i))) / 2 for i in range(1, var_num + 1)}
    # Obtain polynomial with z symbol
    poly_z_symbol = expand(poly_x_symbol.subs(var_relation))
    # Obtain new polynomial
    poly_z = [var_num, poly_z_symbol]
    return poly_z


def symbol_to_list(poly):
    r"""将符号多项式转化为列表形式。

    Args:
        poly (list): 列表第一个元素为变量个数，第二个元素为符号化的多项式

    Returns:
        list: 列表第一个元素为变量个数，第二个元素为多项式各个单项式构成的列表

    代码示例：

    .. code-block:: python

        from paddle_quantum.mbqc.QAOA.mbqc_qaoa import symbol_to_list
        from sympy import symbols
        z_1, z_2, z_3 = symbols(['z_1', 'z_2', 'z_3'])
        poly_as_symbols = - z_1 * z_2 + 2 * z_3
        poly = [3, poly_as_symbols]
        print("The symbolized polynomial is: ", poly)
        new_poly = symbol_to_list(poly)
        print("Polynomial in the list form: ", new_poly)

    ::

        The symbolized polynomial is: [3, -z_1*z_2 + 2*z_3]
        Polynomial in the list form: [3, [[(0, 0, 1), 2.0], [(1, 1, 0), -1.0]]]
    """
    var_num, poly_as_symbols = poly
    poly_as_terms = poly_as_symbols.as_terms()
    monos = poly_as_terms[0]
    # Double check if the number of variables is correct or not
    # In case there is variable cancellation during the variable substitution
    if len(poly_as_terms[1]) != var_num:
        print("The variables are: " + str(poly_as_terms[1]) + ", whose total number is not: " + str(var_num) + ".")
        raise ValueError("the number of input variables is not correct.")
    # Transform a symbol function to a list form
    poly_as_lists = []
    for mono in monos:
        poly_as_lists.append([mono[1][1], mono[1][0][0]])
    new_poly = [var_num, poly_as_lists]
    # Return new polynomial as lists
    return new_poly


def preprocess(poly_x, depth):
    r"""对多项式进行预处理，提取出多项式变量个数、常数项、线性项、非线性项和 QAOA 图。

    MBQC 模型下的 QAOA 算法依赖于相应的 QAOA 图，该图共有四类节点，我们默认将图中每个节点的标签记为 ``(color, v, k)``，
    其中，color 类型为 (str), 取值为 'R', 'G', 'B' 或 'H'；v 的类型为 (int), 表示该节点在图中的纵向位置；
    k 的类型为 (int)，表示当前节点所在的 QAOA 算法的层数。

    Args:
        poly_x (list): 用户输入的多项式，列表第一个元素为变量个数，第二个元素为符号化的多项式
        depth (int): QAOA 算法的电路深度

    Returns:
        list: 经过变量替换和单项式分类的多项式列表，列表的元素分别为多项式变量个数、常数项、线性项、非线性项
        list: 根据给定的多项式构造出的 QAOA 图，由节点和边构成的列表

    代码示例:

    .. code-block:: python

        from sympy import symbols
        from paddle_quantum.mbqc.QAOA.qaoa import preprocess
        x_1, x_2 = symbols(['x_1', 'x_2'])
        poly_as_symbols = x_1 * x_2
        poly_x = [2, poly_as_symbols]
        poly_processed, qaoa_graph = preprocess(poly_x, 1)
        print("Polynomial after variable substitution and classification: \n", poly_processed)
        print("Corresponding QAOA graph is: \n", qaoa_graph)

    ::

    Polynomial after variable substitution and classification:
     [2, 0.25, {1: -0.25, 2: -0.25}, [[(1, 2), 0.25]]]
    Corresponding QAOA graph is:
     [[('R', (1, 2), 1), ('G', 1, 1), ('G', 2, 1), ('B', 1, 1), ('B', 2, 1), ('H', 1, 1), ('H', 2, 1)],
      [(('R', (1, 2), 1), ('G', 1, 1)), (('R', (1, 2), 1), ('G', 2, 1)), (('G', 1, 1), ('B', 1, 1)),
       (('G', 2, 1), ('B', 2, 1)), (('B', 1, 1), ('H', 1, 1)), (('B', 2, 1), ('H', 2, 1))]]
    """
    # Variable substitute
    poly_z = var_substitute(poly_x)
    # Transform polynomial from symbols to lists
    var_num, poly_as_lists = symbol_to_list(poly_z)

    # cons_item is a number, linear_items is a dict, non_linear_items is a list
    cons_item = 0
    linear_items = {i: 0 for i in range(1, var_num + 1)}
    non_linear_items = []

    for mono in poly_as_lists:
        mono_idx = get_all_indices(lst=mono[0], item=1)
        mono_idx = [ele + 1 for ele in mono_idx]
        if len(mono_idx) == 0:
            cons_item += mono[1]
        elif len(mono_idx) == 1:
            linear_items[mono_idx[0]] = mono[1]
        else:
            non_linear_items.append([tuple(mono_idx), mono[1]])
    poly_classified = [var_num, cons_item, linear_items, non_linear_items]

    # Generate the vertices of the QAOA graph
    graph_v_G = [('G', i, j) for j in range(1, depth + 1) for i in range(1, var_num + 1)]
    graph_v_R = [('R', item[0], j) for j in range(1, depth + 1) for item in non_linear_items]
    graph_v_B = [('B', i, j) for j in range(1, depth + 1) for i in range(1, var_num + 1)]
    graph_v_H = [('H', i, depth) for i in range(1, var_num + 1)]
    graph_v = graph_v_R + graph_v_G + graph_v_B + graph_v_H

    # Generate the edges of the QAOA graph
    graph_e_RG = [(v_R, ('G', v, v_R[-1])) for v_R in graph_v_R for v in v_R[1]]
    graph_e_GB = [(('G', i, j), ('B', i, j)) for j in range(1, depth + 1) for i in range(1, var_num + 1)]
    graph_e_BG = [(('B', i, j), ('G', i, j + 1)) for j in range(1, depth) for i in range(1, var_num + 1)]
    graph_e_BH = [(('B', i, depth), ('H', i, depth)) for i in range(1, var_num + 1)]
    graph_e = graph_e_RG + graph_e_GB + graph_e_BG + graph_e_BH
    qaoa_graph = [graph_v, graph_e]

    return poly_classified, qaoa_graph


def adaptive_angle(which_qubit, graph, outcome, theta, eta):
    r"""定义 QAOA 算法的测量角度。

    图中共有四类节点：红色节点、绿色节点、蓝色节点和输出节点，在 QAOA 算法中，我们需要依次对红色节点、绿色节点、蓝色节点进行测量，
    由于 MBQC 模型适应性测量的特点，本次测量的角度依赖于前面某些节点测量的结果（除第一层外），本次测量的结果也会对后续某些节点的测量产生影响。
    因此，单独定义 QAOA 的测量角度函数，测量时只需调用该函数即可求出对应的测量角度。

    Args:
        which_qubit (tuple): 当前测量节点的标签，形如 ``"(color, v, k)"``
        graph (networkx.classes.graph.Graph): QAOA 算法对应的 QAOA 图
        outcome (dict): 测量结果字典，字典中记录对各个节点的测量结果信息 ``"{qubit label: 0/1}"``
        theta (Tensor): 不考虑副产品影响的角度参数
        eta (Tensor): 变量替换后的多项式系数

    Returns:
        Tensor: 考虑了副产品影响的测量角度
    """
    # Check the Color
    color_label = which_qubit[0]
    assert color_label == 'R' or color_label == 'G' or color_label == 'B', \
        "KeyError: the qubit color index is WRONG, it must be 'R', 'G', or 'B'."
    # Current evolution level
    level = which_qubit[-1]

    outcome_sum = 0
    # Red vertices
    if color_label == 'R':
        for k in range(1, level):
            idx = which_qubit[1]
            for v in idx:
                outcome_sum += outcome[('B', v, k)]
        angle = theta.multiply(to_tensor((((-1) ** (outcome_sum + 1)) * (2 * eta)), dtype='float64'))
        return angle

    # Green vertices
    elif color_label == 'G':
        v = which_qubit[1]
        for k in range(1, level):
            outcome_sum += outcome[('B', v, k)]
        angle = theta.multiply(to_tensor((((-1) ** (outcome_sum + 1)) * (2 * eta)), dtype='float64'))
        return angle

    # Blue vertices
    elif color_label == 'B':
        v = which_qubit[1]
        for k in range(1, level + 1):
            v_r_lst = list(set(list(graph.neighbors(('G', v, k)))).difference([('B', v, k)]))
            if k > 1:
                v_r_lst = list(set(v_r_lst).difference([('B', v, k - 1)]))
            for v_r in v_r_lst:
                outcome_sum += outcome[v_r]
            outcome_sum += outcome[('G', v, k)]
        angle = theta.multiply(to_tensor((((-1) ** (outcome_sum + 1)) * 2), dtype='float64'))
        return angle


def byproduct_power(gate, v, graph, outcome, depth):
    r"""MBQC 模型下 QAOA 算法最后纠正副产品的指数。

    Args:
        gate (str): 需要纠正的副产品的类型，输入为：`X` 或者 `Z`
        v (int): 需要纠正副产品的比特位置
        graph (networkx.classes.graph.Graph): QAOA 算法对应的 QAOA 图
        outcome (dict): 测量结果字典，字典中记录对各个节点的测量结果信息 ``"{qubit label: 0/1}"``
        depth (int): QAOA 算法深度

    Returns:
        int: 需要纠正的副产品的指数
    """
    # gate = 'X' or 'Z', which_qubit = v, outcome
    if gate == 'X':
        power = 0
        for k in range(1, depth + 1):
            power += outcome[('B', v, k)]
    elif gate == 'Z':
        power = 0
        for k in range(1, depth + 1):
            v_r_lst = list(set(list(graph.neighbors(('G', v, k)))).difference([('B', v, k)]))
            if k > 1:
                v_r_lst = list(set(v_r_lst).difference([('B', v, k - 1)]))
            for v_r in v_r_lst:
                power += outcome[v_r]
            power += outcome[('G', v, k)]
    else:
        print("The input parameter 'gate' should be either 'X' or 'Z'.")
        raise KeyError(gate)

    return power


def get_cost_hamiltonian(poly_x):
    r"""获得系统哈密顿量。

    输入为以 ``"x"`` 为自变量的目标函数，构造并返回系统哈密顿量。

    Args:
        poly_x (list): 列表第一个元素为变量个数，第二个元素为符号化的多项式

    Returns:
        Tensor: 输出系统哈密顿量，注意，由于 PUBO 问题中的系统哈密顿量为对角阵，所以直接用列向量进行存储

    代码示例:

    .. code-block:: python

        from sympy import symbols
        from paddle_quantum.mbqc.QAOA.mbqc_qaoa import get_cost_hamiltonian
        x_1, x_2, x_3, x_4 = symbols(['x_1', 'x_2', 'x_3', 'x_4'])
        poly_x_symbol = x_1 * x_2 + x_3 * x_4
        poly_x = [4, poly_x_symbol]
        HC = get_cost_hamiltonian(poly_x)
        print("Hamiltonian is: \n", HC.numpy())

    ::

        系统的哈密顿量为：
        [[0.]
         [0.]
         [0.]
         [1.]
         [0.]
         [0.]
         [0.]
         [1.]
         [0.]
         [0.]
         [0.]
         [1.]
         [1.]
         [1.]
         [1.]
         [2.]]
    """
    # Get the Hamiltonian for system,
    # Note: the expected value of Hamiltonian is not the maximum value of the initial objective function

    poly_z = var_substitute(poly_x)
    var_num, poly_as_lists = symbol_to_list(poly_z)

    Z_col = to_tensor([[1], [-1]], dtype='float64')
    I_col = to_tensor([[1], [1]], dtype='float64')

    # Get the Hamiltonian
    HC = zeros(shape=[2 ** var_num, 1], dtype='float64')
    for mono in poly_as_lists:
        mono_idx = mono[0]
        coeff = mono[1]
        HC_lst = [I_col for _ in range(var_num)]
        for i in range(len(mono_idx)):
            if mono_idx[i] == 1:
                HC_lst[i] = Z_col
        HC += multiply(kron(HC_lst), to_tensor([coeff], dtype='float64'))

    return HC


def expecval(vector, H):
    r"""根据量子态和系统的哈密顿量，计算其期望值。

    此处的期望函数采用矩阵乘法。

    Args:
        vector (Tensor): 当前量子态的列向量
        H (Tensor): 哈密顿量的列向量

    Warning:
        MBQC 模型下的损失函数与电路模型下的损失函数在定义上有所差别，差了常数项，所以损失函数的值有所差别，
        但这并不影响参数学习和梯度下降算法对参数的训练过程，最终求得的最大割都是该问题的最优解。

    Returns:
        Tensor: 系统哈密顿量 H 在该量子态下的期望值
    """
    # state and H are both of column vectors
    expec_val = matmul(t(conj(vector)), multiply(H, vector))
    complex128_tensor = to_tensor([], dtype='complex128')
    if expec_val.dtype == complex128_tensor.dtype:
        expec_val = real(expec_val)
    return expec_val


def get_solution_string(vector, shots=1024):
    r"""从最后的量子态中解码原问题的答案。

    对得到的态进行多次测量，得到概率分布，找到概率最大的比特串即为最终解。

    Args:
        vector (Tensor): 量子态的列向量
        shots (int): 测量次数

    Returns:
        str: 输出 PUBO 问题的其中一个最优解
    """
    # Calculate the probability for each string
    vec_len = vector.shape[0]
    vec_size = int(log2(vec_len))
    prob_array = abs(vector)
    prob_array = multiply(prob_array, prob_array)
    prob_array = prob_array.reshape([vec_len]).numpy()

    # Measure the state 1024 times in order to get the probability distribution
    samples = random.choice(range(vec_len), shots, p=prob_array)
    count_samples = Counter(samples)
    # Transform the samples to binary strings
    bit_freq = {}
    for idx in count_samples:
        bit_freq[bin(idx)[2:].zfill(vec_size)] = count_samples[idx]

    max_str = max(bit_freq, key=bit_freq.get)
    return max_str


def mbqc_qaoa(poly_x, depth, gamma, beta):
    r"""MBQC 模型下的 QAOA 算法。

    Args:
        poly_x (list): 用户输入的多项式，列表第一个元素为变量个数，第二个元素为符号化的多项式
        depth (int): 电路深度
        gamma (Tensor): 待训练角度变量 gamma
        beta (Tensor): 待训练角度变量 beta

    Returns:
        Tensor: QAOA 算法结束后输出的量子态列向量
    """
    # Pre-process of the objective function
    poly_classified, qaoa_graph = preprocess(poly_x, depth)
    var_num, cons_item, linear_items, non_linear_items = poly_classified

    # Initialize a MBQC class
    mbqc = MBQC()
    mbqc.set_graph(graph=qaoa_graph)

    # Measure every single qubits
    for i in range(1, depth + 1):
        # Measure red vertices
        for item in non_linear_items:
            angle_r = adaptive_angle(which_qubit=('R', item[0], i),
                                     graph=mbqc.get_graph(),
                                     outcome=mbqc.get_classical_output(),
                                     theta=gamma[i - 1],
                                     eta=to_tensor(item[1], dtype='float64')
                                     )
            mbqc.measure(which_qubit=('R', item[0], i), basis_list=basis('YZ', angle_r))

        # Measure green vertices
        for v in range(1, var_num + 1):
            angle_g = adaptive_angle(which_qubit=('G', v, i),
                                     graph=mbqc.get_graph(),
                                     outcome=mbqc.get_classical_output(),
                                     theta=gamma[i - 1],
                                     eta=linear_items[v])
            mbqc.measure(which_qubit=('G', v, i), basis_list=basis('XY', angle_g))

        # Measure blue vertices
        for v in range(1, var_num + 1):
            angle_b = adaptive_angle(which_qubit=('B', v, i),
                                     graph=mbqc.get_graph(),
                                     outcome=mbqc.get_classical_output(),
                                     theta=beta[i - 1],
                                     eta=to_tensor([1], dtype='float64'))
            mbqc.measure(which_qubit=('B', v, i), basis_list=basis('XY', angle_b))

    # Correct the byproduct operators
    for v in range(1, var_num + 1):
        graph = mbqc.get_graph()
        outcome = mbqc.get_classical_output()
        pow_x = byproduct_power(gate='X', v=v, graph=graph, outcome=outcome, depth=depth)
        pow_z = byproduct_power(gate='Z', v=v, graph=graph, outcome=outcome, depth=depth)
        mbqc.correct_byproduct(gate='X', which_qubit=('H', v, depth), power=pow_x)
        mbqc.correct_byproduct(gate='Z', which_qubit=('H', v, depth), power=pow_z)
    output_label = [('H', i, depth) for i in range(1, var_num + 1)]

    # Permute the system order to fit output_label
    state_out = mbqc.get_quantum_output()
    state_out = permute_systems(state_out, output_label)
    return state_out.vector


class MBQC_QAOA_Net(nn.Layer):
    r"""定义 MBQC 模型下的 QAOA 优化网络，用于实例化一个 MBQC - QAOA 优化网络。

    Attributes:
        depth (int): QAOA 算法深度

    Returns:
        Tensor: 输出损失函数
        Tensor: 输出量子态列向量
    """

    def __init__(
            self,
            depth,  # Depth
            dtype="float64",
    ):
        r"""定义 MBQC 模型下的 QAOA 优化网络。

        Args:
            depth (int): QAOA 算法深度
        """
        super(MBQC_QAOA_Net, self).__init__()
        self.depth = depth
        # Define the training parameters
        self.gamma = self.create_parameter(shape=[self.depth],
                                           default_initializer=nn.initializer.Uniform(low=0.0, high=2 * pi),
                                           dtype=dtype,
                                           is_bias=False)
        self.beta = self.create_parameter(shape=[self.depth],
                                          default_initializer=nn.initializer.Uniform(low=0.0, high=2 * pi),
                                          dtype=dtype,
                                          is_bias=False)

    def forward(self, poly):
        r"""定义优化网络的前向传播机制。

        Args:
            poly (list): 用户输入的多项式，列表第一个元素为变量个数，第二个元素为符号化的多项式

        Returns:
            Tensor: 输出损失函数
            Tensor: 输出量子态列向量
        """
        # Initial the MBQC - QAOA algorithm and return the state out
        vec_out = mbqc_qaoa(poly, self.depth, self.gamma, self.beta)
        # Get cost Hamiltonian
        HC = get_cost_hamiltonian(poly)
        # Calculate loss
        loss = -expecval(vec_out, HC)
        return loss, vec_out


def circuit_qaoa(graph, depth, gamma, beta):
    r"""使用 UAnsatz 电路模型实现 QAOA。

    Args:
        graph (list): 图 [V, E]，V 是点集合，E 是边集合
        depth (int): 电路深度
        gamma (Tensor): 待训练角度变量 gamma
        beta (Tensor): 待训练角度变量 beta

    Returns:
        UAnsatz: 电路模型的 UAnsatz 电路
    """
    vertices = graph[0]
    edges = graph[1]
    qubit_number = len(vertices)
    cir = paddle_quantum.ansatz.Circuit(qubit_number)
    cir.superposition_layer()
    for layer in range(depth):
        for (u, v) in edges:
            u = u - 1
            v = v - 1
            cir.cnot([u, v])
            cir.rz(gamma[layer], v)
            cir.cnot([u, v])
        for v in range(qubit_number):
            cir.rx(beta[layer], v)
    return cir


class Circuit_QAOA_Net(paddle_quantum.gate.Gate):
    r"""定义电路模型下的 QAOA 优化网络，用于实例化一个电路模型下 QAOA 的优化网络。

    Attributes:
        depth (int): QAOA 算法深度

    Returns:
        Tensor: 输出损失函数
        Tensor: 输出量子态列向量
    """

    def __init__(self, depth, graph, H):
        r"""定义电路模型下的 QAOA 优化网络。

        Args:
            depth (int): QAOA 算法深度
        """
        super().__init__()
        V = graph[0]
        E = graph[1]
        V = [item - 1for item in V]
        E = [(edge[0] - 1, edge[1] - 1) for edge in E]
        n = len(V)
        self.net = paddle_quantum.ansatz.Circuit(n)
        self.net.superposition_layer()
        self.net.qaoa_layer(E, V, depth)
        hamiltonian = paddle_quantum.Hamiltonian(H)
        self.loss_func = paddle_quantum.loss.ExpecVal(hamiltonian)

    def forward(self):
        r"""定义优化网络的前向传播机制。

        Args:
            graph (list): 图 [V, E]，V 是点集合，E 是边集合
            H (list): 哈密顿量列表

        Returns:
            Tensor: 输出损失函数
            UAnsatz: 电路模型的 UAnsatz 电路
        """
        state = self.net()
        loss = self.loss_func(state)
        return loss, state
