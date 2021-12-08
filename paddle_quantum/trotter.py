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
Trotter Hamiltonian time evolution circuit module
"""

from paddle_quantum.utils import Hamiltonian
from paddle_quantum.circuit import UAnsatz
from collections import defaultdict
import warnings
import numpy as np
import re
import paddle

PI = paddle.to_tensor(np.pi, dtype='float64')


def construct_trotter_circuit(
    circuit: UAnsatz,
    hamiltonian: Hamiltonian,
    tau: float,
    steps: int,
    method: str = 'suzuki',
    order: int = 1,
    grouping: str = None,
    coefficient: np.ndarray or paddle.Tensor = None,
    permutation: np.ndarray = None
):
    r"""向 circuit 的后面添加 trotter 时间演化电路，即给定一个系统的哈密顿量 H，该电路可以模拟系统的时间演化 :math:`U_{cir}~ e^{-iHt}` 。

    Args:
        circuit (UAnsatz): 需要添加时间演化电路的 UAnsatz 对象
        hamiltonian (Hamiltonian): 需要模拟时间演化的系统的哈密顿量 H
        tau (float): 每个 trotter 块的演化时间长度
        steps (int): 添加多少个 trotter 块（提示： ``steps * tau`` 即演化的时间总长度 t）
        method (str): 搭建时间演化电路的方法，默认为 ``'suzuki'`` ，即使用 Trotter-Suzuki 分解。可以设置为 ``'custom'`` 来使用自定义的演化策略
                    （需要用 permutation 和 coefficient 来定义）
        order (int): Trotter-Suzuki decomposition 的阶数，默认为 ``1`` ，仅在使用 ``method='suzuki'`` 时有效
        grouping (str): 是否对哈密顿量进行指定策略的重新排列，默认为 ``None`` ，支持 ``'xyz'`` 和 ``'even_odd'`` 两种方法
        coefficient (array or Tensor): 自定义时间演化电路的系数，对应哈密顿量中的各项，默认为 ``None`` ，仅在 ``method='custom'`` 时有效
        permutation (array): 自定义哈密顿量的排列方式，默认为 ``None`` ，仅在 ``method='custom'`` 时有效

    代码示例：

    .. code-block:: python

        from paddle_quantum.utils import Hamiltonian
        from paddle_quantum.circuit import UAnsatz
        from paddle_quantum.trotter import construct_trotter_circuit, get_1d_heisenberg_hamiltonian
        import numpy as np

        h = get_1d_heisenberg_hamiltonian(length=3)
        cir = UAnsatz(h.n_qubits)
        t = 1
        r = 10
        # 1st order product formula (PF) circuit
        construct_trotter_circuit(cir, h, tau=t/r, steps=r)
        # 2nd order product formula (PF) circuit
        construct_trotter_circuit(cir, h, tau=t/r, steps=r, order=2)
        # higher order product formula (PF) circuit
        construct_trotter_circuit(cir, h, tau=t/r, steps=r, order=10)

        # customize coefficient and permutation
        # the following codes is equivalent to adding the 1st order PF
        permutation = np.arange(h.n_terms)
        coefficients = np.ones(h.n_terms)
        construct_trotter_circuit(cir, h, tau=t/r, steps=r, method='custom',
                                  permutation=permutation, coefficient=coefficients)

    Hint:
        对于该函数的原理以及其使用方法更详细的解释，可以参考量桨官网中量子模拟部分的教程 https://qml.baidu.com/tutorials/overview.html
    """
    # check the legitimacy of the inputs (coefficient and permutation)
    def check_input_legitimacy(arg_in):
        if not isinstance(arg_in, np.ndarray) and not isinstance(arg_in, paddle.Tensor):
            arg_out = np.array(arg_in)
        else:
            arg_out = arg_in

        if arg_out.ndim == 1 and isinstance(arg_out, np.ndarray):
            arg_out = arg_out.reshape(1, arg_out.shape[0])
        elif arg_out.ndim == 1 and isinstance(arg_out, paddle.Tensor):
            arg_out = arg_out.reshape([1, arg_out.shape[0]])

        return arg_out

    # check compatibility between input method and customization arguments
    if (permutation is not None or coefficient is not None) and (method != 'custom'):
        warning_message = 'method {} is not compatible with customized permutation ' \
                          'or coefficient and will be overlooked'.format(method)
        method = 'custom'
        warnings.warn(warning_message, RuntimeWarning)

    # check the legitimacy of inputs
    if method == 'suzuki':
        if order > 2 and order % 2 != 0 and type(order) != int:
            raise ValueError('The order of the trotter-suzuki decomposition should be either 1, 2 or 2k (k an integer)'
                             ', got order = %i' % order)

    # check and reformat inputs for 'custom' mode
    elif method == 'custom':
        # check permutation
        if permutation is not None:
            permutation = np.array(check_input_legitimacy(permutation), dtype=int)
            # give a warning for using permutation and grouping at the same time
            if grouping:
                warning_message = 'Using specified permutation and automatic grouping {} at the same time, the ' \
                                  'permutation will act on the grouped Hamiltonian'.format(grouping)
                warnings.warn(warning_message, RuntimeWarning)
        # check coefficient
        if coefficient is not None:
            coefficient = check_input_legitimacy(coefficient)
        # if the permutation is not specified, then set it to [[1, 2, ...], ...]
        if coefficient is not None and permutation is None:
            permutation = np.arange(hamiltonian.n_terms) if coefficient.ndim == 1 \
                else np.arange(hamiltonian.n_terms).reshape(1, hamiltonian.n_terms).repeat(coefficient.shape[0], axis=0)
            permutation = np.arange(hamiltonian.n_terms).reshape(1, hamiltonian.n_terms)
            permutation = permutation.repeat(coefficient.shape[0], axis=0)
        # if the coefficient is not specified, set a uniform (normalized) coefficient
        if permutation is not None and coefficient is None:
            coefficient = 1 / len(permutation) * np.ones_like(permutation)
        # the case that the shapes of input coefficient and permutations don't match
        if tuple(permutation.shape) != tuple(coefficient.shape):
            # case not-allowed
            if permutation.shape[1] != coefficient.shape[1]:
                raise ValueError('Shape of the permutation and coefficient array don\'t match, got {} and {}'.format(
                    tuple(permutation.shape), tuple(coefficient.shape)
                ))
            # cases can be fixed by repeating one of the two inputs
            elif permutation.shape[0] != coefficient.shape[0] and permutation[0] == 1:
                permutation = permutation.repeat(coefficient.shape[0])
            elif permutation.shape[0] != coefficient.shape[0] and coefficient[0] == 1:
                if isinstance(coefficient, paddle.Tensor):
                    coefficient = paddle.stack([coefficient for _ in range(permutation.shape[0])])\
                        .reshape([permutation.shape[0], permutation.shape[1]])
                elif isinstance((coefficient, np.ndarray)):
                    coefficient = coefficient.repeat(permutation.shape[0])

    # group the hamiltonian according to the input
    if not grouping:
        grouped_hamiltonian = [hamiltonian]
    elif grouping == 'xyz':
        grouped_hamiltonian = __group_hamiltonian_xyz(hamiltonian=hamiltonian)
    elif grouping == 'even_odd':
        grouped_hamiltonian = __group_hamiltonian_even_odd(hamiltonian=hamiltonian)
    else:
        raise ValueError("Grouping method %s is not supported, valid key words: 'xyz', 'even_odd'" % grouping)

    # apply trotter blocks
    for step in range(steps):
        if method == 'suzuki':
            _add_trotter_block(circuit=circuit, tau=tau, grouped_hamiltonian=grouped_hamiltonian, order=order)
        elif method == 'custom':
            _add_custom_block(circuit=circuit, tau=tau, grouped_hamiltonian=grouped_hamiltonian,
                              custom_coefficients=coefficient, permutation=permutation)
        else:
            raise ValueError("The method %s is not supported, valid method keywords: 'suzuki', 'custom'" % method)


def __get_suzuki_num(order):
    r"""计算阶数为 order 的 suzuki product formula 的 trotter 数。
    """
    if order == 1 or order == 2:
        n_suzuki = order
    elif order > 2 and order % 2 == 0:
        n_suzuki = 2 * 5 ** (order // 2 - 1)
    else:
        raise ValueError('The order of the trotter-suzuki decomposition should be either 1, 2 or 2k (k an integer)'
                         ', got order = %i' % order)

    return n_suzuki


def __sort_pauli_word(pauli_word, site):
    r""" 将 pauli_word 按照 site 的大小进行排列，并同时返回排序后的 pauli_word 和 site。

    Note:
        这是一个内部函数，一般你不需要直接使用它。
    """
    sort_index = np.argsort(np.array(site))
    return ''.join(np.array(list(pauli_word))[sort_index].tolist()), np.array(site)[sort_index]


def _add_trotter_block(circuit, tau, grouped_hamiltonian, order):
    r""" 添加一个 trotter 块，i.e. :math:`e^{-iH\tau}`，并使用 Trotter-Suzuki 分解对其进行展开。

    Args:
        circuit (UAnsatz): 需要添加 trotter 块的电路
        tau (float or tensor): 该 trotter 块的演化时间
        grouped_hamiltonian (list): 一个由 Hamiltonian 对象组成的列表，该函数会默认该列表中的哈密顿量为 Trotter-Suzuki 展开的基本项
        order (int): Trotter-Suzuki 展开的阶数

    Note (关于 grouped_hamiltonian 的使用方法):
        以二阶的 trotter-suzki 分解 S2(t) 为例，若 grouped_hamiltonian = [H_1, H_2]，则会按照
        (H_1, t/2)(H_2, t/2)(H_2, t/2)(H_1, t/2) 的方法进行添加 trotter 电路
        特别地，若用户没有预先对 Hamiltonian 进行 grouping 的话，传入一个单个的 Hamiltonian 对象，则该函数会按照该 Hamiltonian
        的顺序进行正则（canonical）的分解：依然以二阶 trotter 为例，若传入单个 H，则添加 (H[0:-1:1], t/2)(H[-1:0:-1], t/2) 的电路

    Warning:
        本函数一般情况下为内部函数，不会对输入的合法性进行检测和尝试修正。推荐使用 construct_trotter_circuit() 来构建时间演化电路
    """
    if order == 1:
        __add_first_order_trotter_block(circuit, tau, grouped_hamiltonian)
    elif order == 2:
        __add_second_order_trotter_block(circuit, tau, grouped_hamiltonian)
    else:
        __add_higher_order_trotter_block(circuit, tau, grouped_hamiltonian, order)
    pass


def _add_custom_block(circuit, tau, grouped_hamiltonian, custom_coefficients, permutation):
    r""" 添加一个自定义形式的 trotter 块

    Args:
        circuit (UAnsatz): 需要添加 trotter 块的电路
        tau (float or tensor): 该 trotter 块的演化时间
        grouped_hamiltonian (list): 一个由 Hamiltonian 对象组成的列表，该函数会默认该列表中的哈密顿量为 trotter-suzuki 展开的基本项
        order (int): trotter-suzuki 展开的阶数
        permutation (np.ndarray): 自定义置换
        custom_coefficients (np.ndarray or Tensor): 自定义系数

    Warning:
        本函数一般情况下为内部函数，不会对输入的合法性进行检测和尝试修正。推荐使用 construct_trotter_circuit() 来构建时间演化电路
    """

    # combine the grouped hamiltonian into one single hamiltonian
    hamiltonian = sum(grouped_hamiltonian, Hamiltonian([]))

    # apply trotter circuit according to coefficient and
    h_coeffs, pauli_words, sites = hamiltonian.decompose_with_sites()
    for i in range(permutation.shape[0]):
        for term_index in range(permutation.shape[1]):
            custom_coeff = custom_coefficients[i][term_index]
            term_index = permutation[i][term_index]
            pauli_word, site = __sort_pauli_word(pauli_words[term_index], sites[term_index])
            coeff = h_coeffs[term_index] * custom_coeff
            add_n_pauli_gate(circuit, 2 * tau * coeff, pauli_word, site)


def __add_first_order_trotter_block(circuit, tau, grouped_hamiltonian, reverse=False):
    r""" 添加一阶 trotter-suzuki 分解的时间演化块

    Notes:
        这是一个内部函数，你不需要使用它
    """
    if not reverse:
        for hamiltonian in grouped_hamiltonian:
            assert isinstance(hamiltonian, Hamiltonian)
            
            #将原哈密顿量中相同site的XX，YY，ZZ组合到一起
            grouped_hamiltonian = []
            coeffs, pauli_words, sites = hamiltonian.decompose_with_sites()
            grouped_terms_indices = []
            left_over_terms_indices = []
            d = defaultdict(list)
            #合并相同site的XX,YY,ZZ
            for term_index in range(len(coeffs)):
                site = sites[term_index]
                pauli_word = pauli_words[term_index]
                for pauli in ['XX', 'YY', 'ZZ']:
                    assert isinstance(pauli_word, str), "Each pauli word should be a string type"
                    if (pauli_word==pauli or pauli_word==pauli.lower()):
                        key = tuple(sorted(site))
                        d[key].append((pauli,term_index))
                        if len(d[key])==3:
                            terms_indices_to_be_grouped = [x[1] for x in d[key]]
                            grouped_terms_indices.extend(terms_indices_to_be_grouped)
                            grouped_hamiltonian.append(hamiltonian[terms_indices_to_be_grouped])
            #其他的剩余项
            for term_index in range(len(coeffs)):
                if term_index not in grouped_terms_indices:
                    left_over_terms_indices.append(term_index)
            if len(left_over_terms_indices):
                for term_index in left_over_terms_indices:
                    grouped_hamiltonian.append(hamiltonian[term_index])
            #得到新的哈密顿量
            res = grouped_hamiltonian[0]
            for i in range(1,len(grouped_hamiltonian)):
                res+=grouped_hamiltonian[i]
            hamiltonian = res
            
            # decompose the Hamiltonian into 3 lists
            coeffs, pauli_words, sites = hamiltonian.decompose_with_sites()
            # apply rotational gate of each term
            term_index = 0
            while term_index <len(coeffs):
                if term_index+3<=len(coeffs) and \
                len(set(y for x in sites[term_index:term_index+3] for y in x ))==2 and\
                set(pauli_words[term_index:term_index+3])=={'XX','YY','ZZ'}:
                    optimal_circuit(circuit,[tau*i for i in coeffs[term_index:term_index+3]],sites[term_index])
                    term_index+=3
                else:
                    # get the sorted pauli_word and site (an array of qubit indices) according to their qubit indices
                    pauli_word, site = __sort_pauli_word(pauli_words[term_index], sites[term_index])
                    add_n_pauli_gate(circuit, 2 * tau * coeffs[term_index], pauli_word, site)
                    term_index+=1
    # in the reverse mode, if the Hamiltonian is a single element list, reverse the order its each term
    else:
        if len(grouped_hamiltonian) == 1:
            coeffs, pauli_words, sites = grouped_hamiltonian[0].decompose_with_sites()
            for term_index in reversed(range(len(coeffs))):
                pauli_word, site = __sort_pauli_word(pauli_words[term_index], sites[term_index])
                add_n_pauli_gate(circuit, 2 * tau * coeffs[term_index], pauli_word, site)
        # otherwise, if it is a list of multiple Hamiltonian, only reverse the order of that list
        else:
            for hamiltonian in reversed(grouped_hamiltonian):
                assert isinstance(hamiltonian, Hamiltonian)
                coeffs, pauli_words, sites = hamiltonian.decompose_with_sites()
                for term_index in range(len(coeffs)):
                    pauli_word, site = __sort_pauli_word(pauli_words[term_index], sites[term_index])
                    add_n_pauli_gate(circuit, 2 * tau * coeffs[term_index], pauli_word, site)


def __add_second_order_trotter_block(circuit, tau, grouped_hamiltonian):
    r""" 添加二阶 trotter-suzuki 分解的时间演化块

    Notes:
        这是一个内部函数，你不需要使用它
    """
    __add_first_order_trotter_block(circuit, tau / 2, grouped_hamiltonian)
    __add_first_order_trotter_block(circuit, tau / 2, grouped_hamiltonian, reverse=True)


def __add_higher_order_trotter_block(circuit, tau, grouped_hamiltonian, order):
    r""" 添加高阶（2k 阶） trotter-suzuki 分解的时间演化块

    Notes:
        这是一个内部函数，你不需要使用它
    """
    assert order % 2 == 0
    p_values = get_suzuki_p_values(order)
    if order - 2 != 2:
        for p in p_values:
            __add_higher_order_trotter_block(circuit, p * tau, grouped_hamiltonian, order - 2)
    else:
        for p in p_values:
            __add_second_order_trotter_block(circuit, p * tau, grouped_hamiltonian)


def add_n_pauli_gate(circuit, theta, pauli_word, which_qubits):
    r""" 添加一个对应着 N 个泡利算符张量积的旋转门，例如 :math:`e^{-\theta/2 * X \otimes I \otimes X \otimes Y}`

    Args:
        circuit (UAnsatz): 需要添加门的电路
        theta (tensor or float): 旋转角度
        pauli_word (str): 泡利算符组成的字符串，例如 ``"XXZ"``
        which_qubits (list or np.ndarray): ``pauli_word`` 中的每个算符所作用的量子比特编号
    """
    if isinstance(which_qubits, tuple) or isinstance(which_qubits, list):
        which_qubits = np.array(which_qubits)
    elif not isinstance(which_qubits, np.ndarray):
        raise ValueError('which_qubits should be either a list, tuple or np.ndarray')

    if not isinstance(theta, paddle.Tensor):
        theta = paddle.to_tensor(theta, dtype='float64')
    # the following assert is not working properly
    # assert isinstance(circuit, UAnsatz), 'the circuit should be an UAnstaz object'

    # if it is a single-Pauli case, apply the single qubit rotation gate accordingly
    if len(which_qubits) == 1:
        if re.match(r'X', pauli_word[0], flags=re.I):
            circuit.rx(theta, which_qubit=which_qubits[0])
        elif re.match(r'Y', pauli_word[0], flags=re.I):
            circuit.ry(theta, which_qubit=which_qubits[0])
        elif re.match(r'Z', pauli_word[0], flags=re.I):
            circuit.rz(theta, which_qubit=which_qubits[0])

    # if it is a multiple-Pauli case, implement a Pauli tensor rotation
    # we use a scheme described in 4.7.3 of Nielson & Chuang, that is, basis change + tensor Z rotation
    # (tensor Z rotation is 2 layers of CNOT and a Rz rotation)
    else:
        which_qubits.sort()

        # Change the basis for qubits on which the acting operators are not 'Z'
        for qubit_index in range(len(which_qubits)):
            if re.match(r'X', pauli_word[qubit_index], flags=re.I):
                circuit.h(which_qubits[qubit_index])
            elif re.match(r'Y', pauli_word[qubit_index], flags=re.I):
                circuit.rx(PI / 2, which_qubits[qubit_index])

        # Add a Z tensor n rotational gate
        for i in range(len(which_qubits) - 1):
            circuit.cnot([which_qubits[i], which_qubits[i + 1]])
        circuit.rz(theta, which_qubits[-1])
        for i in reversed(range(len(which_qubits) - 1)):
            circuit.cnot([which_qubits[i], which_qubits[i + 1]])

        # Change the basis for qubits on which the acting operators are not 'Z'
        for qubit_index in range(len(which_qubits)):
            if re.match(r'X', pauli_word[qubit_index], flags=re.I):
                circuit.h(which_qubits[qubit_index])
            elif re.match(r'Y', pauli_word[qubit_index], flags=re.I):
                circuit.rx(- PI / 2, which_qubits[qubit_index])
                
def optimal_circuit(circuit,theta,which_qubits):
    r""" 添加一个优化电路，哈密顿量为'XXYYZZ'`

    Args:
        circuit (UAnsatz): 需要添加门的电路
        theta list(tensor or float): 旋转角度需要传入三个参数
        which_qubits (list or np.ndarray): ``pauli_word`` 中的每个算符所作用的量子比特编号
    """
    p = np.pi/2
    x,y,z = theta
    alpha = paddle.to_tensor(3*p-4*x*p+2*x,dtype='float64')
    beta = paddle.to_tensor(-3*p+4*y*p-2*y,dtype='float64')
    gamma = paddle.to_tensor(2*z-p,dtype='float64')
    which_qubits.sort()
    a,b = which_qubits
    circuit.rz(paddle.to_tensor(p,dtype='float64'),b)
    circuit.cnot([b,a])
    circuit.rz(gamma,a)
    circuit.ry(alpha,b)
    circuit.cnot([a,b])
    circuit.ry(beta,b)
    circuit.cnot([b,a])
    circuit.rz(paddle.to_tensor(-p,dtype='float64'),a)

def __group_hamiltonian_xyz(hamiltonian):
    r""" 将哈密顿量拆分成 X、Y、Z 以及剩余项四个部分，并返回由他们组成的列表

    Args:
        hamiltonian (Hamiltonian): Paddle Quantum 中的 Hamiltonian 类

    Notes:
        X、Y、Z 项分别指的是该项的 Pauli word 只含有 X、Y、Z，例如 'XXXY' 就会被分类到剩余项
    """
    grouped_hamiltonian = []
    coeffs, pauli_words, sites = hamiltonian.decompose_with_sites()
    grouped_terms_indices = []
    left_over_terms_indices = []
    for pauli in ['X', 'Y', 'Z']:
        terms_indices_to_be_grouped = []
        for term_index in range(len(coeffs)):
            pauli_word = pauli_words[term_index]
            assert isinstance(pauli_word, str), "Each pauli word should be a string type"
            if pauli_word.count(pauli) == len(pauli_word) or pauli_word.count(pauli.lower()) == len(pauli_word):
                terms_indices_to_be_grouped.append(term_index)
        grouped_terms_indices.extend(terms_indices_to_be_grouped)
        grouped_hamiltonian.append(hamiltonian[terms_indices_to_be_grouped])

    for term_index in range(len(coeffs)):
        if term_index not in grouped_terms_indices:
            left_over_terms_indices.append(term_index)
    if len(left_over_terms_indices):
        for term_index in left_over_terms_indices:
            grouped_hamiltonian.append(hamiltonian[term_index])
    return grouped_hamiltonian


def __group_hamiltonian_even_odd(hamiltonian):
    r""" 将哈密顿量拆分为奇数项和偶数项两部分

    Args:
        hamiltonian (Hamiltonian):

    Warning:
        注意该分解方法并不能保证拆分后的奇数项和偶数项内部一定相互对易，因此不正确的使用该方法反而会增加 trotter 误差。
        请在使用该方法前检查哈密顿量是否为可以进行奇偶分解：例如一维最近邻相互作用系统的哈密顿量可以进行奇偶分解
    """
    grouped_hamiltonian = []
    coeffs, pauli_words, sites = hamiltonian.decompose_with_sites()
    grouped_terms_indices = []
    left_over_terms_indices = []

    for offset in range(2):
        terms_indices_to_be_grouped = []
        for term_index in range(len(coeffs)):
            if not isinstance(sites[term_index], np.ndarray):
                site = np.array(sites[term_index])
            else:
                site = sites[term_index]
            site.sort()
            if site.min() % 2 == offset:
                terms_indices_to_be_grouped.append(term_index)
        grouped_terms_indices.extend(terms_indices_to_be_grouped)
        grouped_hamiltonian.append(hamiltonian[terms_indices_to_be_grouped])

    for term_index in range(len(coeffs)):
        if term_index not in grouped_terms_indices:
            left_over_terms_indices.append(term_index)

    if len(left_over_terms_indices):
        grouped_hamiltonian.append(hamiltonian[left_over_terms_indices])
    return grouped_hamiltonian


def get_suzuki_permutation(length, order):
    r""" 计算 Suzuki 分解对应的置换数组。

    Args:
        length (int): 对应哈密顿量中的项数，即需要置换的项数
        order (int): Suzuki 分解的阶数

    Returns:
        np.ndarray : 置换数组
    """
    if order == 1:
        return np.arange(length)
    if order == 2:
        return np.vstack([np.arange(length), np.arange(length - 1, -1, -1)])
    else:
        return np.vstack([get_suzuki_permutation(length=length, order=order - 2) for _ in range(5)])


def get_suzuki_p_values(k):
    r""" 计算 Suzuki 分解中递推关系中的因数 p(k)。

    Args:
        k (int): Suzuki 分解的阶数

    Returns:
        list : 一个长度为 5 的列表，其形式为 [p, p, (1 - 4 * p), p, p]
    """
    p = 1 / (4 - 4 ** (1 / (k - 1)))
    return [p, p, (1 - 4 * p), p, p]


def get_suzuki_coefficients(length, order):
    r""" 计算 Suzuki 分解对应的系数数组。

    Args:
        length (int): 对应哈密顿量中的项数，即需要置换的项数
        order (int): Suzuki 分解的阶数

    Returns:
        np.ndarray : 系数数组
    """
    if order == 1:
        return np.ones(length)
    if order == 2:
        return np.vstack([1 / 2 * np.ones(length), 1 / 2 * np.ones(length)])
    else:
        p_values = get_suzuki_p_values(order)
        return np.vstack([get_suzuki_coefficients(length=length, order=order - 2) * p_value
                          for p_value in p_values])


def get_1d_heisenberg_hamiltonian(
        length: int,
        j_x: float = 1.,
        j_y: float = 1.,
        j_z: float = 1.,
        h_z: float or np.ndarray = 0.,
        periodic_boundary_condition: bool = True
):
    r"""生成一个一维海森堡链的哈密顿量。

    Args:
        length (int): 链长
        j_x (float): x 方向的自旋耦合强度 Jx，默认为 ``1``
        j_y (float): y 方向的自旋耦合强度 Jy，默认为 ``1``
        j_z (float): z 方向的自旋耦合强度 Jz，默认为 ``1``
        h_z (float or np.ndarray): z 方向的磁场，默认为 ``0`` ，若输入为单个 float 则认为是均匀磁场（施加在每一个格点上）
        periodic_boundary_condition (bool): 是否考虑周期性边界条件，即 l + 1 = 0，默认为 ``True``

    Returns:
        Hamiltonian :该海森堡链的哈密顿量
    """
    # Pauli words for Heisenberg interactions and their coupling strength
    interactions = ['XX', 'YY', 'ZZ']
    interaction_strength = [j_x, j_y, j_z]
    pauli_str = []  # The Pauli string defining the Hamiltonian

    # add terms (0, 1), (1, 2), ..., (n - 1, n) by adding [j_x, 'X0, X1'], ... into the Pauli string
    for i in range(length - 1):
        for interaction_idx in range(len(interactions)):
            term_str = ''
            interaction = interactions[interaction_idx]
            for idx_word in range(len(interaction)):
                term_str += interaction[idx_word] + str(i + idx_word)
                if idx_word != len(interaction) - 1:
                    term_str += ', '
            pauli_str.append([interaction_strength[interaction_idx], term_str])

    # add interactions on (0, n) for closed periodic boundary condition
    if periodic_boundary_condition:
        boundary_sites = [0, length - 1]
        for interaction_idx in range(len(interactions)):
            term_str = ''
            interaction = interactions[interaction_idx]
            for idx_word in range(len(interaction)):
                term_str += interaction[idx_word] + str(boundary_sites[idx_word])
                if idx_word != len(interaction) - 1:
                    term_str += ', '
            pauli_str.append([interaction_strength[interaction_idx], term_str])

    # add magnetic field, if h_z is a single value, then add a uniform field on each site
    if isinstance(h_z, np.ndarray) or isinstance(h_z, list) or isinstance(h_z, tuple):
        assert len(h_z) == length, 'length of the h_z array do not match the length of the system'
        for i in range(length):
            pauli_str.append([h_z[i], 'Z' + str(i)])
    elif h_z:
        for i in range(length):
            pauli_str.append([h_z, 'Z' + str(i)])

    # instantiate a Hamiltonian object with the Pauli string
    h = Hamiltonian(pauli_str)
    return h
