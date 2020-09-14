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

from functools import reduce

import numpy 
from numpy import absolute, log
from numpy import diag, dot, identity
from numpy import kron as np_kron
from numpy import trace as np_trace
from numpy import matmul as np_matmul
from numpy import random as np_random
from numpy import linalg, sqrt
from numpy import sum as np_sum
from numpy import transpose as np_transpose
from numpy import zeros as np_zeros

from paddle.complex import elementwise_add
from paddle.complex import kron as pp_kron
from paddle.complex import matmul
from paddle.complex import transpose as pp_transpose

from paddle.fluid.dygraph import to_variable
from paddle.fluid.framework import ComplexVariable
from paddle.fluid.layers import concat, cos, ones, reshape, sin
from paddle.fluid.layers import zeros as pp_zeros

from scipy.linalg import logm, sqrtm

__all__ = [
    "partial_trace",
    "state_fidelity",
    "gate_fidelity",
    "purity",
    "von_neumann_entropy",
    "relative_entropy",
    "NKron",
    "dagger",
    "random_pauli_str_generator",
    "pauli_str_to_matrix"
]


def partial_trace(rho_AB, dim1, dim2, A_or_B):
    r"""计算量子态的偏迹。

    Args:
        rho_AB (ComplexVariable): 输入的量子态
        dim1 (int): 系统A的维数
        dim2 (int): 系统B的维数
        A_or_B (int): 1或者2，1表示去除A，2表示去除B

    Returns:
        ComplexVariable: 量子态的偏迹

    """
    if A_or_B == 2:
        dim1, dim2 = dim2, dim1

    idty_np = identity(dim2).astype("complex128")
    idty_B = to_variable(idty_np)

    zero_np = np_zeros([dim2, dim2], "complex128")
    res = to_variable(zero_np)

    for dim_j in range(dim1):
        row_top = pp_zeros([1, dim_j], dtype="float64")
        row_mid = ones([1, 1], dtype="float64")
        row_bot = pp_zeros([1, dim1 - dim_j - 1], dtype="float64")
        bra_j_re = concat([row_top, row_mid, row_bot], axis=1)
        bra_j_im = pp_zeros([1, dim1], dtype="float64")
        bra_j = ComplexVariable(bra_j_re, bra_j_im)

        if A_or_B == 1:
            row_tmp = pp_kron(bra_j, idty_B)
            res = elementwise_add(res, matmul(matmul(row_tmp, rho_AB),
                                              pp_transpose(ComplexVariable(row_tmp.real, -row_tmp.imag),
                                                           perm=[1, 0]), ), )

        if A_or_B == 2:
            row_tmp = pp_kron(idty_B, bra_j)
            res = elementwise_add(res, matmul(matmul(row_tmp, rho_AB),
                                              pp_transpose(ComplexVariable(row_tmp.real, -row_tmp.imag),
                                                           perm=[1, 0]), ), )

    return res


def state_fidelity(rho, sigma):
    r"""计算两个量子态的保真度。

    .. math::
        F(\rho, \sigma) = \text{tr}(\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}})

    Args:
        rho (numpy.ndarray): 量子态的密度矩阵形式
        sigma (numpy.ndarray): 量子态的密度矩阵形式

    Returns:
        float: 输入的量子态之间的保真度
    """
    assert rho.shape == sigma.shape, 'The shape of two quantum states are different'
    fidelity = numpy.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))).real

    return fidelity


def gate_fidelity(U, V):
    r"""计算两个量子门的保真度。

    .. math::

        F(U, V) = |\text{tr}(UV^\dagger)|/2^n

    :math:`U` 是一个 :math:`2^n\times 2^n` 的 Unitary 矩阵。

    Args:
        U (numpy.ndarray): 量子门的酉矩阵形式
        V (numpy.ndarray): 量子门的酉矩阵形式

    Returns:
        float: 输入的量子门之间的保真度
    """
    assert U.shape == V.shape, 'The shape of two unitary matrices are different'
    dim = U.shape[0]
    fidelity = absolute(np_trace(np_matmul(U, V.conj().T)))/dim
    
    return fidelity


def purity(rho):
    r"""计算量子态的纯度。

    .. math::

        P = \text{tr}(\rho^2)

    Args:
        rho (numpy.ndarray): 量子态的密度矩阵形式

    Returns:
        float: 输入的量子态的纯度
    """
    gamma = np_trace(np_matmul(rho, rho))
    
    return gamma.real
    

def von_neumann_entropy(rho):
    r"""计算量子态的冯诺依曼熵。

    .. math::

        S = -\text{tr}(\rho \log(\rho))

    Args:
        rho(numpy.ndarray): 量子态的密度矩阵形式

    Returns:
        float: 输入的量子态的冯诺依曼熵
    """
    rho_eigenvalue, _ = linalg.eig(rho)
    entropy = -np_sum(rho_eigenvalue*log(rho_eigenvalue))
    
    return entropy.real


def relative_entropy(rho, sig):
    r"""计算两个量子态的相对熵。

    .. math::

        S(\rho \| \sigma)=\text{tr} \rho(\log \rho-\log \sigma)

    Args:
        rho (numpy.ndarray): 量子态的密度矩阵形式
        sig (numpy.ndarray): 量子态的密度矩阵形式

    Returns:
        float: 输入的量子态之间的相对熵
    """
    assert rho.shape == sig.shape, 'The shape of two quantum states are different'
    res = numpy.trace(rho @ logm(rho) - rho @ logm(sig))
    return res.real


def NKron(matrix_A, matrix_B, *args):
    r"""计算两个及以上的矩阵的Kronecker积。

    Args:
        matrix_A (numpy.ndarray): 一个矩阵
        matrix_B (numpy.ndarray): 一个矩阵
        *args (numpy.ndarray): 其余矩阵

    Returns:
        ComplexVariable: 输入矩阵的Kronecker积

    .. code-block:: python
    
        from paddle_quantum.state import density_op_random
        from paddle_quantum.utils import NKron
        A = density_op_random(2)
        B = density_op_random(2)
        C = density_op_random(2)
        result = NKron(A, B, C)

    ``result`` 应为： :math:`A \otimes B \otimes C`
    """
    return reduce(lambda result, index: np_kron(result, index), args, np_kron(matrix_A, matrix_B), )


def dagger(matrix):
    r"""计算矩阵的埃尔米特转置，即Hermitian transpose。

    Args:
        matrix (ComplexVariable): 需要埃尔米特转置的矩阵

    Returns:
        ComplexVariable: 输入矩阵的埃尔米特转置

    代码示例:

    .. code-block:: python
    
        from paddle_quantum.utils import dagger
        from paddle import fluid
        import numpy as np
        with fluid.dygraph.guard():
            rho = fluid.dygraph.to_variable(np.array([[1+1j, 2+2j], [3+3j, 4+4j]]))
            print(dagger(rho).numpy())

    ::

        [[1.-1.j 3.-3.j]
        [2.-2.j 4.-4.j]]
    """
    matrix_dagger = pp_transpose(ComplexVariable(matrix.real, -matrix.imag), perm=[1, 0])

    return matrix_dagger


def random_pauli_str_generator(n, terms=3):
    r"""随机生成一个可观测量（observable）的列表（ ``list`` ）形式。

    一个可观测量 :math:`O=0.3X\otimes I\otimes I+0.5Y\otimes I\otimes Z` 的
    列表形式为 ``[[0.3, 'x0'], [0.5, 'y0,z2']]`` 。这样一个可观测量是由
    调用 ``random_pauli_str_generator(3, terms=2)`` 生成的。

    Args:
        n (int): 量子比特数量
        terms (int, optional): 可观测量的项数

    Returns:
        list: 随机生成的可观测量的列表形式
    """
    pauli_str = []
    for sublen in np_random.randint(1, high=n+1, size=terms):
        # Tips: -1 <= coef < 1
        coef = np_random.rand()*2-1
        ops = np_random.choice(['x', 'y', 'z'], size=sublen)
        pos = np_random.choice(range(n), size=sublen, replace=False)
        op_list = [ops[i]+str(pos[i]) for i in range(sublen)]
        pauli_str.append([coef, ','.join(op_list)])
    return pauli_str


def pauli_str_to_matrix(pauli_str, n):
    r"""将输入的可观测量（observable）的列表（ ``list`` ）形式转换为其矩阵形式。

    如输入的 ``pauli_str`` 为 ``[[0.7, 'z0,x1'], [0.2, 'z1']]`` 且 ``n=3`` ,
    则此函数返回可观测量 :math:`0.7Z\otimes X\otimes I+0.2I\otimes Z\otimes I` 的
    矩阵形式。

    Args:
        pauli_str (list): 一个可观测量的列表形式
        n (int): 量子比特数量

    Returns:
        numpy.ndarray: 输入列表对应的可观测量的矩阵形式
    """
    pauli_dict = {'i': numpy.eye(2) + 0j, 'x': numpy.array([[0, 1], [1, 0]]) + 0j,
                  'y': numpy.array([[0, -1j], [1j, 0]]), 'z': numpy.array([[1, 0], [0, -1]]) + 0j}

    # Parse pauli_str; 'x0,z1,y4' to 'xziiy'
    new_pauli_str = []
    for coeff, op_str in pauli_str:
        init = list('i'*n)
        op_list = op_str.split(',')
        for op in op_list:
            pos = int(op[1:])
            assert pos < n, 'n is too small'
            init[pos] = op[0]
        new_pauli_str.append([coeff, ''.join(init)])

    # Convert new_pauli_str to matrix; 'xziiy' to NKron(x, z, i, i, y)
    matrices = []
    for coeff, op_str in new_pauli_str:
        sub_matrices = []
        for op in op_str:
            sub_matrices.append(pauli_dict[op])
        if len(op_str) == 1:
            matrices.append(coeff * sub_matrices[0])
        else:
            matrices.append(coeff * NKron(sub_matrices[0], sub_matrices[1], *sub_matrices[2:]))

    return sum(matrices)
