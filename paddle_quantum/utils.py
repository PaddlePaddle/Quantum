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

from functools import reduce
from math import log2
from math import sqrt
import os.path
import copy
import re
import numpy as np
from scipy.linalg import logm, sqrtm
from scipy.special import logsumexp
from tqdm import tqdm
from matplotlib import colors as mplcolors
import matplotlib.pyplot as plt
import paddle
from paddle import add, to_tensor
from paddle import kron as kron
from paddle import matmul
from paddle import transpose
from paddle import concat, ones
from paddle import zeros
from scipy import sparse
import matplotlib as mpl
from paddle_quantum import simulator
import matplotlib.animation as animation
import matplotlib.image

__all__ = [
    "partial_trace",
    "partial_trace_discontiguous",
    "state_fidelity",
    "trace_distance",
    "gate_fidelity",
    "purity",
    "von_neumann_entropy",
    "relative_entropy",
    "NKron",
    "dagger",
    "random_pauli_str_generator",
    "pauli_str_to_matrix",
    "partial_transpose_2",
    "partial_transpose",
    "negativity",
    "logarithmic_negativity",
    "is_ppt",
    "haar_orthogonal",
    "haar_unitary",
    "haar_state_vector",
    "haar_density_operator",
    "schmidt_decompose",
    "plot_state_in_bloch_sphere",
    "plot_multi_qubits_state_in_bloch_sphere",
    "plot_rotation_in_bloch_sphere",
    "plot_density_matrix_graph",
    "image_to_density_matrix",
    "Hamiltonian",
    "QuantumFisher",
    "ClassicalFisher",
]


def partial_trace(rho_AB, dim1, dim2, A_or_B):
    r"""计算量子态的偏迹。

    Args:
        rho_AB (Tensor): 输入的量子态
        dim1 (int): 系统A的维数
        dim2 (int): 系统B的维数
        A_or_B (int): 1或者2，1表示计算系统A上的偏迹，2表示计算系统B上的偏迹

    Returns:
        Tensor: 输入的量子态的偏迹

    """
    if A_or_B == 2:
        dim1, dim2 = dim2, dim1

    idty_np = np.identity(dim2).astype("complex128")
    idty_B = to_tensor(idty_np)

    zero_np = np.zeros([dim2, dim2], "complex128")
    res = to_tensor(zero_np)

    for dim_j in range(dim1):
        row_top = zeros([1, dim_j], dtype="float64")
        row_mid = ones([1, 1], dtype="float64")
        row_bot = zeros([1, dim1 - dim_j - 1], dtype="float64")
        bra_j = concat([row_top, row_mid, row_bot], axis=1)
        bra_j = paddle.cast(bra_j, 'complex128')

        if A_or_B == 1:
            row_tmp = kron(bra_j, idty_B)
            row_tmp_conj = paddle.conj(row_tmp)
            res = add(res, matmul(matmul(row_tmp, rho_AB), transpose(row_tmp_conj, perm=[1, 0]), ), )

        if A_or_B == 2:
            row_tmp = kron(idty_B, bra_j)
            row_tmp_conj = paddle.conj(row_tmp)
            res = add(res, matmul(matmul(row_tmp, rho_AB), transpose(row_tmp_conj, perm=[1, 0]), ), )

    return res


def partial_trace_discontiguous(rho, preserve_qubits=None):
    r"""计算量子态的偏迹，可选取任意子系统。

    Args:
        rho (Tensor): 输入的量子态
        preserve_qubits (list): 要保留的量子比特，默认为 None，表示全保留
    """
    if preserve_qubits is None:
        return rho
    else:
        n = int(log2(rho.size) // 2)
        num_preserve = len(preserve_qubits)

        shape = paddle.ones((n + 1,))
        shape = 2 * shape
        shape[n] = 2 ** n
        shape = paddle.cast(shape, "int32")
        identity = paddle.eye(2 ** n)
        identity = paddle.reshape(identity, shape=shape)
        discard = list()
        for idx in range(0, n):
            if idx not in preserve_qubits:
                discard.append(idx)
        addition = [n]
        preserve_qubits.sort()

        preserve_qubits = paddle.to_tensor(preserve_qubits)
        discard = paddle.to_tensor(discard)
        addition = paddle.to_tensor(addition)
        permute = paddle.concat([discard, preserve_qubits, addition])

        identity = paddle.transpose(identity, perm=permute)
        identity = paddle.reshape(identity, (2 ** n, 2 ** n))

        result = np.zeros((2 ** num_preserve, 2 ** num_preserve), dtype="complex64")
        result = paddle.to_tensor(result)

        for i in range(0, 2 ** (n - num_preserve)):
            bra = identity[i * 2 ** num_preserve:(i + 1) * 2 ** num_preserve, :]
            result = result + matmul(matmul(bra, rho), transpose(bra, perm=[1, 0]))

        return result


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
    fidelity = np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))).real

    return fidelity


def trace_distance(rho, sigma):
    r"""计算两个量子态的迹距离。

    .. math::
        D(\rho, \sigma) = 1 / 2 * \text{tr}|\rho-\sigma|

    Args:
        rho (numpy.ndarray): 量子态的密度矩阵形式
        sigma (numpy.ndarray): 量子态的密度矩阵形式

    Returns:
        float: 输入的量子态之间的迹距离
    """
    assert rho.shape == sigma.shape, 'The shape of two quantum states are different'
    A = rho - sigma
    distance = 1 / 2 * np.sum(np.abs(np.linalg.eigvals(A)))

    return distance


def gate_fidelity(U, V):
    r"""计算两个量子门的保真度。

    .. math::

        F(U, V) = |\text{tr}(UV^\dagger)|/2^n

    :math:`U` 是一个 :math:`2^n\times 2^n` 的 Unitary 矩阵。

    Args:
        U (numpy.ndarray): 量子门 :math:`U` 的酉矩阵形式
        V (numpy.ndarray): 量子门 :math:`V` 的酉矩阵形式

    Returns:
        float: 输入的量子门之间的保真度
    """
    assert U.shape == V.shape, 'The shape of two unitary matrices are different'
    dim = U.shape[0]
    fidelity = np.absolute(np.trace(np.matmul(U, V.conj().T))) / dim

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
    gamma = np.trace(np.matmul(rho, rho))

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
    rho_eigenvalues = np.real(np.linalg.eigvals(rho))
    entropy = 0
    for eigenvalue in rho_eigenvalues:
        if np.abs(eigenvalue) < 1e-8:
            continue
        entropy -= eigenvalue * np.log(eigenvalue)

    return entropy


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
    res = np.trace(rho @ logm(rho) - rho @ logm(sig))
    return res.real


def NKron(matrix_A, matrix_B, *args):
    r"""计算两个及以上的矩阵的Kronecker积。

    Args:
        matrix_A (numpy.ndarray): 一个矩阵
        matrix_B (numpy.ndarray): 一个矩阵
        *args (numpy.ndarray): 其余矩阵

    Returns:
        Tensor: 输入矩阵的Kronecker积

    .. code-block:: python

        from paddle_quantum.state import density_op_random
        from paddle_quantum.utils import NKron
        A = density_op_random(2)
        B = density_op_random(2)
        C = density_op_random(2)
        result = NKron(A, B, C)

    ``result`` 应为 :math:`A \otimes B \otimes C`
    """
    return reduce(lambda result, index: np.kron(result, index), args, np.kron(matrix_A, matrix_B), )


def dagger(matrix):
    r"""计算矩阵的埃尔米特转置，即Hermitian transpose。

    Args:
        matrix (Tensor): 需要埃尔米特转置的矩阵

    Returns:
        Tensor: 输入矩阵的埃尔米特转置

    代码示例:

    .. code-block:: python

        from paddle_quantum.utils import dagger
        import numpy as np
        rho = paddle.to_tensor(np.array([[1+1j, 2+2j], [3+3j, 4+4j]]))
        print(dagger(rho).numpy())

    ::

        [[1.-1.j 3.-3.j]
        [2.-2.j 4.-4.j]]
    """
    matrix_conj = paddle.conj(matrix)
    matrix_dagger = transpose(matrix_conj, perm=[1, 0])

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
    for sublen in np.random.randint(1, high=n + 1, size=terms):
        # Tips: -1 <= coeff < 1
        coeff = np.random.rand() * 2 - 1
        ops = np.random.choice(['x', 'y', 'z'], size=sublen)
        pos = np.random.choice(range(n), size=sublen, replace=False)
        op_list = [ops[i] + str(pos[i]) for i in range(sublen)]
        pauli_str.append([coeff, ','.join(op_list)])
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
    pauli_dict = {'i': np.eye(2) + 0j, 'x': np.array([[0, 1], [1, 0]]) + 0j,
                  'y': np.array([[0, -1j], [1j, 0]]), 'z': np.array([[1, 0], [0, -1]]) + 0j}

    # Parse pauli_str; 'x0,z1,y4' to 'xziiy'
    new_pauli_str = []
    for coeff, op_str in pauli_str:
        init = list('i' * n)
        op_list = re.split(r',\s*', op_str.lower())
        for op in op_list:
            if len(op) > 1:
                pos = int(op[1:])
                assert pos < n, 'n is too small'
                init[pos] = op[0]
            elif op.lower() != 'i':
                raise ValueError('Only Pauli operator "I" can be accepted without specifying its position')
        new_pauli_str.append([coeff, ''.join(init)])

    # Convert new_pauli_str to matrix; 'xziiy' to NKron(x, z, i, i, y)
    matrices = []
    for coeff, op_str in new_pauli_str:
        sub_matrices = []
        for op in op_str:
            sub_matrices.append(pauli_dict[op.lower()])
        if len(op_str) == 1:
            matrices.append(coeff * sub_matrices[0])
        else:
            matrices.append(coeff * NKron(sub_matrices[0], sub_matrices[1], *sub_matrices[2:]))

    return sum(matrices)


def partial_transpose_2(density_op, sub_system=None):
    r"""计算输入量子态的 partial transpose :math:`\rho^{T_A}`

    Args:
        density_op (numpy.ndarray): 量子态的密度矩阵形式
        sub_system (int): 1或2，表示关于哪个子系统进行 partial transpose，默认为第二个

    Returns:
        float: 输入的量子态的 partial transpose

    代码示例:

    .. code-block:: python

        from paddle_quantum.utils import partial_transpose_2
        rho_test = np.arange(1,17).reshape(4,4)
        partial_transpose_2(rho_test, sub_system=1)

    ::

       [[ 1,  2,  9, 10],
        [ 5,  6, 13, 14],
        [ 3,  4, 11, 12],
        [ 7,  8, 15, 16]]
    """
    sys_idx = 2 if sub_system is None else 1

    # Copy the density matrix and not corrupt the original one
    transposed_density_op = np.copy(density_op)
    if sys_idx == 2:
        for j in [0, 2]:
            for i in [0, 2]:
                transposed_density_op[i:i + 2, j:j + 2] = density_op[i:i + 2, j:j + 2].transpose()
    else:
        transposed_density_op[2:4, 0:2] = density_op[0:2, 2:4]
        transposed_density_op[0:2, 2:4] = density_op[2:4, 0:2]

    return transposed_density_op


def partial_transpose(density_op, n):
    r"""计算输入量子态的 partial transpose :math:`\rho^{T_A}`。

    Args:
        density_op (numpy.ndarray): 量子态的密度矩阵形式

    Returns:
        float: 输入的量子态的 partial transpose
    """

    # Copy the density matrix and not corrupt the original one
    transposed_density_op = np.copy(density_op)
    for j in range(0, 2 ** n, 2):
        for i in range(0, 2 ** n, 2):
            transposed_density_op[i:i + 2, j:j + 2] = density_op[i:i + 2, j:j + 2].transpose()

    return transposed_density_op


def negativity(density_op):
    r"""计算输入量子态的 Negativity :math:`N = ||\frac{\rho^{T_A}-1}{2}||`。

    Args:
        density_op (numpy.ndarray): 量子态的密度矩阵形式

    Returns:
        float: 输入的量子态的 Negativity

    代码示例:

    .. code-block:: python

        from paddle_quantum.utils import negativity
        from paddle_quantum.state import bell_state
        rho = bell_state(2)
        print("Negativity of the Bell state is:", negativity(rho))

    ::

        Negativity of the Bell state is: 0.5
    """
    # Implement the partial transpose
    density_op_T = partial_transpose_2(density_op)

    # Calculate through the equivalent expression N = sum(abs(\lambda_i)) when \lambda_i<0
    n = 0
    eigen_val, _ = np.linalg.eig(density_op_T)
    for val in eigen_val:
        if val < 0:
            n = n + np.abs(val)
    return n


def logarithmic_negativity(density_op):
    r"""计算输入量子态的 Logarithmic Negativity :math:`E_N = ||\rho^{T_A}||`。

    Args:
        density_op (numpy.ndarray): 量子态的密度矩阵形式

    Returns:
        float: 输入的量子态的 Logarithmic Negativity

    代码示例:

    .. code-block:: python

        from paddle_quantum.utils import logarithmic_negativity
        from paddle_quantum.state import bell_state
        rho = bell_state(2)
        print("Logarithmic negativity of the Bell state is:", logarithmic_negativity(rho))

    ::

        Logarithmic negativity of the Bell state is: 1.0
    """
    # Calculate the negativity
    n = negativity(density_op)

    # Calculate through the equivalent expression
    log2_n = np.log2(2 * n + 1)
    return log2_n


def is_ppt(density_op):
    r"""计算输入量子态是否满足 PPT 条件。

    Args:
        density_op (numpy.ndarray): 量子态的密度矩阵形式

    Returns:
        bool: 输入的量子态是否满足 PPT 条件

    代码示例:

    .. code-block:: python

        from paddle_quantum.utils import is_ppt
        from paddle_quantum.state import bell_state
        rho = bell_state(2)
        print("Whether the Bell state satisfies PPT condition:", is_ppt(rho))

    ::

        Whether the Bell state satisfies PPT condition: False
    """
    # By default the PPT condition is satisfied
    ppt = True

    # Detect negative eigenvalues from the partial transposed density_op
    if negativity(density_op) > 0:
        ppt = False
    return ppt


def haar_orthogonal(n):
    r"""生成一个服从 Haar random 的正交矩阵。采样算法参考文献：arXiv:math-ph/0609050v2

        Args:
            n (int): 正交矩阵对应的量子比特数

        Returns:
            numpy.ndarray: 一个形状为 ``(2**n, 2**n)`` 随机正交矩阵
    """
    # Matrix dimension
    d = 2 ** n
    # Step 1: sample from Ginibre ensemble
    g = (np.random.randn(d, d))
    # Step 2: perform QR decomposition of G
    q, r = np.linalg.qr(g)
    # Step 3: make the decomposition unique
    lam = np.diag(r) / abs(np.diag(r))
    u = q @ np.diag(lam)

    return u


def haar_unitary(n):
    r"""生成一个服从 Haar random 的酉矩阵。采样算法参考文献：arXiv:math-ph/0609050v2

        Args:
            n (int): 酉矩阵对应的量子比特数

        Returns:
            numpy.ndarray: 一个形状为 ``(2**n, 2**n)`` 随机酉矩阵
    """
    # Matrix dimension
    d = 2 ** n
    # Step 1: sample from Ginibre ensemble
    g = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) / np.sqrt(2)
    # Step 2: perform QR decomposition of G
    q, r = np.linalg.qr(g)
    # Step 3: make the decomposition unique
    lam = np.diag(r) / abs(np.diag(r))
    u = q @ np.diag(lam)

    return u


def haar_state_vector(n, real=False):
    r"""生成一个服从 Haar random 的态矢量。

        Args:
            n (int): 量子态的量子比特数
            real (bool): 生成的态矢量是否为实态矢量，默认为 ``False``

        Returns:
            numpy.ndarray: 一个形状为 ``(2**n, 1)`` 随机态矢量
    """
    # Vector dimension
    d = 2 ** n
    if real:
        # Generate a Haar random orthogonal matrix
        o = haar_orthogonal(n)
        # Perform u onto |0>, i.e., the first column of o
        phi = o[:, 0]
    else:
        # Generate a Haar random unitary
        u = haar_unitary(n)
        # Perform u onto |0>, i.e., the first column of u
        phi = u[:, 0]

    return phi


def haar_density_operator(n, k=None, real=False):
    r"""生成一个服从 Haar random 的密度矩阵。

        Args:
            n (int): 量子态的量子比特数
            k (int): 密度矩阵的秩，默认为 ``None``，表示满秩
            real (bool): 生成的密度矩阵是否为实矩阵，默认为 ``False``

        Returns:
            numpy.ndarray: 一个形状为 ``(2**n, 2**n)`` 随机密度矩阵
    """
    d = 2 ** n
    k = k if k is not None else d
    assert 0 < k <= d, 'rank is an invalid number'
    if real:
        ginibre_matrix = np.random.randn(d, k)
        rho = ginibre_matrix @ ginibre_matrix.T
    else:
        ginibre_matrix = np.random.randn(d, k) + 1j * np.random.randn(d, k)
        rho = ginibre_matrix @ ginibre_matrix.conj().T

    return rho / np.trace(rho)


def schmidt_decompose(psi, sys_A=None):
    r"""计算输入量子态的施密特分解 :math:`\lvert\psi\rangle=\sum_ic_i\lvert i_A\rangle\otimes\lvert i_B \rangle`。

    Args:
        psi (numpy.ndarray): 量子态的向量形式，形状为（2**n）
        sys_A (list): 包含在子系统 A 中的 qubit 下标（其余 qubit 包含在子系统B中），默认为量子态 :math:`\lvert \psi\rangle` 的前半数 qubit

    Returns:
        tuple: 包含如下元素
            - numpy.ndarray: 由施密特系数组成的一维数组，形状为 ``(k)``
            - numpy.ndarray: 由子系统A的基 :math:`\lvert i_A\rangle` 组成的高维数组，形状为 ``(k, 2**m, 1)``
            - numpy.ndarray: 由子系统B的基 :math:`\lvert i_B\rangle` 组成的高维数组，形状为 ``(k, 2**l, 1)``

    Warning:
        小于 ``1e-13`` 的施密特系数将被视作浮点误差并舍去

    .. code-block:: python

        # zz = 1/√2 * (<010| + <101|)
        zz = 1/np.sqrt(2) * (np.array([0., 0., 1., 0., 0., 0., 0., 0.]) + np.array([0., 0., 0., 0., 0., 1., 0., 0.]))
        print('input state:', zz)
        l, u, v = schmidt_decompose(zz, [0, 2])

        print('output state:')
        for i in range(len(l)):
            print('+', l[i], '*', u[i].reshape([-1]), '⨂', v[i].reshape([-1]))

    Note:
        代码示例结果应为 1/√2 * (<00|⨂<1| + <11|⨂<0|)。注意，输入态的第 0、2 个 qubit 处于输出态的子系统 A，输入态的第1个 qubit 处于输出态的子系统 B。
    """

    assert psi.ndim == 1, 'Psi must be a one dimensional vector.'
    assert np.log2(psi.size).is_integer(), 'The number of amplitutes must be an integral power of 2.'

    tot_qu = int(np.log2(psi.size))
    sys_A = sys_A if sys_A is not None else [i for i in range(tot_qu//2)]
    sys_B = [i for i in range(tot_qu) if i not in sys_A]

    # Permute qubit indices
    psi = psi.reshape([2] * tot_qu).transpose(sys_A + sys_B)

    # construct amplitute matrix
    amp_mtr = psi.reshape([2**len(sys_A), 2**len(sys_B)])

    # Standard process to obtain schmidt decomposition
    u, c, v = np.linalg.svd(amp_mtr)

    k = np.count_nonzero(c > 1e-13)
    c = c[:k]
    u = u.T[:k].reshape([k, -1, 1])
    v = v[:k].reshape([k, -1, 1])
    return c, u, v


def __density_matrix_convert_to_bloch_vector(density_matrix):
    r"""该函数将密度矩阵转化为bloch球面上的坐标

    Args:
        density_matrix (numpy.ndarray): 输入的密度矩阵

    Returns:
        bloch_vector (numpy.ndarray): 存储bloch向量的 x,y,z 坐标，向量的模长，向量的颜色
    """

    # Pauli Matrix
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])

    # Convert a density matrix to a Bloch vector.
    ax = np.trace(np.dot(density_matrix, pauli_x)).real
    ay = np.trace(np.dot(density_matrix, pauli_y)).real
    az = np.trace(np.dot(density_matrix, pauli_z)).real

    # Calc the length of bloch vector
    length = ax ** 2 + ay ** 2 + az ** 2
    length = sqrt(length)
    if length > 1.0:
        length = 1.0

    # Calc the color of bloch vector, the value of the color is proportional to the length
    color = length

    bloch_vector = [ax, ay, az, length, color]

    # You must use an array, which is followed by slicing and taking a column
    bloch_vector = np.array(bloch_vector)

    return bloch_vector


def __plot_bloch_sphere(
        ax,
        bloch_vectors=None,
        show_arrow=False,
        clear_plt=True,
        rotating_angle_list=None,
        view_angle=None,
        view_dist=None,
        set_color=None
):
    r"""将 Bloch 向量展示在 Bloch 球面上

    Args:
        ax (Axes3D(fig)): 画布的句柄
        bloch_vectors (numpy.ndarray): 存储bloch向量的 x,y,z 坐标，向量的模长，向量的颜色
        show_arrow (bool): 是否展示向量的箭头，默认为 False
        clear_plt (bool): 是否要清空画布，默认为 True，每次画图的时候清空画布再画图
        rotating_angle_list (list): 旋转角度的列表，用于展示旋转轨迹
        view_angle (list): 视图的角度，
            第一个元素为关于xy平面的夹角[0-360],第二个元素为关于xz平面的夹角[0-360], 默认为 (30, 45)
        view_dist (int): 视图的距离，默认为 7
        set_color (str): 设置指定的颜色，请查阅cmap表，默认为 "红-黑-根据向量的模长渐变" 颜色方案
    """
    # Assign a value to an empty variable
    if view_angle is None:
        view_angle = (30, 45)
    if view_dist is None:
        view_dist = 7
    # Define my_color
    if set_color is None:
        color = 'rainbow'
        black_code = '#000000'
        red_code = '#F24A29'
        if bloch_vectors is not None:
            black_to_red = mplcolors.LinearSegmentedColormap.from_list(
                'my_color',
                [(0, black_code), (1, red_code)],
                N=len(bloch_vectors[:, 4])
            )
            map_vir = plt.get_cmap(black_to_red)
            color = map_vir(bloch_vectors[:, 4])
    else:
        color = set_color

    # Set the view angle and view distance
    ax.view_init(view_angle[0], view_angle[1])
    ax.dist = view_dist

    # Draw the general frame
    def draw_general_frame():

        # Do not show the grid and original axes
        ax.grid(False)
        ax.set_axis_off()
        ax.view_init(view_angle[0], view_angle[1])
        ax.dist = view_dist

        # Set the lower limit and upper limit of each axis
        # To make the bloch_ball look less flat, the default is relatively flat
        ax.set_xlim3d(xmin=-1.5, xmax=1.5)
        ax.set_ylim3d(ymin=-1.5, ymax=1.5)
        ax.set_zlim3d(zmin=-1, zmax=1.3)

        # Draw a new axes
        coordinate_start_x, coordinate_start_y, coordinate_start_z = \
            np.array([[-1.5, 0, 0], [0, -1.5, 0], [0, 0, -1.5]])
        coordinate_end_x, coordinate_end_y, coordinate_end_z = \
            np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
        ax.quiver(
            coordinate_start_x, coordinate_start_y, coordinate_start_z,
            coordinate_end_x, coordinate_end_y, coordinate_end_z,
            arrow_length_ratio=0.03, color="black", linewidth=0.5
        )
        ax.text(0, 0, 1.7, r"|0⟩", color="black", fontsize=16)
        ax.text(0, 0, -1.9, r"|1⟩", color="black", fontsize=16)
        ax.text(1.9, 0, 0, r"|+⟩", color="black", fontsize=16)
        ax.text(-1.7, 0, 0, r"|–⟩", color="black", fontsize=16)
        ax.text(0, 1.7, 0, r"|i+⟩", color="black", fontsize=16)
        ax.text(0, -1.9, 0, r"|i–⟩", color="black", fontsize=16)

        # Draw a surface
        horizontal_angle = np.linspace(0, 2 * np.pi, 80)
        vertical_angle = np.linspace(0, np.pi, 80)
        surface_point_x = np.outer(np.cos(horizontal_angle), np.sin(vertical_angle))
        surface_point_y = np.outer(np.sin(horizontal_angle), np.sin(vertical_angle))
        surface_point_z = np.outer(np.ones(np.size(horizontal_angle)), np.cos(vertical_angle))
        ax.plot_surface(
            surface_point_x, surface_point_y, surface_point_z, rstride=1, cstride=1,
            color="black", linewidth=0.05, alpha=0.03
        )

        # Draw circle
        def draw_circle(circle_horizon_angle, circle_vertical_angle, linewidth=0.5, alpha=0.2):
            r = 1
            circle_point_x = r * np.cos(circle_vertical_angle) * np.cos(circle_horizon_angle)
            circle_point_y = r * np.cos(circle_vertical_angle) * np.sin(circle_horizon_angle)
            circle_point_z = r * np.sin(circle_vertical_angle)
            ax.plot(
                circle_point_x, circle_point_y, circle_point_z,
                color="black", linewidth=linewidth, alpha=alpha
            )

        # draw longitude and latitude
        def draw_longitude_and_latitude():
            # Draw longitude
            num = 3
            theta = np.linspace(0, 0, 100)
            psi = np.linspace(0, 2 * np.pi, 100)
            for i in range(num):
                theta = theta + np.pi / num
                draw_circle(theta, psi)

            # Draw latitude
            num = 6
            theta = np.linspace(0, 2 * np.pi, 100)
            psi = np.linspace(-np.pi / 2, -np.pi / 2, 100)
            for i in range(num):
                psi = psi + np.pi / num
                draw_circle(theta, psi)

            # Draw equator
            theta = np.linspace(0, 2 * np.pi, 100)
            psi = np.linspace(0, 0, 100)
            draw_circle(theta, psi, linewidth=0.5, alpha=0.2)

            # Draw prime meridian
            theta = np.linspace(0, 0, 100)
            psi = np.linspace(0, 2 * np.pi, 100)
            draw_circle(theta, psi, linewidth=0.5, alpha=0.2)

        # If the number of data points exceeds 20, no longitude and latitude lines will be drawn.
        if bloch_vectors is not None and len(bloch_vectors) < 52:
            draw_longitude_and_latitude()
        elif bloch_vectors is None:
            draw_longitude_and_latitude()

        # Draw three invisible points
        invisible_points = np.array([[0.03440399, 0.30279721, 0.95243384],
                                     [0.70776026, 0.57712403, 0.40743499],
                                     [0.46991358, -0.63717908, 0.61088792]])
        ax.scatter(
            invisible_points[:, 0], invisible_points[:, 1], invisible_points[:, 2],
            c='w', alpha=0.01
        )

    # clean plt
    if clear_plt:
        ax.cla()
        draw_general_frame()

    # Draw the data points
    if bloch_vectors is not None:
        ax.scatter(
            bloch_vectors[:, 0], bloch_vectors[:, 1], bloch_vectors[:, 2], c=color, alpha=1.0
        )

    # if show the rotating angle
    if rotating_angle_list is not None:
        bloch_num = len(bloch_vectors)
        rotating_angle_theta, rotating_angle_phi, rotating_angle_lam = rotating_angle_list[bloch_num - 1]
        rotating_angle_theta = round(rotating_angle_theta, 6)
        rotating_angle_phi = round(rotating_angle_phi, 6)
        rotating_angle_lam = round(rotating_angle_lam, 6)

        # Shown at the top right of the perspective
        display_text_angle = [-(view_angle[0] - 10), (view_angle[1] + 10)]
        text_point_x = 2 * np.cos(display_text_angle[0]) * np.cos(display_text_angle[1])
        text_point_y = 2 * np.cos(display_text_angle[0]) * np.sin(-display_text_angle[1])
        text_point_z = 2 * np.sin(-display_text_angle[0])
        ax.text(text_point_x, text_point_y, text_point_z, r'$\theta=' + str(rotating_angle_theta) + r'$',
                color="black", fontsize=14)
        ax.text(text_point_x, text_point_y, text_point_z - 0.1, r'$\phi=' + str(rotating_angle_phi) + r'$',
                color="black", fontsize=14)
        ax.text(text_point_x, text_point_y, text_point_z - 0.2, r'$\lambda=' + str(rotating_angle_lam) + r'$',
                color="black", fontsize=14)

    # If show the bloch_vector
    if show_arrow:
        ax.quiver(
            0, 0, 0, bloch_vectors[:, 0], bloch_vectors[:, 1], bloch_vectors[:, 2],
            arrow_length_ratio=0.05, color=color, alpha=1.0
        )


def plot_state_in_bloch_sphere(
        state,
        show_arrow=False,
        save_gif=False,
        filename=None,
        view_angle=None,
        view_dist=None,
        set_color=None
):
    r"""将输入的量子态展示在 Bloch 球面上

    Args:
        state (list(numpy.ndarray or paddle.Tensor)): 输入的量子态列表，可以支持态矢量和密度矩阵
        show_arrow (bool): 是否展示向量的箭头，默认为 ``False``
        save_gif (bool): 是否存储 gif 动图，默认为 ``False``
        filename (str): 存储的 gif 动图的名字
        view_angle (list or tuple): 视图的角度，
            第一个元素为关于 xy 平面的夹角 [0-360]，第二个元素为关于 xz 平面的夹角 [0-360], 默认为 ``(30, 45)``
        view_dist (int): 视图的距离，默认为 7
        set_color (str): 若要设置指定的颜色，请查阅 ``cmap`` 表。默认为红色到黑色的渐变颜色
    """
    # Check input data
    __input_args_dtype_check(show_arrow, save_gif, filename, view_angle, view_dist)

    assert type(state) == list or type(state) == paddle.Tensor or type(state) == np.ndarray, \
        'the type of "state" must be "list" or "paddle.Tensor" or "np.ndarray".'
    if type(state) == paddle.Tensor or type(state) == np.ndarray:
        state = [state]
    state_len = len(state)
    assert state_len >= 1, '"state" is NULL.'
    for i in range(state_len):
        assert type(state[i]) == paddle.Tensor or type(state[i]) == np.ndarray, \
            'the type of "state[i]" should be "paddle.Tensor" or "numpy.ndarray".'
    if set_color is not None:
        assert type(set_color) == str, \
            'the type of "set_color" should be "str".'

    # Assign a value to an empty variable
    if filename is None:
        filename = 'state_in_bloch_sphere.gif'
    if view_angle is None:
        view_angle = (30, 45)
    if view_dist is None:
        view_dist = 7

    # Convert Tensor to numpy
    for i in range(state_len):
        if type(state[i]) == paddle.Tensor:
            state[i] = state[i].numpy()

    # Convert state_vector to density_matrix
    for i in range(state_len):
        if state[i].size == 2:
            state_vector = state[i]
            state[i] = np.outer(state_vector, np.conj(state_vector))

    # Calc the bloch_vectors
    bloch_vector_list = []
    for i in range(state_len):
        bloch_vector_tmp = __density_matrix_convert_to_bloch_vector(state[i])
        bloch_vector_list.append(bloch_vector_tmp)

    # List must be converted to array for slicing.
    bloch_vectors = np.array(bloch_vector_list)

    # A update function for animation class
    def update(frame):
        view_rotating_angle = 5
        new_view_angle = [view_angle[0], view_angle[1] + view_rotating_angle * frame]
        __plot_bloch_sphere(
            ax, bloch_vectors, show_arrow, clear_plt=True,
            view_angle=new_view_angle, view_dist=view_dist, set_color=set_color
        )

    # Dynamic update and save
    if save_gif:
        # Helper function to plot vectors on a sphere.
        fig = plt.figure(figsize=(8, 8), dpi=100)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = fig.add_subplot(111, projection='3d')

        frames_num = 7
        anim = animation.FuncAnimation(fig, update, frames=frames_num, interval=600, repeat=False)
        anim.save(filename, dpi=100, writer='pillow')
        # close the plt
        plt.close(fig)

    # Helper function to plot vectors on a sphere.
    fig = plt.figure(figsize=(8, 8), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, projection='3d')

    __plot_bloch_sphere(
        ax, bloch_vectors, show_arrow, clear_plt=True,
        view_angle=view_angle, view_dist=view_dist, set_color=set_color
    )

    plt.show()


def plot_multi_qubits_state_in_bloch_sphere(
        state,
        which_qubits=None,
        show_arrow=False,
        save_gif=False,
        save_pic=True,
        filename=None,
        view_angle=None,
        view_dist=None,
        set_color='#0000FF'
):
    r"""将输入的多量子比特的量子态展示在 Bloch 球面上

    Args:
        state (numpy.ndarray or paddle.Tensor): 输入的量子态，可以支持态矢量和密度矩阵
        which_qubits (list or None): 要展示的量子比特，默认为全展示
        show_arrow (bool): 是否展示向量的箭头，默认为 ``False``
        save_gif (bool): 是否存储 gif 动图，默认为 ``False``
        save_pic (bool): 是否存储静态图片，默认为 ``True``
        filename (str): 存储的图片的名字
        view_angle (list or tuple): 视图的角度，第一个元素为关于 xy 平面的夹角 [0-360]，第二个元素为关于 xz 平面的夹角 [0-360], 默认为 ``(30, 45)``
        view_dist (int): 视图的距离，默认为 7
        set_color (str): 若要设置指定的颜色，请查阅 ``cmap`` 表。默认为蓝色
    """
    # Check input data
    __input_args_dtype_check(show_arrow, save_gif, filename, view_angle, view_dist)

    assert type(state) == paddle.Tensor or type(state) == np.ndarray, \
        'the type of "state" must be "paddle.Tensor" or "np.ndarray".'
    assert type(set_color) == str, \
        'the type of "set_color" should be "str".'

    n_qubits = int(np.log2(state.shape[0]))

    if which_qubits is None:
        which_qubits = list(range(n_qubits))
    else:
        assert type(which_qubits) == list, 'the type of which_qubits should be None or list'
        assert 1 <= len(which_qubits) <= n_qubits, '展示的量子数量需要小于n_qubits'
        for i in range(len(which_qubits)):
            assert 0 <= which_qubits[i] < n_qubits, '0<which_qubits[i]<n_qubits'

    # Assign a value to an empty variable
    if filename is None:
        filename = 'state_in_bloch_sphere.gif'
    if view_angle is None:
        view_angle = (30, 45)
    if view_dist is None:
        view_dist = 7

    # Convert Tensor to numpy
    if type(state) == paddle.Tensor:
        state = state.numpy()

    # state_vector to density matrix
    if state.shape[0] >= 2 and state.size == state.shape[0]:
        state_vector = state
        state = np.outer(state_vector, np.conj(state_vector))

    # multi qubits state decompose
    if state.shape[0] > 2:
        rho = paddle.to_tensor(state)
        tmp_s = []
        for q in which_qubits:
            tmp_s.append(partial_trace_discontiguous(rho, [q]))
        state = tmp_s
    else:
        state = [state]
    state_len = len(state)

    # Calc the bloch_vectors
    bloch_vector_list = []
    for i in range(state_len):
        bloch_vector_tmp = __density_matrix_convert_to_bloch_vector(state[i])
        bloch_vector_list.append(bloch_vector_tmp)

    # List must be converted to array for slicing.
    bloch_vectors = np.array(bloch_vector_list)

    # A update function for animation class
    def update(frame):
        view_rotating_angle = 5
        new_view_angle = [view_angle[0], view_angle[1] + view_rotating_angle * frame]
        __plot_bloch_sphere(
            ax, bloch_vectors, show_arrow, clear_plt=True,
            view_angle=new_view_angle, view_dist=view_dist, set_color=set_color
        )

    # Dynamic update and save
    if save_gif:
        # Helper function to plot vectors on a sphere.
        fig = plt.figure(figsize=(8, 8), dpi=100)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = fig.add_subplot(111, projection='3d')

        frames_num = 7
        anim = animation.FuncAnimation(fig, update, frames=frames_num, interval=600, repeat=False)
        anim.save(filename, dpi=100, writer='pillow')
        # close the plt
        plt.close(fig)

    # Helper function to plot vectors on a sphere.
    fig = plt.figure(figsize=(8, 8), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    dim = np.ceil(sqrt(len(which_qubits)))
    for i in range(1, len(which_qubits)+1):
        ax = fig.add_subplot(dim, dim, i, projection='3d')
        bloch_vector = np.array([bloch_vectors[i-1]])
        __plot_bloch_sphere(
            ax, bloch_vector, show_arrow, clear_plt=True,
            view_angle=view_angle, view_dist=view_dist, set_color=set_color
        )
    if save_pic:
        plt.savefig('n_qubit_state_in_bloch.png', bbox_inches='tight')
    plt.show()


def plot_rotation_in_bloch_sphere(
        init_state,
        rotating_angle,
        show_arrow=False,
        save_gif=False,
        filename=None,
        view_angle=None,
        view_dist=None,
        color_scheme=None,
):
    r"""在 Bloch 球面上刻画从初始量子态开始的旋转轨迹

    Args:
        init_state (numpy.ndarray or paddle.Tensor): 输入的初始量子态，可以支持态矢量和密度矩阵
        rotating_angle (list(float)): 旋转角度 ``[theta, phi, lam]``
        show_arrow (bool): 是否展示向量的箭头，默认为 ``False``
        save_gif (bool): 是否存储 gif 动图，默认为 ``False``
        filename (str): 存储的 gif 动图的名字
        view_angle (list or tuple): 视图的角度，
            第一个元素为关于 xy 平面的夹角 [0-360]，第二个元素为关于 xz 平面的夹角 [0-360], 默认为 ``(30, 45)``
        view_dist (int): 视图的距离，默认为 7
        color_scheme (list(str,str,str)): 分别是初始颜色，轨迹颜色，结束颜色。若要设置指定的颜色，请查阅 ``cmap`` 表。默认为红色
    """
    # Check input data
    __input_args_dtype_check(show_arrow, save_gif, filename, view_angle, view_dist)

    assert type(init_state) == paddle.Tensor or type(init_state) == np.ndarray, \
        'the type of input data should be "paddle.Tensor" or "numpy.ndarray".'
    assert type(rotating_angle) == tuple or type(rotating_angle) == list, \
        'the type of rotating_angle should be "tuple" or "list".'
    assert len(rotating_angle) == 3, \
        'the rotating_angle must include [theta=paddle.Tensor, phi=paddle.Tensor, lam=paddle.Tensor].'
    for i in range(3):
        assert type(rotating_angle[i]) == paddle.Tensor or type(rotating_angle[i]) == float, \
            'the rotating_angle must include [theta=paddle.Tensor, phi=paddle.Tensor, lam=paddle.Tensor].'
    if color_scheme is not None:
        assert type(color_scheme) == list and len(color_scheme) <= 3, \
            'the type of "color_scheme" should be "list" and ' \
            'the length of "color_scheme" should be less than or equal to "3".'
        for i in range(len(color_scheme)):
            assert type(color_scheme[i]) == str, \
                'the type of "color_scheme[i] should be "str".'

    # Assign a value to an empty variable
    if filename is None:
        filename = 'rotation_in_bloch_sphere.gif'

    # Assign colors to bloch vectors
    color_list = ['orangered', 'lightsalmon', 'darkred']
    if color_scheme is not None:
        for i in range(len(color_scheme)):
            color_list[i] = color_scheme[i]
    set_init_color, set_trac_color, set_end_color = color_list

    theta, phi, lam = rotating_angle

    # Convert Tensor to numpy
    if type(init_state) == paddle.Tensor:
        init_state = init_state.numpy()

    # Convert state_vector to density_matrix
    if init_state.size == 2:
        state_vector = init_state
        init_state = np.outer(state_vector, np.conj(state_vector))

    # Rotating angle
    def rotating_operation(rotating_angle_each):
        gate_matrix = simulator.u_gate_matrix(rotating_angle_each)
        return np.matmul(np.matmul(gate_matrix, init_state), gate_matrix.conj().T)

    # Rotating angle division
    rotating_frame = 50
    rotating_angle_list = []
    state = []
    for i in range(rotating_frame + 1):
        angle_each = [theta / rotating_frame * i, phi / rotating_frame * i, lam / rotating_frame * i]
        rotating_angle_list.append(angle_each)
        state.append(rotating_operation(angle_each))

    state_len = len(state)
    # Calc the bloch_vectors
    bloch_vector_list = []
    for i in range(state_len):
        bloch_vector_tmp = __density_matrix_convert_to_bloch_vector(state[i])
        bloch_vector_list.append(bloch_vector_tmp)

    # List must be converted to array for slicing.
    bloch_vectors = np.array(bloch_vector_list)

    # A update function for animation class
    def update(frame):
        frame = frame + 2
        if frame <= len(bloch_vectors) - 1:
            __plot_bloch_sphere(
                ax, bloch_vectors[1:frame], show_arrow=show_arrow, clear_plt=True,
                rotating_angle_list=rotating_angle_list,
                view_angle=view_angle, view_dist=view_dist, set_color=set_trac_color
            )

            # The starting and ending bloch vector has to be shown
            # show starting vector
            __plot_bloch_sphere(
                ax, bloch_vectors[:1],  show_arrow=True, clear_plt=False,
                view_angle=view_angle, view_dist=view_dist, set_color=set_init_color
            )

        # Show ending vector
        if frame == len(bloch_vectors):
            __plot_bloch_sphere(
                ax, bloch_vectors[frame - 1:frame], show_arrow=True, clear_plt=False,
                view_angle=view_angle, view_dist=view_dist, set_color=set_end_color
            )

    if save_gif:
        # Helper function to plot vectors on a sphere.
        fig = plt.figure(figsize=(8, 8), dpi=100)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = fig.add_subplot(111, projection='3d')

        # Dynamic update and save
        stop_frames = 15
        frames_num = len(bloch_vectors) - 2 + stop_frames
        anim = animation.FuncAnimation(fig, update, frames=frames_num, interval=100, repeat=False)
        anim.save(filename, dpi=100, writer='pillow')
        # close the plt
        plt.close(fig)

    # Helper function to plot vectors on a sphere.
    fig = plt.figure(figsize=(8, 8), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, projection='3d')

    # Draw the penultimate bloch vector
    update(len(bloch_vectors) - 3)
    # Draw the last bloch vector
    update(len(bloch_vectors) - 2)

    plt.show()


def plot_density_matrix_graph(density_matrix, size=0.3):
    r"""密度矩阵可视化工具。

    Args:
        density_matrix (numpy.ndarray or paddle.Tensor): 多量子比特的量子态的状态向量或者密度矩阵,要求量子数大于 1
        size (float): 条宽度，在 0 到 1 之间，默认为 0.3
    """
    if not isinstance(density_matrix, (np.ndarray, paddle.Tensor)):
        msg = f'Expected density_matrix to be np.ndarray or paddle.Tensor, but got {type(density_matrix)}'
        raise TypeError(msg)
    if isinstance(density_matrix, paddle.Tensor):
        density_matrix = density_matrix.numpy()
    if density_matrix.shape[0] != density_matrix.shape[1]:
        msg = f'Expected density matrix dim0 equal to dim1, but got dim0={density_matrix.shape[0]}, dim1={density_matrix.shape[1]}'
        raise ValueError(msg)

    real = density_matrix.real
    imag = density_matrix.imag

    figure = plt.figure()
    ax_real = figure.add_subplot(121, projection='3d', title="real")
    ax_imag = figure.add_subplot(122, projection='3d', title="imag")

    xx, yy = np.meshgrid(
        list(range(real.shape[0])), list(range(real.shape[1])))
    xx, yy = xx.ravel(), yy.ravel()
    real = real.reshape(-1)
    imag = imag.reshape(-1)

    ax_real.bar3d(xx, yy, np.zeros_like(real), size, size, np.abs(real))
    ax_imag.bar3d(xx, yy, np.zeros_like(imag), size, size, np.abs(imag))
    plt.show()

    return


def image_to_density_matrix(image_filepath):
    r"""将图片编码为密度矩阵

    Args:
        image_filepath (str): 图片文件的路径

    Return:
        rho (numpy.ndarray): 编码得到的密度矩阵
    """
    image_matrix = matplotlib.image.imread(image_filepath)

    # Converting images to grayscale
    image_matrix = image_matrix.mean(axis=2)

    # Fill the matrix so that it becomes a matrix whose shape is [2**n,2**n]
    length = int(2**np.ceil(np.log2(np.max(image_matrix.shape))))
    image_matrix = np.pad(image_matrix, ((0, length-image_matrix.shape[0]), (0, length-image_matrix.shape[1])), 'constant')
    # Density matrix whose trace  is 1
    rho = image_matrix@image_matrix.T
    rho = rho/np.trace(rho)
    return rho


def pauli_basis(n):
    r"""生成 n 量子比特的泡利基空间

    Args:
        n (int): 量子比特的数量

    Return:
        tuple:
            - basis_str: 泡利基空间的一组基底表示（array形式）
            - label_str: 泡利基空间对应的一组基底表示（标签形式），形如``[ 'X', 'Y', 'Z', 'I']``
    """
    sigma_x = np.array([[0, 1],  [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0],  [0, -1]], dtype=np.complex128)
    sigma_id = np.array([[1, 0],  [0, 1]], dtype=np.complex128)
    pauli = [sigma_x, sigma_y, sigma_z, sigma_id]
    labels = ['X', 'Y', 'Z', 'I']

    num_qubits = n
    num = 1
    if num_qubits > 0:
        basis_str = pauli[:]
        label_str = labels[:]
        pauli_basis = pauli[:]
        pauli_label = labels[:]
        while num < num_qubits:
            length = len(basis_str)
            for i in range(4):
                for j in range(length):
                    basis_str.append(np.kron(basis_str[j], pauli_basis[i]))
                    label_str.append(label_str[j] + pauli_label[i])
            basis_str = basis_str[-1:-4**(num+1)-1:-1]
            label_str = label_str[-1:-4**(num+1)-1:-1]
            num += 1
        return basis_str, label_str


def decompose(matrix):
    r"""生成 n 量子比特的泡利基空间

    Args:
        matrix (numpy.ndarray): 要分解的矩阵

    Return:
        pauli_form (list): 返回矩阵分解后的哈密顿量，形如 ``[[1, 'Z0, Z1'], [2, 'I']]``
    """
    if np.log2(len(matrix)) % 1 != 0:
        print("Please input correct matrix!")
        return -1
    basis_space, label_str = pauli_basis(np.log2(len(matrix)))
    coefficients = []  # 对应的系数
    pauli_word = []  # 对应的label
    pauli_form = []  # 输出pauli_str list形式：[[1, 'Z0, Z1'], [2, 'I']]
    for i in range(len(basis_space)):
        # 求系数
        a_ij = 1/len(matrix) * np.trace(matrix@basis_space[i])
        if a_ij != 0:
            if a_ij.imag != 0:
                coefficients.append(a_ij)
            else:
                coefficients.append(a_ij.real)
            pauli_word.append(label_str[i])
    for i in range(len(coefficients)):
        pauli_site = []  # 临时存放一个基
        pauli_site.append(coefficients[i])
        word = ''
        for j in range(len(pauli_word[0])):
            if pauli_word[i] == 'I'*int(np.log2(len(matrix))):
                word = 'I'  # 和Hamiltonian类似，若全是I就用一个I指代
                break
            if pauli_word[i][j] == 'I':
                continue   # 如果是I就不加数字下标
            if j != 0 and len(word) != 0:
                word += ','
            word += pauli_word[i][j]
            word += str(j)  # 对每一个label加标签，说明是作用在哪个比特
        pauli_site.append(word)  # 添加上对应作用的门
        pauli_form.append(pauli_site)

    return pauli_form


class Hamiltonian:
    r""" Paddle Quantum 中的 Hamiltonian ``class``。

    用户可以通过一个 Pauli string 来实例化该 ``class``。
    """
    def __init__(self, pauli_str, compress=True):
        r""" 创建一个 Hamiltonian 类。

        Args:
            pauli_str (list): 用列表定义的 Hamiltonian，如 ``[(1, 'Z0, Z1'), (2, 'I')]``
            compress (bool): 是否对输入的 list 进行自动合并（例如 ``(1, 'Z0, Z1')`` 和 ``(2, 'Z1, Z0')`` 这两项将被自动合并），默认为 ``True``

        Note:
            如果设置 ``compress=False``，则不会对输入的合法性进行检验。
        """
        self.__coefficients = None
        self.__terms = None
        self.__pauli_words_r = []
        self.__pauli_words = []
        self.__sites = []
        self.__nqubits = None
        # when internally updating the __pauli_str, be sure to set __update_flag to True
        self.__pauli_str = pauli_str
        self.__update_flag = True
        self.__decompose()
        if compress:
            self.__compress()

    def __getitem__(self, indices):
        new_pauli_str = []
        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, slice):
            indices = list(range(self.n_terms)[indices])
        elif isinstance(indices, tuple):
            indices = list(indices)

        for index in indices:
            new_pauli_str.append([self.coefficients[index], ','.join(self.terms[index])])
        return Hamiltonian(new_pauli_str, compress=False)

    def __add__(self, h_2):
        new_pauli_str = self.pauli_str.copy()
        if isinstance(h_2, float) or isinstance(h_2, int):
            new_pauli_str.extend([[float(h_2), 'I']])
        else:
            new_pauli_str.extend(h_2.pauli_str)
        return Hamiltonian(new_pauli_str)

    def __mul__(self, other):
        new_pauli_str = copy.deepcopy(self.pauli_str)
        for i in range(len(new_pauli_str)):
            new_pauli_str[i][0] *= other
        return Hamiltonian(new_pauli_str, compress=False)

    def __sub__(self, other):
        return self.__add__(other.__mul__(-1))

    def __str__(self):
        str_out = ''
        for idx in range(self.n_terms):
            str_out += '{} '.format(self.coefficients[idx])
            for _ in range(len(self.terms[idx])):
                str_out += self.terms[idx][_]
                if _ != len(self.terms[idx]) - 1:
                    str_out += ', '
            if idx != self.n_terms - 1:
                str_out += '\n'
        return str_out

    def __repr__(self):
        return 'paddle_quantum.Hamiltonian object: \n' + self.__str__()

    @property
    def n_terms(self):
        r"""返回该哈密顿量的项数

        Returns:
            int :哈密顿量的项数
        """
        return len(self.__pauli_str)

    @property
    def pauli_str(self):
        r"""返回哈密顿量对应的 Pauli string

        Returns:
            list : 哈密顿量对应的 Pauli string
        """
        return self.__pauli_str

    @property
    def terms(self):
        r"""返回哈密顿量中的每一项构成的列表

        Returns:
            list :哈密顿量中的每一项，i.e. ``[['Z0, Z1'], ['I']]``
        """
        if self.__update_flag:
            self.__decompose()
            return self.__terms
        else:
            return self.__terms

    @property
    def coefficients(self):
        r""" 返回哈密顿量中的每一项对应的系数构成的列表

        Returns:
            list :哈密顿量中每一项的系数，i.e. ``[1.0, 2.0]``
        """
        if self.__update_flag:
            self.__decompose()
            return self.__coefficients
        else:
            return self.__coefficients

    @property
    def pauli_words(self):
        r"""返回哈密顿量中每一项对应的 Pauli word 构成的列表

        Returns:
            list :每一项对应的 Pauli word，i.e. ``['ZIZ', 'IIX']``
        """
        if self.__update_flag:
            self.__decompose()
            return self.__pauli_words
        else:
            return self.__pauli_words

    @property
    def pauli_words_r(self):
        r"""返回哈密顿量中每一项对应的简化（不包含 I） Pauli word 组成的列表

        Returns:
            list :不包含 "I" 的 Pauli word 构成的列表，i.e. ``['ZXZZ', 'Z', 'X']``
        """
        if self.__update_flag:
            self.__decompose()
            return self.__pauli_words_r
        else:
            return self.__pauli_words_r

    @property
    def sites(self):
        r"""返回该哈密顿量中的每一项对应的量子比特编号组成的列表

        Returns:
            list :哈密顿量中每一项所对应的量子比特编号构成的列表，i.e. ``[[1, 2], [0]]``
        """
        if self.__update_flag:
            self.__decompose()
            return self.__sites
        else:
            return self.__sites

    @property
    def n_qubits(self):
        r"""返回该哈密顿量对应的量子比特数

        Returns:
            int :量子比特数
        """
        if self.__update_flag:
            self.__decompose()
            return self.__nqubits
        else:
            return self.__nqubits

    def __decompose(self):
        r"""将哈密顿量分解为不同的形式

        Notes:
            这是一个内部函数，你不需要直接使用它
            这是一个比较基础的函数，它负责将输入的 Pauli string 拆分为不同的形式并存储在内部变量中
        """
        self.__pauli_words = []
        self.__pauli_words_r = []
        self.__sites = []
        self.__terms = []
        self.__coefficients = []
        self.__nqubits = 1
        new_pauli_str = []
        for coefficient, pauli_term in self.__pauli_str:
            pauli_word_r = ''
            site = []
            single_pauli_terms = re.split(r',\s*', pauli_term.upper())
            self.__coefficients.append(float(coefficient))
            self.__terms.append(single_pauli_terms)
            for single_pauli_term in single_pauli_terms:
                match_I = re.match(r'I', single_pauli_term, flags=re.I)
                if match_I:
                    assert single_pauli_term[0].upper() == 'I', \
                        'The offset is defined with a sole letter "I", i.e. (3.0, "I")'
                    pauli_word_r += 'I'
                    site.append('')
                else:
                    match = re.match(r'([XYZ])([0-9]+)', single_pauli_term, flags=re.I)
                    if match:
                        pauli_word_r += match.group(1).upper()
                        assert int(match.group(2)) not in site, 'each Pauli operator should act on different qubit'
                        site.append(int(match.group(2)))
                    else:
                        raise Exception(
                            'Operators should be defined with a string composed of Pauli operators followed' +
                            'by qubit index on which it act, separated with ",". i.e. "Z0, X1"')
                    self.__nqubits = max(self.__nqubits, int(match.group(2)) + 1)
            self.__pauli_words_r.append(pauli_word_r)
            self.__sites.append(site)
            new_pauli_str.append([float(coefficient), pauli_term.upper()])

        for term_index in range(len(self.__pauli_str)):
            pauli_word = ['I' for _ in range(self.__nqubits)]
            site = self.__sites[term_index]
            for site_index in range(len(site)):
                if type(site[site_index]) == int:
                    pauli_word[site[site_index]] = self.__pauli_words_r[term_index][site_index]
            self.__pauli_words.append(''.join(pauli_word))
            self.__pauli_str = new_pauli_str
            self.__update_flag = False

    def __compress(self):
        r""" 对同类项进行合并。

        Notes:
            这是一个内部函数，你不需要直接使用它
        """
        if self.__update_flag:
            self.__decompose()
        else:
            pass
        new_pauli_str = []
        flag_merged = [False for _ in range(self.n_terms)]
        for term_idx_1 in range(self.n_terms):
            if not flag_merged[term_idx_1]:
                for term_idx_2 in range(term_idx_1 + 1, self.n_terms):
                    if not flag_merged[term_idx_2]:
                        if self.pauli_words[term_idx_1] == self.pauli_words[term_idx_2]:
                            self.__coefficients[term_idx_1] += self.__coefficients[term_idx_2]
                            flag_merged[term_idx_2] = True
                    else:
                        pass
                if self.__coefficients[term_idx_1] != 0:
                    new_pauli_str.append([self.__coefficients[term_idx_1], ','.join(self.__terms[term_idx_1])])
        self.__pauli_str = new_pauli_str
        self.__update_flag = True

    def decompose_with_sites(self):
        r"""将 pauli_str 分解为系数、泡利字符串的简化形式以及它们分别作用的量子比特下标。

        Returns:
            tuple: 包含如下元素的 tuple:

                 - coefficients (list): 元素为每一项的系数
                 - pauli_words_r (list): 元素为每一项的泡利字符串的简化形式，例如 'Z0, Z1, X3' 这一项的泡利字符串为 'ZZX'
                 - sites (list): 元素为每一项作用的量子比特下标，例如 'Z0, Z1, X3' 这一项的 site 为 [0, 1, 3]

        """
        if self.__update_flag:
            self.__decompose()
        return self.coefficients, self.__pauli_words_r, self.__sites

    def decompose_pauli_words(self):
        r"""将 pauli_str 分解为系数和泡利字符串。

        Returns:
            tuple: 包含如下元素的 tuple:

                - coefficients(list): 元素为每一项的系数
                - pauli_words(list): 元素为每一项的泡利字符串，例如 'Z0, Z1, X3' 这一项的泡利字符串为 'ZZIX'
        """
        if self.__update_flag:
            self.__decompose()
        else:
            pass
        return self.coefficients, self.__pauli_words

    def construct_h_matrix(self, qubit_num=None):
        r"""构建 Hamiltonian 在 Z 基底下的矩阵。

        Returns:
            np.ndarray: Z 基底下的哈密顿量矩阵形式
        """
        coefs, pauli_words, sites = self.decompose_with_sites()
        if qubit_num is None:
            qubit_num = 1
            for site in sites:
                if type(site[0]) is int:
                    qubit_num = max(qubit_num, max(site) + 1)
        else:
            assert qubit_num >= self.n_qubits, "输入的量子数不小于哈密顿量表达式中所对应的量子比特数"
        n_qubit = qubit_num
        h_matrix = np.zeros([2 ** n_qubit, 2 ** n_qubit], dtype='complex64')
        spin_ops = SpinOps(n_qubit, use_sparse=True)
        for idx in range(len(coefs)):
            op = coefs[idx] * sparse.eye(2 ** n_qubit, dtype='complex64')
            for site_idx in range(len(sites[idx])):
                if re.match(r'X', pauli_words[idx][site_idx], re.I):
                    op = op.dot(spin_ops.sigx_p[sites[idx][site_idx]])
                elif re.match(r'Y', pauli_words[idx][site_idx], re.I):
                    op = op.dot(spin_ops.sigy_p[sites[idx][site_idx]])
                elif re.match(r'Z', pauli_words[idx][site_idx], re.I):
                    op = op.dot(spin_ops.sigz_p[sites[idx][site_idx]])
            h_matrix += op
        return h_matrix


class SpinOps:
    r"""矩阵表示下的自旋算符，可以用来构建哈密顿量矩阵或者自旋可观测量。

    """
    def __init__(self, size: int, use_sparse=False):
        r"""SpinOps 的构造函数，用于实例化一个 SpinOps 对象。

        Args:
            size (int): 系统的大小（有几个量子比特）
            use_sparse (bool): 是否使用 sparse matrix 计算，默认为 ``False``
        """
        self.size = size
        self.id = sparse.eye(2, dtype='complex128')
        self.__sigz = sparse.bsr.bsr_matrix([[1, 0], [0, -1]], dtype='complex64')
        self.__sigy = sparse.bsr.bsr_matrix([[0, -1j], [1j, 0]], dtype='complex64')
        self.__sigx = sparse.bsr.bsr_matrix([[0, 1], [1, 0]], dtype='complex64')
        self.__sigz_p = []
        self.__sigy_p = []
        self.__sigx_p = []
        self.__sparse = use_sparse
        for i in range(self.size):
            self.__sigz_p.append(self.__direct_prod_op(spin_op=self.__sigz, spin_index=i))
            self.__sigy_p.append(self.__direct_prod_op(spin_op=self.__sigy, spin_index=i))
            self.__sigx_p.append(self.__direct_prod_op(spin_op=self.__sigx, spin_index=i))

    @property
    def sigz_p(self):
        r""" :math:`Z` 基底下的 :math:`S^z_i` 算符。

        Returns:
            list : :math:`S^z_i` 算符组成的列表，其中每一项对应不同的 :math:`i`
        """
        return self.__sigz_p

    @property
    def sigy_p(self):
        r""" :math:`Z` 基底下的 :math:`S^y_i` 算符。

        Returns:
            list : :math:`S^y_i` 算符组成的列表，其中每一项对应不同的 :math:`i`
        """
        return self.__sigy_p

    @property
    def sigx_p(self):
        r""" :math:`Z` 基底下的 :math:`S^x_i` 算符。

        Returns:
            list : :math:`S^x_i` 算符组成的列表，其中每一项对应不同的 :math:`i`
        """
        return self.__sigx_p

    def __direct_prod_op(self, spin_op, spin_index):
        r"""直积，得到第 n 个自旋（量子比特）上的自旋算符

        Args:
            spin_op: 单体自旋算符
            spin_index: 标记第 n 个自旋（量子比特）

        Returns:
            scipy.sparse or np.ndarray: 直积后的自旋算符，其数据类型取决于 self.__use_sparse
        """
        s_p = copy.copy(spin_op)
        for i in range(self.size):
            if i < spin_index:
                s_p = sparse.kron(self.id, s_p)
            elif i > spin_index:
                s_p = sparse.kron(s_p, self.id)
        if self.__sparse:
            return s_p
        else:
            return s_p.toarray()


def __input_args_dtype_check(
        show_arrow,
        save_gif,
        filename,
        view_angle,
        view_dist
):
    r"""
    该函数实现对输入默认参数的数据类型检查，保证输入函数中的参数为所允许的数据类型

    Args:
        show_arrow (bool): 是否展示向量的箭头，默认为 False
        save_gif (bool): 是否存储 gif 动图
        filename (str): 存储的 gif 动图的名字
        view_angle (list or tuple): 视图的角度，
            第一个元素为关于xy平面的夹角[0-360],第二个元素为关于xz平面的夹角[0-360], 默认为 (30, 45)
        view_dist (int): 视图的距离，默认为 7
    """

    if show_arrow is not None:
        assert type(show_arrow) == bool, \
            'the type of "show_arrow" should be "bool".'
    if save_gif is not None:
        assert type(save_gif) == bool, \
            'the type of "save_gif" should be "bool".'
    if save_gif:
        if filename is not None:
            assert type(filename) == str, \
                'the type of "filename" should be "str".'
            other, ext = os.path.splitext(filename)
            assert ext == '.gif', 'The suffix of the file name must be "gif".'
            # If it does not exist, create a folder
            path, file = os.path.split(filename)
            if not os.path.exists(path):
                os.makedirs(path)
    if view_angle is not None:
        assert type(view_angle) == list or type(view_angle) == tuple, \
            'the type of "view_angle" should be "list" or "tuple".'
        for i in range(2):
            assert type(view_angle[i]) == int, \
                'the type of "view_angle[0]" and "view_angle[1]" should be "int".'
    if view_dist is not None:
        assert type(view_dist) == int, \
            'the type of "view_dist" should be "int".'


class QuantumFisher:
    r"""量子费舍信息及相关量的计算器。

    Attributes:
        cir (UAnsatz): 需要计算量子费舍信息的参数化量子电路
    """
    def __init__(self, cir):
        r"""QuantumFisher 的构造函数，用于实例化一个量子费舍信息的计算器。

        Args:
            cir (UAnsatz): 需要计算量子费舍信息的参数化量子电路
        """
        self.cir = cir

    def get_qfisher_matrix(self):
        r"""利用二阶参数平移规则计算量子费舍信息矩阵。

        Returns:
            numpy.ndarray: 量子费舍信息矩阵

        代码示例:

        .. code-block:: python

            import paddle
            from paddle_quantum.circuit import UAnsatz
            from paddle_quantum.utils import QuantumFisher

            cir = UAnsatz(1)
            cir.ry(theta=paddle.to_tensor(0., dtype="float64", stop_gradient=False),
                which_qubit=0)
            cir.rz(theta=paddle.to_tensor(0., dtype="float64", stop_gradient=False),
                which_qubit=0)

            qf = QuantumFisher(cir)
            qfim = qf.get_qfisher_matrix()
            print(f'The QFIM at {cir.get_param().tolist()} is \n {qfim}.')

        ::

            The QFIM at [0.0, 0.0] is
            [[1. 0.]
            [0. 0.]].
        """
        # Get the real-time parameters from the UAnsatz class
        list_param = self.cir.get_param().tolist()
        num_param = len(list_param)
        # Initialize a numpy array to record the QFIM
        qfim = np.zeros((num_param, num_param))
        # Assign the signs corresponding to the four terms in a QFIM element
        list_sign = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
        # Run the circuit and record the current state vector
        psi = self.cir.run_state_vector().numpy()
        # For each QFIM element
        for i in range(0, num_param):
            for j in range(i, num_param):
                # For each term in each element
                for sign_i, sign_j in list_sign:
                    # Shift the parameters by pi/2 * sign
                    list_param[i] += np.pi / 2 * sign_i
                    list_param[j] += np.pi / 2 * sign_j
                    # Update the parameters in the circuit
                    self.cir.update_param(list_param)
                    # Run the shifted circuit and record the shifted state vector
                    psi_shift = self.cir.run_state_vector().numpy()
                    # Calculate each term as the fidelity with a sign factor
                    qfim[i][j] += abs(np.vdot(
                        psi_shift, psi))**2 * sign_i * sign_j * (-0.5)
                    # De-shift the parameters
                    list_param[i] -= np.pi / 2 * sign_i
                    list_param[j] -= np.pi / 2 * sign_j
                    self.cir.update_param(list_param)
                if i != j:
                    # The QFIM is symmetric
                    qfim[j][i] = qfim[i][j]

        return qfim

    def get_qfisher_norm(self, direction, step_size=0.01):
        r"""利用有限差分计算沿给定方向的量子费舍信息的投影。

        Args:
            direction (list): 要计算量子费舍信息投影的方向
            step_size (float, optional): 有限差分的步长，默认为 0.01

        Returns:
            float: 沿给定方向的量子费舍信息的投影

        代码示例:

        .. code-block:: python

            import paddle
            from paddle_quantum.circuit import UAnsatz
            from paddle_quantum.utils import QuantumFisher

            cir = UAnsatz(2)
            cir.ry(theta=paddle.to_tensor(0., dtype="float64", stop_gradient=False),
                which_qubit=0)
            cir.ry(theta=paddle.to_tensor(0., dtype="float64", stop_gradient=False),
                which_qubit=1)
            cir.cnot(control=[0, 1])
            cir.ry(theta=paddle.to_tensor(0., dtype="float64", stop_gradient=False),
                which_qubit=0)
            cir.ry(theta=paddle.to_tensor(0., dtype="float64", stop_gradient=False),
                which_qubit=1)

            qf = QuantumFisher(cir)
            v = [1,1,1,1]
            qfi_norm = qf.get_qfisher_norm(direction=v)
            print(f'The QFI norm along {v} at {cir.get_param().tolist()} is {qfi_norm:.7f}')

        ::

            The QFI norm along [1, 1, 1, 1] at [0.0, 0.0, 0.0, 0.0] is 5.9996250
        """
        # Get the real-time parameters
        list_param = self.cir.get_param().tolist()
        # Run the circuit and record the current state vector
        psi = self.cir.run_state_vector().numpy()
        # Check whether the length of the input direction vector is equal to the number of the variational parameters
        assert len(list_param) == len(
            direction
        ), "the length of direction vector should be equal to the number of the parameters"
        # Shift the parameters by step_size * direction
        array_params_shift = np.array(
            list_param) + np.array(direction) * step_size
        # Update the parameters in the circuit
        self.cir.update_param(array_params_shift.tolist())
        # Run the shifted circuit and record the shifted state vector
        psi_shift = self.cir.run_state_vector().numpy()
        # Calculate quantum Fisher-Rao norm along the given direction
        qfisher_norm = (1 - abs(np.vdot(psi_shift, psi))**2) * 4 / step_size**2
        # De-shift the parameters and update
        self.cir.update_param(list_param)

        return qfisher_norm

    def get_eff_qdim(self, num_param_samples=4, tol=None):
        r"""计算有效量子维数，即量子费舍信息矩阵的秩在整个参数空间的最大值。

        Args:
            num_param_samples (int, optional): 用来估计有效量子维数时所用的参数样本量，默认为 4
            tol (float, optional): 奇异值的最小容差，低于此容差的奇异值认为是 0，默认为 None，其含义同 ``numpy.linalg.matrix_rank()``

        Returns:
            int: 给定量子电路对应的有效量子维数

        代码示例:

        .. code-block:: python

            import paddle
            from paddle_quantum.circuit import UAnsatz
            from paddle_quantum.utils import QuantumFisher

            cir = UAnsatz(1)
            cir.rz(theta=paddle.to_tensor(0., dtype="float64", stop_gradient=False),
                which_qubit=0)
            cir.ry(theta=paddle.to_tensor(0., dtype="float64", stop_gradient=False),
                which_qubit=0)

            qf = QuantumFisher(cir)
            print(cir)
            print(f'The number of parameters of -Rz-Ry- is {len(cir.get_param().tolist())}')
            print(f'The effective quantum dimension -Rz-Ry- is {qf.get_eff_qdim()}')

        ::

            --Rz(0.000)----Ry(0.000)--

            The number of parameters of -Rz-Ry- is 2
            The effective quantum dimension -Rz-Ry- is 1
        """
        # Get the real-time parameters
        list_param = self.cir.get_param().tolist()
        num_param = len(list_param)
        # Generate random parameters
        param_samples = 2 * np.pi * np.random.random(
            (num_param_samples, num_param))
        # Record the ranks
        list_ranks = []
        # Here it has been assumed that the set of points that do not maximize the rank of QFIMs, as singularities, form a null set.
        # Thus one can find the maximal rank using a few samples.
        for param in param_samples:
            # Set the random parameters
            self.cir.update_param(param.tolist())
            # Calculate the ranks
            list_ranks.append(self.get_qfisher_rank(tol))
        # Recover the original parameters
        self.cir.update_param(list_param)

        return max(list_ranks)

    def get_qfisher_rank(self, tol=None):
        r"""计算量子费舍信息矩阵的秩。

        Args:
            tol (float, optional): 奇异值的最小容差，低于此容差的奇异值认为是 0，默认为 None，其含义同 ``numpy.linalg.matrix_rank()``

        Returns:
            int: 量子费舍信息矩阵的秩
        """
        qfisher_rank = np.linalg.matrix_rank(self.get_qfisher_matrix(),
                                             tol,
                                             hermitian=True)
        return qfisher_rank


class ClassicalFisher:
    r"""经典费舍信息及相关量的计算器。

    Attributes:
        model (paddle.nn.Layer): 经典或量子神经网络模型的实例
        num_thetas (int): 参数集合的数量
        num_inputs (int): 输入的样本数量
    """
    def __init__(self,
                 model,
                 num_thetas,
                 num_inputs,
                 model_type='quantum',
                 **kwargs):
        r"""ClassicalFisher 的构造函数，用于实例化一个经典费舍信息的计算器。

        Args:
            model (paddle.nn.Layer): 经典或量子神经网络模型的实例
            num_thetas (int): 参数集合的数量
            num_inputs (int): 输入的样本数量
            model_type (str, optional): 模型是经典 "classical" 的还是量子 "quantum" 的，默认是量子的

        Note:
            这里 ``**kwargs`` 包含如下选项:
                - size (list): 经典神经网络各层神经元的数量
                - num_qubits (int): 量子神经网络量子比特的数量
                - depth (int): 量子神经网络的深度
                - encoding (str): 量子神经网络中经典数据的编码方式，目前支持 "IQP" 和 "re-uploading"

        Raises:
            ValueError: 不被支持的编码方式
            ValueError: 不被支持的模型类别
        """
        self.model = model
        self.num_thetas = num_thetas
        self.num_inputs = num_inputs
        self._model_type = model_type
        if self._model_type == 'classical':
            layer_dims = kwargs['size']
            self.model_args = [layer_dims]
            self.input_size = layer_dims[0]
            self.output_size = layer_dims[-1]
            self.num_params = sum(layer_dims[i] * layer_dims[i + 1]
                                  for i in range(len(layer_dims) - 1))
        elif self._model_type == 'quantum':
            num_qubits = kwargs['num_qubits']
            depth = kwargs['depth']
            # Supported QNN encoding: ‘IQP' and 're-uploading'
            encoding = kwargs['encoding']
            self.model_args = [num_qubits, depth, encoding]
            self.input_size = num_qubits
            # Default dimension of output layer = 1
            self.output_size = 1
            # Determine the number of model parameters for different encoding types
            if encoding == 'IQP':
                self.num_params = 3 * depth * num_qubits
            elif encoding == 're-uploading':
                self.num_params = 3 * (depth + 1) * num_qubits
            else:
                raise ValueError('Non-existent encoding method')
        else:
            raise ValueError(
                'The model type should be equal to either classical or quantum'
            )

        # Generate random data
        np.random.seed(0)
        x = np.random.normal(0, 1, size=(num_inputs, self.input_size))
        # Use the same input data for each theta set
        self.x = np.tile(x, (num_thetas, 1))

    def get_gradient(self, x):
        r"""计算输出层关于变分参数的梯度。

        Args:
            x (numpy.ndarray): 输入样本

        Returns:
            numpy.ndarray: 输出层关于变分参数的梯度，数组形状为（输入样本数量, 输出层维数, 变分参数数量）
        """
        if not paddle.is_tensor(x):
            x = paddle.to_tensor(x, stop_gradient=True)
        gradvectors = []
        seed = 0

        pbar = tqdm(desc="running in get_gradient: ",
                    total=len(x),
                    ncols=100,
                    ascii=True)

        for m in range(len(x)):
            pbar.update(1)
            if m % self.num_inputs == 0:
                seed += 1
            paddle.seed(seed)
            net = self.model(*self.model_args)
            output = net(x[m])
            logoutput = paddle.log(output)
            grad = []
            for i in range(self.output_size):
                net.clear_gradients()
                logoutput[i].backward(retain_graph=True)
                grads = []
                for param in net.parameters():
                    grads.append(param.grad.reshape((-1, )))
                gr = paddle.concat(grads)
                grad.append(gr * paddle.sqrt(output[i]))
            jacobian = paddle.concat(grad)
            # Jacobian matrix corresponding to each data point
            jacobian = paddle.reshape(jacobian,
                                      (self.output_size, self.num_params))
            gradvectors.append(jacobian.detach().numpy())

        pbar.close()

        return gradvectors

    def get_cfisher(self, gradients):
        r"""利用雅可比矩阵计算经典费舍信息矩阵。

        Args:
            gradients (numpy.ndarray): 输出层关于变分参数的梯度, 数组形状为（输入样本数量, 输出层维数, 变分参数数量）

        Returns:
            numpy.ndarray: 经典费舍信息矩阵，数组形状为（输入样本数量, 变分参数数量, 变分参数数量）
        """
        fishers = np.zeros((len(gradients), self.num_params, self.num_params))
        for i in range(len(gradients)):
            grads = gradients[i]
            temp_sum = np.zeros(
                (self.output_size, self.num_params, self.num_params))
            for j in range(self.output_size):
                temp_sum[j] += np.array(
                    np.outer(grads[j], np.transpose(grads[j])))
            fishers[i] += np.sum(temp_sum, axis=0)

        return fishers

    def get_normalized_cfisher(self):
        r"""计算归一化的经典费舍信息矩阵。

        Returns:
            numpy.ndarray: 归一化的经典费舍信息矩阵，数组形状为（输入样本数量, 变分参数数量, 变分参数数量）
        """
        grads = self.get_gradient(self.x)
        fishers = self.get_cfisher(grads)
        fisher_trace = np.trace(np.average(fishers, axis=0))
        # Average over input data
        fisher = np.average(np.reshape(fishers,
                                       (self.num_thetas, self.num_inputs,
                                        self.num_params, self.num_params)),
                            axis=1)
        normalized_cfisher = self.num_params * fisher / fisher_trace

        return normalized_cfisher, fisher_trace

    def get_eff_dim(self, normalized_cfisher, list_num_samples, gamma=1):
        r"""计算经典的有效维数。

        Args:
            normalized_cfisher (numpy.ndarray): 归一化的经典费舍信息矩阵
            list_num_samples (list): 不同样本量构成的列表
            gamma (int, optional): 有效维数定义中包含的一个人为可调参数，默认为 1

        Returns:
            list: 对于不同样本量的有效维数构成的列表
        """
        eff_dims = []
        for n in list_num_samples:
            one_plus_F = np.eye(
                self.num_params) + normalized_cfisher * gamma * n / (
                    2 * np.pi * np.log(n))
            det = np.linalg.slogdet(one_plus_F)[1]
            r = det / 2
            eff_dims.append(2 * (logsumexp(r) - np.log(self.num_thetas)) /
                            np.log(gamma * n / (2 * np.pi * np.log(n))))

        return eff_dims
