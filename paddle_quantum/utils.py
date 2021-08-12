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
from matplotlib import colors as mplcolors
import matplotlib.pyplot as plt
import paddle
from paddle import add, to_tensor
from paddle import kron as kron
from paddle import matmul
from paddle import transpose
from paddle import concat, ones
from paddle import zeros
from scipy.linalg import logm, sqrtm
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import sqrt
from paddle_quantum import simulator
import matplotlib.animation as animation

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
    "pauli_str_to_matrix",
    "partial_transpose_2",
    "partial_transpose",
    "negativity",
    "logarithmic_negativity",
    "is_ppt",
    "Hamiltonian",
    "plot_state_in_bloch_sphere",
    "plot_rotation_in_bloch_sphere",
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

        for i in range(0, 2 ** num_preserve):
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
    rho_eigenvalue, _ = np.linalg.eig(rho)
    entropy = -np.sum(rho_eigenvalue * np.log(rho_eigenvalue))

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
            list :哈密顿量中每一项的系数，i.e.``[1.0, 2.0]``
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
        r"""将 pauli_str 分解为系数，泡利字符串的简化形式，以及它们分别作用的量子比特下标

        Returns:
            tuple: 包含如下元素的 tuple
                coefficients (list): 元素为每一项的系数
                pauli_words_r (list): 元素为每一项的泡利字符串的简化形式，例如 'Z0, Z1, X3' 这一项的泡利字符串为 'ZZX'
                sites (list): 元素为每一项作用的量子比特下标，例如 'Z0, Z1, X3' 这一项的 site 为 [0, 1, 3]
        """
        if self.__update_flag:
            self.__decompose()
        return self.coefficients, self.__pauli_words_r, self.__sites

    def decompose_pauli_words(self):
        r"""将 pauli_str 分解为系数，泡利字符串，以及它们分别作用的量子比特下标

        Returns:
            tuple: 包含如下元素的 tuple
                coefficients(list): 元素为每一项的系数
                pauli_words(list): 元素为每一项的泡利字符串，例如 'Z0, Z1, X3' 这一项的泡利字符串为 'ZZIX'
        """
        if self.__update_flag:
            self.__decompose()
        else:
            pass
        return self.coefficients, self.__pauli_words

    def construct_h_matrix(self):
        r"""构建 Hamiltonian 在 Z 基底下的矩阵

        Returns:
            np.ndarray: Z 基底下的哈密顿量矩阵形式
        """
        coefs, pauli_words, sites = self.decompose_with_sites()
        n_qubit = 1
        for site in sites:
            if type(site[0]) is int:
                n_qubit = max(n_qubit, max(site) + 1)
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
    r"""矩阵表示下的自旋算符，可以用来构建哈密顿量矩阵。

    """
    def __init__(self, size: int, use_sparse=False):
        r"""SpinOps 的构造函数，用于实例化一个 SpinOps 对象

        Args:
            size (int): 系统的大小（有几个量子比特）
            use_sparse (bool): 是否使用 sparse matrix 计算，默认为 True
        """
        self.size = size
        self.id = sparse.eye(2, dtype='complex128')
        self.sigz = sparse.bsr.bsr_matrix([[1, 0], [0, -1]], dtype='complex64')
        self.sigy = sparse.bsr.bsr_matrix([[0, -1j], [1j, 0]], dtype='complex64')
        self.sigx = sparse.bsr.bsr_matrix([[0, 1], [1, 0]], dtype='complex64')
        self.sigz_p = []
        self.sigy_p = []
        self.sigx_p = []
        self.__sparse = use_sparse
        for i in range(self.size):
            self.sigz_p.append(self.__direct_prod_op(spin_op=self.sigz, spin_index=i))
            self.sigy_p.append(self.__direct_prod_op(spin_op=self.sigy, spin_index=i))
            self.sigx_p.append(self.__direct_prod_op(spin_op=self.sigx, spin_index=i))

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
        show_arrow (bool): 是否展示向量的箭头，默认为False
        save_gif (bool): 是否存储gif动图，默认10帧，3帧长出bloch向量，7帧转动bloch视角
        filename (str): 存储的gif动图的名字。
        view_angle (list or tuple): 视图的角度，list内第一个元素为关于xy平面的夹角[0-360],第二个元素为关于xz平面的夹角[0-360]
        view_dist (int): 视图的距离，默认为7
    """

    if show_arrow is not None:
        assert type(show_arrow) == bool, \
            'the type of show_arrow should be "bool"'
    if save_gif is not None:
        assert type(save_gif) == bool, \
            'the type of save_gif should be "bool"'
    if save_gif:
        if filename is not None:
            assert type(filename) == str, \
                'the type of filename should be "str"'
            other, ext = os.path.splitext(filename)
            assert ext == '.gif', 'The suffix of the file name must be "gif"'
            # If it does not exist, create a folder 
            path, file = os.path.split(filename)
            if not os.path.exists(path):
                os.makedirs(path)
    if view_angle is not None:
        assert type(view_angle) == list or type(view_angle) == tuple, \
            'the type of view_angle should be "list" or "tuple"'
        for i in range(2):
            assert type(view_angle[i]) == int, \
                'the type of view_angle[0] and view_angle[1] should be "int"'
    if view_dist is not None:
        assert type(view_dist) == int, \
            'the type of view_dist should be "int"'


def __density_matrix_convert_to_bloch_vector(density_matrix):
    r"""该函数将密度矩阵转化为bloch球面上的坐标

    Args:
        density_matrix (numpy.ndarray): 输入的密度矩阵
    Returns:
        bloch_vector (numpy.ndarray): 存储bloch向量的x,y,z坐标，向量的模长，向量的颜色
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
        view_dist=None
):
    r"""将 Bloch 向量展示在 Bloch 球面上

    Args:
        ax (Axes3D(fig)): 画布的句柄
        bloch_vectors (numpy.ndarray): 存储bloch向量的x,y,z坐标，向量的模长，向量的颜色
        show_arrow (bool): 是否展示向量的箭头，默认为False
        clear_plt (bool): 是否要清空画布，默认为True，每次画图的时候清空画布再画图
        rotating_angle_list (list): 旋转角度的列表，用于展示旋转轨迹
        view_angle (list): 视图的角度，list内第一个元素为关于xy平面的夹角[0-360],第二个元素为关于xz平面的夹角[0-360]
        view_dist (int): 视图的距离，默认为7
    """
    # Assign a value to an empty variable
    if view_angle is None:
        view_angle = [30, 45]
    if view_dist is None:
        view_dist = 7

    # Define my_color
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
            bloch_vectors[:, 0], bloch_vectors[:, 1], bloch_vectors[:, 2], c=color, alpha=1
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
            arrow_length_ratio=0.05, color=color, alpha=1,
        )


def plot_state_in_bloch_sphere(
        state,
        show_arrow=False,
        save_gif=False,
        filename=None,
        view_angle=None,
        view_dist=None
):
    r"""将输入的量子态展示在 Bloch 球面上

    Args:
        state (list(numpy.ndarray or paddle.Tensor)): 输入的量子态列表，可以支持态矢量和密度矩阵
        show_arrow (bool): 是否展示向量的箭头，默认为 ``False``
        save_gif (bool): 是否存储 gif 动图，默认为 ``False``
        filename (str): 存储的 gif 动图的名字
        view_angle (list or tuple): 视图的角度，list 内第一个元素为关于 xy 平面的夹角 [0-360]，第二个元素为关于 xz 平面的夹角 [0-360]
        view_dist (int): 视图的距离，默认为 7
    """
    # Check input data
    __input_args_dtype_check(show_arrow, save_gif, filename, view_angle, view_dist)

    assert type(state) == list or type(state) == paddle.Tensor or type(state) == np.ndarray, \
        'the type of input data must be "list" or "paddle.Tensor" or "np.ndarray"'
    if type(state) == paddle.Tensor or type(state) == np.ndarray:
        state = [state]
    state_len = len(state)
    assert state_len >= 1, 'input data is NULL.'
    for i in range(state_len):
        assert type(state[i]) == paddle.Tensor or type(state[i]) == np.ndarray, \
            'the type of input data should be "paddle.Tensor" or "numpy.ndarray".'

    # Assign a value to an empty variable
    if filename is None:
        filename = 'state_in_bloch_sphere.gif'
    if view_angle is None:
        view_angle = [30, 45]
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

    # Helper function to plot vectors on a sphere.
    fig = plt.figure(figsize=(8, 8), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, projection='3d')

    # A update function for animation class
    def update(frame):
        stretch = 3
        view_rotating_angle = 5
        if frame <= stretch:
            new_bloch_vectors = np.zeros(shape=(len(bloch_vectors), 0))
            for j in range(3):
                bloch_column_tmp = bloch_vectors[:, j] / stretch * frame
                new_bloch_vectors = np.insert(new_bloch_vectors, j, values=bloch_column_tmp, axis=1)
            for j in range(2):
                new_bloch_vectors = np.insert(new_bloch_vectors, 3 + j, values=bloch_vectors[:, 3 + j], axis=1)

            __plot_bloch_sphere(
                ax, new_bloch_vectors, show_arrow, clear_plt=True, view_angle=view_angle, view_dist=view_dist
            )

        else:
            new_view_angle = [view_angle[0], view_angle[1] + view_rotating_angle * (frame - stretch)]
            __plot_bloch_sphere(
                ax, bloch_vectors, show_arrow, clear_plt=True, view_angle=new_view_angle, view_dist=view_dist
            )

    # Dynamic update and save
    if save_gif:
        frames_num = 10
        anim = animation.FuncAnimation(fig, update, frames=frames_num, interval=600, repeat=False)
        anim.save(filename, dpi=100, writer='Pillow')
    else:
        __plot_bloch_sphere(
            ax, bloch_vectors, show_arrow, clear_plt=True, view_angle=view_angle, view_dist=view_dist
        )

    plt.show()


def plot_rotation_in_bloch_sphere(
        init_state,
        rotating_angle,
        show_arrow=False,
        save_gif=False,
        filename=None,
        view_angle=None,
        view_dist=None
):
    r"""在 Bloch 球面上刻画从初始量子态开始的旋转轨迹

    Args:
        init_state (numpy.ndarray or paddle.Tensor): 输入的初始量子态，可以支持态矢量和密度矩阵
        rotating_angle (list(float)): 旋转角度 ``[theta, phi, lam]``
        show_arrow (bool): 是否展示向量的箭头，默认为 ``False``
        save_gif (bool): 是否存储 gif 动图，默认为 ``False``
        filename (str): 存储的 gif 动图的名字
        view_angle (list or tuple): 视图的角度，list 内第一个元素为关于 xy 平面的夹角 [0-360]，第二个元素为关于 xz 平面的夹角 [0-360]
        view_dist (int): 视图的距离，默认为 7
    """
    # Check input data
    __input_args_dtype_check(show_arrow, save_gif, filename, view_angle, view_dist)

    assert type(init_state) == paddle.Tensor or type(init_state) == np.ndarray, \
        'the type of input data should be "paddle.Tensor" or "numpy.ndarray".'
    assert type(rotating_angle) == tuple or type(rotating_angle) == list, \
        'the type of rotating_angle should be "tuple" or "list"'
    assert len(rotating_angle) == 3, \
        'the rotating_angle must include [theta=paddle.Tensor, phi=paddle.Tensor, lam=paddle.Tensor].'
    for i in range(3):
        assert type(rotating_angle[i]) == paddle.Tensor or type(rotating_angle[i]) == float, \
            'the rotating_angle must include [theta=paddle.Tensor, phi=paddle.Tensor, lam=paddle.Tensor].'

    # Assign a value to an empty variable
    if filename is None:
        filename = 'rotation_in_bloch_sphere.gif'
    if view_angle is None:
        view_angle = [30, 45]
    if view_dist is None:
        view_dist = 7

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

    # Helper function to plot vectors on a sphere.
    fig = plt.figure(figsize=(8, 8), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, projection='3d')

    # A update function for animation class
    def update(frame):
        frame = frame + 1
        if frame <= len(bloch_vectors):
            __plot_bloch_sphere(
                ax, bloch_vectors[:frame], show_arrow=show_arrow, clear_plt=True,
                rotating_angle_list=rotating_angle_list,
                view_angle=view_angle, view_dist=view_dist
            )

            # The starting and ending bloch vector has to be shown
            # show starting vector
            __plot_bloch_sphere(
                ax, bloch_vectors[:1],  show_arrow=True, clear_plt=False, view_angle=view_angle, view_dist=view_dist,
            )
            # Show ending vector
            if frame == len(bloch_vectors):
                __plot_bloch_sphere(
                    ax, bloch_vectors[frame - 1:frame], show_arrow=True, clear_plt=False,
                    view_angle=view_angle, view_dist=view_dist
                )

    # Dynamic update and save
    stop_frames = 10
    frames_num = len(bloch_vectors) + stop_frames
    anim = animation.FuncAnimation(fig, update, frames=frames_num, interval=100, repeat=False)
    if save_gif:
        anim.save(filename, dpi=100, writer='Pillow')

    plt.show()
