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

import numpy as np
from numpy import trace as np_trace
from numpy import matmul as np_matmul
from numpy import random as np_random
from numpy import zeros as np_zeros
from numpy import eye as np_eye

__all__ = [
    "vec",
    "vec_random",
    "w_state",
    "GHZ_state",
    "density_op",
    "density_op_random",
    "completely_mixed_computational",
    "bell_state",
    "bell_diagonal_state",
    "R_state",
    "S_state",
    "isotropic_state"
]


def vec(i, n):
    r"""生成计算基态 :math:`|e_{i}\rangle` 的 numpy 形式，其中 :math:`|e_{i}\rangle` 的第 :math:`i` 个元素为 1，其余元素为 0。

    Args:
        i(int): 计算基态 :math`|e_{i}\rangle` 的下标 :math:`i`
        n(int): 生成的量子态的量子比特数量

    Returns:
        numpy.ndarray: 计算基态 :math:`|e_{i}\rangle` 的态矢量形式。

    代码示例:

    .. code-block:: python

        from paddle_quantum.state import vec
        vector = vec(1, 3)
        print(vector)

    ::

        [[0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]]
    """
    assert n > 0, 'qubit number must be larger than 1'
    assert 0 <= i <= 2 ** n - 1, 'i should >= 0 and < 2**n (the dimension of the Hilbert space)'
    state = np_zeros([1, 2 ** n])
    state[0][i] = 1
    return state.astype("complex128")


def vec_random(n, real_or_complex=2):
    r"""随机生成一个量子态的 numpy 形式。

    Args:
        n (int): 量子比特数量
        real_or_complex (int, optional): 默认为 2，即生成复数组；若为 1，则生成实数组

    Returns:
        numpy.ndarray: 一个形状为 ``(1, 2**n)`` 的 numpy 数组
    """
    assert n > 0, 'qubit number must be larger than 1'
    assert real_or_complex == 1 or real_or_complex == 2, 'real_or_complex must be 1 or 2'
    # real
    if real_or_complex == 1:
        psi = np_random.randn(1, 2 ** n)
    # complex
    else:
        psi = np_random.randn(1, 2 ** n) + 1j * np_random.randn(1, 2 ** n)
    psi = psi / np.linalg.norm(psi)

    return psi.astype("complex128")


def w_state(n, coeff=None):
    r"""生成一个 W-state 的 numpy 形式。

    Args:
        n (int): 量子比特数量
        coeff (numpy.ndarray, optional): 默认为 ``None`` ，即生成平均概率幅（系数）

    Returns:
        numpy.ndarray: 一个形状为 ``(1, 2**n)`` 的 numpy 数组

    代码示例:

    .. code-block:: python
    
        from paddle_quantum.state import w_state
        vector = w_state(3)
        print(vector)

    ::
    
        [[0.        +0.j 0.57735027+0.j 0.57735027+0.j 0.        +0.j
        0.57735027+0.j 0.        +0.j 0.        +0.j 0.        +0.j]]
    """
    assert n > 0, 'qubit number must be larger than 1'
    
    c = coeff if coeff is not None else np.ones((1, 2 ** n)) / np.sqrt(n)
    assert c.shape[0] == 1 and c.shape[1] == 2 ** n, 'The dimension of coeff is not right'

    state = np_zeros((1, 2 ** n))
    for i in range(n):
        state[0][2 ** i] = c[0][n - i - 1]
        
    return state.astype("complex128")


def GHZ_state(n):
    r"""生成一个 GHZ-state 的 numpy 形式。

    Args:
        n (int): 量子比特数量

    Returns:
        numpy.ndarray: 一个形状为 ``(1, 2**n)`` 的 numpy 数组

    代码示例:

    .. code-block:: python

        from paddle_quantum.state import GHZ_state
        vector = GHZ_state(3)
        print(vector)

    ::

        [[0.70710678+0.j 0.        +0.j 0.        +0.j 0.        +0.j
          0.        +0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]]
    """
    assert n > 2, 'qubit number must be larger than 2'
    state = np_zeros((1, 2 ** n))
    state[0][0] = 1 / np.sqrt(2)
    state[0][-1] = 1 / np.sqrt(2)

    return state.astype("complex128")


def density_op(n):
    r"""生成密度矩阵 :math:`|00..0\rangle \langle00..0|` 的 numpy 形式。

    Args:
        n (int): 量子比特数量

    Returns:
        numpy.ndarray: 一个形状为 ``(2**n, 2**n)`` 的 numpy 数组

    代码示例:

    .. code-block:: python
    
        from paddle_quantum.state import density_op
        state = density_op(2)
        print(state)

    ::

        [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
        [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
        [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
        [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]

    """
    assert n > 0, 'qubit number must be positive'
    rho = np_zeros((2 ** n, 2 ** n))
    rho[0, 0] = 1

    return rho.astype("complex128")


def density_op_random(n, real_or_complex=2, rank=None):
    r"""随机生成一个密度矩阵的 numpy 形式。

    Args:
        n (int): 量子比特数量
        real_or_complex (int, optional): 默认为 2，即生成复数组，若为 1，则生成实数组
        rank (int, optional): 矩阵的秩，默认为 :math:`2^n` （当 ``rank`` 为 ``None`` 时）

    Returns:
        numpy.ndarray: 一个形状为 ``(2**n, 2**n)`` 的 numpy 数组
    """
    assert n > 0, 'qubit number must be positive'
    rank = rank if rank is not None else 2 ** n
    assert 0 < rank <= 2 ** n, 'rank is an invalid number'

    if real_or_complex == 1:
        psi = np_random.randn(2 ** n, rank)
    else:
        psi = np_random.randn(2 ** n, rank) + 1j * np_random.randn(2 ** n, rank)

    psi_dagger = psi.conj().T
    rho = np_matmul(psi, psi_dagger)
    rho = rho / np_trace(rho)
    
    return rho.astype('complex128')


def completely_mixed_computational(n):
    r"""生成完全混合态的密度矩阵的 numpy 形式。

    其矩阵形式为：

    .. math::

        \frac{I}{2^n}

    Args:
        n (int): 量子比特数量

    Returns:
        numpy.ndarray: 一个形状为 ``(2**n, 2**n)`` 的 numpy 数组

    代码示例:

    .. code-block:: python
    
        from paddle_quantum.state import completely_mixed_computational
        state = completely_mixed_computational(2)
        print(state)

    ::

        [[0.25+0.j 0.  +0.j 0.  +0.j 0.  +0.j]
        [0.  +0.j 0.25+0.j 0.  +0.j 0.  +0.j]
        [0.  +0.j 0.  +0.j 0.25+0.j 0.  +0.j]
        [0.  +0.j 0.  +0.j 0.  +0.j 0.25+0.j]]
    """
    assert n > 0, 'qubit number must be positive'
    rho = np_eye(2 ** n) / (2 ** n)
    
    return rho.astype('complex128')


def bell_state(n):
    r"""生成（推广）贝尔态的密度矩阵的 numpy 形式。

    其数学表达形式为：

    .. math::

        |\Phi_{D}\rangle=\frac{1}{\sqrt{D}} \sum_{j=0}^{D-1}|j\rangle_{A}|j\rangle_{B}


    Args:
        n (int): 量子比特数量，必须为大于等于 2 的偶数

    Returns:
        numpy.ndarray: 一个形状为 ``(2**n, 2**n)`` 的 numpy 数组

    代码示例:

    .. code-block:: python
    
        from paddle_quantum.state import bell_state
        state = bell_state(2)
        print(state)

    ::

        [[0.5+0.j 0. +0.j 0. +0.j 0.5+0.j]
        [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
        [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
        [0.5+0.j 0. +0.j 0. +0.j 0.5+0.j]]
    """
    assert n > 0, "Qubit number must be positive"
    assert n % 2 == 0, "Qubit number must be even"

    dim = 2 ** n
    local_dim = 2 ** int(n / 2)
    coeff = 1 / local_dim
    state = np.zeros((dim, dim))
    for i in range(0, dim, local_dim + 1):
        for j in range(0, dim, local_dim + 1):
            state[i, j] = coeff

    return state.astype("complex128")


def bell_diagonal_state(p1, p2, p3, p4):
    r"""生成对角贝尔态的密度矩阵的 numpy 形式。

    其数学表达形式为：

    .. math::

        p_{1}|\Phi^{+}\rangle\langle\Phi^{+}|+p_{2}| \Psi^{+}\rangle\langle\Psi^{+}|+p_{3}| \Phi^{-}\rangle\langle\Phi^{-}| +
        p4|\Psi^{-}\rangle\langle\Psi^{-}|

    Args:
        p1 (float): 第一个分量。
        p2 (float): 第二个分量。
        p3 (float): 第三个分量。
        p4 (float): 第四个分量。
    
    Note:
        四个参数构成了一个概率分布，因此它们是非负数，加起来必为 1。

    Returns:
        numpy.ndarray: 一个形状为 ``(4, 4)`` 的 numpy 数组

    代码示例:

    .. code-block:: python
    
        from paddle_quantum.state import bell_diagonal_state
        state = bell_diagonal_state(0.25, 0.25, 0.25, 0.25)
        print(state)

    ::

        [[0.25+0.j 0.  +0.j 0.  +0.j 0.  +0.j]
        [0.  +0.j 0.25+0.j 0.  +0.j 0.  +0.j]
        [0.  +0.j 0.  +0.j 0.25+0.j 0.  +0.j]
        [0.  +0.j 0.  +0.j 0.  +0.j 0.25+0.j]]
    """
    assert 0 <= p1 <= 1 and 0 <= p2 <= 1 and 0 <= p3 <= 1 and 0 <= p4 <= 1, "Each probability must be in [0, 1]"
    assert abs(p1 + p2 + p3 + p4 - 1) < 1e-6, "The sum of probabilities should be 1"

    coeff = np.sqrt(0.5)
    phi_p_vec = np.array([[coeff, 0, 0, coeff]])
    phi_p_mat = np.matmul(phi_p_vec.T, phi_p_vec)
    phi_m_vec = np.array([[coeff, 0, 0, -coeff]])
    phi_m_mat = np.matmul(phi_m_vec.T, phi_m_vec)
    psi_p_vec = np.array([[0, coeff, coeff, 0]])
    psi_p_mat = np.matmul(psi_p_vec.T, psi_p_vec)
    psi_m_vec = np.array([[0, coeff, -coeff, 0]])
    psi_m_mat = np.matmul(psi_m_vec.T, psi_m_vec)

    state = p1 * phi_p_mat + p2 * psi_p_mat + p3 * phi_m_mat + p4 * psi_m_mat

    return state.astype("complex128")


def R_state(p):
    r"""生成 R-state 的密度矩阵的 numpy 形式。

    其数学表达形式为：

    .. math::

        p|\Psi^{+}\rangle\langle\Psi^{+}| + (1 - p)|11\rangle\langle11|

    Args:
        p (float): 控制生成 R-state 的参数，属于 :math:`[0, 1]` 区间内

    Returns:
        numpy.ndarray: 一个形状为 ``(4, 4)`` 的 numpy 数组

    代码示例:

    .. code-block:: python
    
        from paddle_quantum.state import R_state
        state = R_state(0.5)
        print(state)

    ::

        [[0.  +0.j 0.  +0.j 0.  +0.j 0.  +0.j]
        [0.  +0.j 0.25+0.j 0.25+0.j 0.  +0.j]
        [0.  +0.j 0.25+0.j 0.25+0.j 0.  +0.j]
        [0.  +0.j 0.  +0.j 0.  +0.j 0.5 +0.j]]
    """
    assert 0 <= p <= 1, "Probability must be in [0, 1]"

    coeff = np.sqrt(0.5)
    psi_p_vec = np.array([[0, coeff, coeff, 0]])
    psi_p_mat = np.matmul(psi_p_vec.T, psi_p_vec)
    state_11 = np.zeros((4, 4))
    state_11[3, 3] = 1

    state = p * psi_p_mat + (1 - p) * state_11

    return state.astype("complex128")


def S_state(p):
    r"""生成 S-state 的密度矩阵的 numpy 形式。

    其数学表达形式为：

    .. math::

        p|\Phi^{+}\rangle\langle\Phi^{+}| + (1 - p)|00\rangle\langle00|

    Args:
        p (float): 控制生成 S-state 的参数，属于 :math:`[0, 1]` 区间内

    Returns:
        numpy.ndarray: 一个形状为 ``(4, 4)`` 的 numpy 数组

    代码示例:

    .. code-block:: python
    
        from paddle_quantum.state import S_state
        state = S_state(0.5)
        print(state)

    ::

        [[0.75+0.j 0.  +0.j 0.  +0.j 0.25+0.j]
        [0.  +0.j 0.  +0.j 0.  +0.j 0.  +0.j]
        [0.  +0.j 0.  +0.j 0.  +0.j 0.  +0.j]
        [0.25+0.j 0.  +0.j 0.  +0.j 0.25+0.j]]
    """
    assert 0 <= p <= 1, "Probability must be in [0, 1]"

    phi_p = bell_state(2)
    psi0 = np.zeros_like(phi_p)
    psi0[0, 0] = 1

    state = p * phi_p + (1 - p) * psi0
    return state.astype("complex128")


def isotropic_state(n, p):
    r"""生成 isotropic state 的密度矩阵的 numpy 形式。

    其数学表达形式为：

    .. math::

        p(\frac{1}{\sqrt{D}} \sum_{j=0}^{D-1}|j\rangle_{A}|j\rangle_{B}) + (1 - p)\frac{I}{2^n}

    Args:
        n (int): 量子比特数量
        p (float): 控制生成 isotropic state 的参数，属于 :math:`[0, 1]` 区间内

    Returns:
        numpy.ndarray: 一个形状为 ``(2**n, 2**n)`` 的 numpy 数组

    代码示例:

    .. code-block:: python
    
        from paddle_quantum.state import isotropic_state
        state = isotropic_state(2, 0.5)
        print(state)

    ::

        [[0.375+0.j 0.   +0.j 0.   +0.j 0.25 +0.j]
        [0.   +0.j 0.125+0.j 0.   +0.j 0.   +0.j]
        [0.   +0.j 0.   +0.j 0.125+0.j 0.   +0.j]
        [0.25 +0.j 0.   +0.j 0.   +0.j 0.375+0.j]]
    """
    assert 0 <= p <= 1, "Probability must be in [0, 1]"

    dim = 2 ** n
    state = p * bell_state(n) + (1 - p) * np.eye(dim) / dim

    return state
