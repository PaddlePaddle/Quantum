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

import numpy 
from numpy import concatenate
from numpy import trace as np_trace
from numpy import matmul as np_matmul
from numpy import random as np_random
from numpy import zeros as np_zeros
from numpy import eye as np_eye

__all__ = [
    "vec",
    "vec_random",
    "w_state",
    "density_op",
    "density_op_random",
    "completely_mixed_computational"
]


def vec(n):
    r"""生成量子态 :math:`|00...0\rangle` 的numpy形式。

    Args:
        n(int): 量子比特数量

    Returns:
        numpy.ndarray: 一个形状为 ``(1, 2**n)`` 的numpy数组 ``[[1, 0, 0, ..., 0]]``

    代码示例:

    .. code-block:: python
    
        from paddle_quantum.state import vec
        vector = vec(3)
        print(vector)

    ::

        [[1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]]
    """
    assert n > 0, 'qubit number must be larger than 1'
    state = concatenate(([[1.0]], np_zeros([1, 2**n- 1])), axis=1)
    return state.astype("complex128")


def vec_random(n, real_or_complex=2):
    r"""随机生成一个量子态的numpy形式。

    Args:
        n (int): 量子比特数量
        real_or_complex (int, optional): 默认为2，即生成复数组；若为1，则生成实数组

    Returns:
        numpy.ndarray: 一个形状为 ``(1, 2**n)`` 的numpy数组
    """
    assert n > 0, 'qubit number must be larger than 1'
    assert real_or_complex == 1 or real_or_complex == 2, 'real_or_complex must be 1 or 2'
    # real
    if real_or_complex == 1:
        psi = np_random.randn(1, 2**n)
    # complex
    else:
        psi = np_random.randn(1, 2**n) + 1j * np_random.randn(1, 2**n)
    psi = psi/numpy.linalg.norm(psi)

    return psi.astype("complex128")


def w_state(n, coeff=None):
    r"""生成一个W-state的numpy形式。

    Args:
        n (int): 量子比特数量
        coeff (numpy.ndarray, optional): 默认为 ``None`` ，即生成平均概率幅（系数）

    Returns:
        numpy.ndarray: 一个形状为 ``(1, 2**n)`` 的numpy数组

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
    
    c = coeff if coeff is not None else numpy.ones((1, 2**n))/numpy.sqrt(n)
    assert c.shape[0] == 1 and c.shape[1] == 2**n, 'The dimension of coeff is not right'

    state = np_zeros((1, 2**n))
    for i in range(n):
        state[0][2**i] = c[0][n-i-1]
        
    return state.astype("complex128")


def density_op(n):
    r"""生成密度矩阵 :math:`|00..0\rangle \langle00..0|` 的numpy形式。

    Args:
        n (int): 量子比特数量

    Returns:
        numpy.ndarray: 一个形状为 ``(2**n, 2**n)`` 的numpy数组

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
    rho = np_zeros((2**n, 2**n))
    rho[0, 0] = 1

    return rho.astype("complex128")


def density_op_random(n, real_or_complex=2, rank=None):
    r"""随机生成一个密度矩阵的numpy形式。

    Args:
        n (int): 量子比特数量
        real_or_complex (int, optional): 默认为2，即生成复数组，若为1，则生成实数组
        rank (int, optional): 矩阵的秩，默认为 :math:`2^n` （当 ``rank`` 为 ``None`` 时）

    Returns:
        numpy.ndarray: 一个形状为 ``(2**n, 2**n)`` 的numpy数组
    """
    assert n > 0, 'qubit number must be positive'
    rank = rank if rank is not None else 2**n
    assert 0 < rank <= 2**n, 'rank is an invalid number'

    if real_or_complex == 1:
        psi = np_random.randn(2**n, rank)
    else:
        psi = np_random.randn(2**n, rank) + 1j*np_random.randn(2**n, rank)

    psi_dagger = psi.conj().T
    rho = np_matmul(psi, psi_dagger)
    rho = rho/np_trace(rho)
    
    return rho.astype('complex128')


def completely_mixed_computational(n):
    r"""生成完全混合态的密度矩阵的numpy形式。

    Args:
        n (int): 量子比特数量

    Returns:
        numpy.ndarray: 一个形状为 ``(2**n, 2**n)`` 的numpy数组

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
    rho = np_eye(2**n)/2**n
    
    return rho.astype('complex128')
