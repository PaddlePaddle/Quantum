# !/usr/bin/env python3
# Copyright (c) 2021 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
qchem 开发所需的线性代数操作
"""

from typing import OrderedDict, Tuple
from collections import OrderedDict
import math
from copy import deepcopy
import paddle


DEFAULT_TOL = 1e-8


def get_givens_rotation_parameters(
        a: paddle.Tensor,
        b: paddle.Tensor
) -> Tuple[paddle.Tensor]:
    r"""计算 Givens 旋转的 (c,s) 参数。
    详细过程参见：https://www.netlib.org/lapack/lawnspdf/lawn148.pdf

    Note:
        :math:`r = \operatorname{sign}(a)\sqrt{a^2+b^2}, c = |a|/|r|, s = \operatorname{sign}(a)*b/|r|`

    Args:
        a (paddle.Tensor(dtype=float64)): 计算所用参数
        b (paddle.Tensor(dtype=float64)): 计算所用参数

    Returns:
        tuple:
            - c (paddle.Tensor(dtype=float64))
            - s (paddle.Tensor(dtype=float64))
    """
    assert a.dtype is paddle.float64 and b.dtype is paddle.float64,\
        "dtype not match, require dtype of a, b be paddle.float64!"

    if math.isclose(b.item(), 0.0, abs_tol=DEFAULT_TOL):
        r = a
        c = paddle.to_tensor(1.0, dtype=paddle.float64)
        s = paddle.to_tensor(0.0, dtype=paddle.float64)
    elif math.isclose(a.item(), 0.0, abs_tol=DEFAULT_TOL):
        r = b.abs()
        c = paddle.to_tensor(0.0, dtype=paddle.float64)
        s = paddle.sign(b)
    else:
        abs_r = paddle.sqrt(a.abs()**2+b.abs()**2)
        r = paddle.sign(a)*abs_r
        c = a.abs()/abs_r
        s = paddle.sign(a)*b/abs_r
    return r, c, s


def parameter_to_givens_matrix1(c: paddle.Tensor, s: paddle.Tensor, j: int, n_size: int):
    r"""该函数可利用 (c,s) 参数来构造 Givens 旋转矩阵以消去 A[i,j] 上的元素。

    Args:
        c (paddle.Tensor): :math:`|A[i,j-1]|/r`, 其中 :math:`r=\sqrt{A[i,j-1]^2+A[i,j]^2}`;
        s (paddle.Tensor): :math:`\operatorname{sign}{(A[i,j]*A[i,j-1])*A[i,j]/r)}`;
        j (int): 计算所用参数
        n_size (int): Givens 旋转矩阵的维度

    Returns:
        Givens 旋转矩阵 (paddle.Tensor)
    """
    G = paddle.eye(n_size, dtype=paddle.float64)

    G[j - 1, j - 1] = c
    G[j, j] = c
    G[j - 1, j] = -s
    G[j, j - 1] = s

    return G


def givens_decomposition(A: paddle.Tensor) -> OrderedDict:
    r"""对于一个给定的矩阵 A，该函数会返回一个 Givens 旋转操作的 list，用户可以使用其中的 Givens 旋转操作消除掉 A 的上三角的元素。

    Note:
        :math:`r = \operatorname{sign}(a)\sqrt{a^2+b^2}, c = |a|/|r|`

        :math:`s = \operatorname{sign}(a)*b/|r| , \theta = arc\cos(c), phi = \operatorname{sign}(a*b)`

        :math:`A^{\prime}[:,j-1] = c*A[:,j-1]+s*A[:,j]`

        :math:`A^{\prime}[:,j] = -s*A[:,j-1]+c*A[:,j]`

    Args:
        A (paddle.Tensor(dtype=paddle.float64)): 矩阵

    Returns:
        OrderedDict, 包含 Givens 旋转操作以及其对应的参数。
    """
    n = A.shape[0]
    assert n == A.shape[1], "The input tensor should be a square matrix."
    assert A.dtype is paddle.float64, "dtype of input tensor must be paddle.float64!"

    # The givens rotations are not parallel !!!
    A1 = deepcopy(A)
    rotations = []
    for i in range(n):
        for j in range(n-1, i, -1):
            _, c, s = get_givens_rotation_parameters(A1[i, j - 1], A1[i, j])
            theta = paddle.acos(c)
            phi = paddle.sign(A1[i, j - 1]*A1[i, j])
            rotations.append((f"{i:>d},{j:>d}", (theta, phi)))

            # update A matrix
            A1_jprev = c*A1[:, j - 1] + s*A1[:, j]
            A1_j = -s*A1[:, j-1] + c*A1[:, j]
            A1[:, j-1] = A1_jprev
            A1[:, j] = A1_j

    return OrderedDict(rotations)


def parameters_to_givens_matrix2(
        theta: paddle.Tensor,
        phi: paddle.Tensor,
        j: int,
        n_size: int
) -> paddle.Tensor:
    r"""该函数用来从 :math:`(\theta,\phi)` 参数中构建 Givens 旋转矩阵消去 :math:`A[i,j]`。

    Args:
        theta (paddle.Tensor): arccos(c);
        phi (paddle.Tensor): :math:`\operatorname{sign}(A[i,j-1]*A[i,j])`;
        j (int): 计算所用参数
        n_size (int): Givens 旋转矩阵的维度。

    Returns:
        paddle.Tensor, Givens 旋转矩阵
    """
    G = paddle.eye(int(n_size), dtype=paddle.float64)

    G[j - 1, j - 1] = paddle.cos(theta)
    G[j, j] = paddle.cos(theta)
    G[j - 1, j] = -phi*paddle.sin(theta)
    G[j, j - 1] = phi*paddle.sin(theta)

    return G
