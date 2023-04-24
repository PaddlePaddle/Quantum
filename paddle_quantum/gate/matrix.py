# !/usr/bin/env python3
# Copyright (c) 2023 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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
The library of gate matrices.
"""

import math
import paddle
from typing import Optional

from ..base import get_dtype
from ..backend.unitary_matrix import unitary_transformation
from ..intrinsic import _zero, _one


def __get_complex_dtype(dtype: paddle.dtype) -> str:
    if dtype == paddle.float32:
        complex_dtype = 'complex64'
    elif dtype == paddle.float64:
        complex_dtype = 'complex128'
    else:
        raise ValueError(
            f"The dtype should be paddle.float32 or paddle.float64: received {dtype}")
    return complex_dtype


# ------------------------------------------------- Split line -------------------------------------------------
r"""
    Belows are single-qubit matrices.
"""


def h_gate(dtype: Optional[str] = None) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        H = \frac{1}{\sqrt{2}}
            \begin{bmatrix}
                1&1\\
                1&-1
            \end{bmatrix}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.
        
    Returns:
        the matrix of H gate.
    
    """
    dtype = get_dtype() if dtype is None else dtype
    element = math.sqrt(2) / 2
    gate_matrix = [
        [element, element],
        [element, -element],
    ]
    return paddle.to_tensor(gate_matrix, dtype=dtype)


def s_gate(dtype: Optional[str] = None) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        S =
            \begin{bmatrix}
                1&0\\
                0&i
            \end{bmatrix}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.
        
    Returns:
        the matrix of S gate.
    
    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0],
        [0, 1j],
    ]
    return paddle.to_tensor(gate_matrix, dtype=dtype)


def sdg_gate(dtype: Optional[str] = None) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        S^\dagger =
            \begin{bmatrix}
                1&0\\
                0&-i
            \end{bmatrix}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.
        
    Returns:
        the matrix of Sdg gate.
    
    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0],
        [0, -1j],
    ]
    return paddle.to_tensor(gate_matrix, dtype=dtype)


def t_gate(dtype: Optional[str] = None) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        T = \begin{bmatrix}
                1&0\\
                0&e^\frac{i\pi}{4}
            \end{bmatrix}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.
        
    Returns:
        the matrix of T gate.
    
    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0],
        [0, math.sqrt(2) / 2 + math.sqrt(2) / 2 * 1j],
    ]
    return paddle.to_tensor(gate_matrix, dtype=dtype)


def tdg_gate(dtype: Optional[str] = None) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        T^\dagger =
            \begin{bmatrix}
                1&0\\
                0&e^{-\frac{i\pi}{4}}
            \end{bmatrix}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.
        
    Returns:
        the matrix of Sdg gate.
    
    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0],
        [0, math.sqrt(2) / 2 - math.sqrt(2) / 2 * 1j],
    ]
    return paddle.to_tensor(gate_matrix, dtype=dtype)


def x_gate(dtype: Optional[str] = None) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        X = \begin{bmatrix}
                0 & 1 \\
                1 & 0
            \end{bmatrix}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.
        
    Returns:
        the matrix of X gate.
    
    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [0, 1],
        [1, 0],
    ]
    return paddle.to_tensor(gate_matrix, dtype=dtype)


def y_gate(dtype: Optional[str] = None) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        Y = \begin{bmatrix}
                0 & -i \\
                i & 0
            \end{bmatrix}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.
        
    Returns:
        the matrix of Y gate.
    
    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [0, -1j],
        [1j, 0],
    ]
    return paddle.to_tensor(gate_matrix, dtype=dtype)


def z_gate(dtype: Optional[str] = None) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        Z = \begin{bmatrix}
                1 & 0 \\
                0 & -1
            \end{bmatrix}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.
        
    Returns:
        the matrix of Z gate.
    
    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0],
        [0, -1],
    ]
    return paddle.to_tensor(gate_matrix, dtype=dtype)


def p_gate(theta: paddle.Tensor) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        P(\theta) = \begin{bmatrix}
                1 & 0 \\
                0 & e^{i\theta}
            \end{bmatrix}

    Args:
        theta: the parameter of this matrix.
        
    Returns:
        the matrix of P gate.
    
    """
    dtype = __get_complex_dtype(theta.dtype)
    gate_matrix = [
        _one(dtype), _zero(dtype),
        _zero(dtype), paddle.cos(theta).cast(dtype) + 1j * paddle.sin(theta).cast(dtype),
    ]
    return paddle.reshape(paddle.concat(gate_matrix), [2, 2])


def rx_gate(theta: paddle.Tensor) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        R_X(\theta) = \begin{bmatrix}
                \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
                -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}

    Args:
        theta: the parameter of this matrix.
        
    Returns:
        the matrix of R_X gate.
    
    """
    dtype = __get_complex_dtype(theta.dtype)
    gate_matrix = [
        paddle.cos(theta / 2).cast(dtype), -1j * paddle.sin(theta / 2).cast(dtype),
        -1j * paddle.sin(theta / 2).cast(dtype), paddle.cos(theta / 2).cast(dtype),
    ]
    return paddle.reshape(paddle.concat(gate_matrix), [2, 2])


def ry_gate(theta: paddle.Tensor) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        R_Y(\theta) = \begin{bmatrix}
                \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
                \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}

    Args:
        theta: the parameter of this matrix.
        
    Returns:
        the matrix of R_Y gate.
    
    """
    dtype = __get_complex_dtype(theta.dtype)
    gate_matrix = [
        paddle.cos(theta / 2).cast(dtype), (-paddle.sin(theta / 2)).cast(dtype),
        paddle.sin(theta / 2).cast(dtype), paddle.cos(theta / 2).cast(dtype),
    ]
    return paddle.reshape(paddle.concat(gate_matrix), [2, 2])


def rz_gate(theta: paddle.Tensor) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        R_Z(\theta) = \begin{bmatrix}
                e^{-i\frac{\theta}{2}} & 0 \\
                0 & e^{i\frac{\theta}{2}}
        \end{bmatrix}

    Args:
        theta: the parameter of this matrix.
        
    Returns:
        the matrix of R_Z gate.
    
    """
    dtype = __get_complex_dtype(theta.dtype)
    gate_matrix = [
        paddle.cos(theta / 2).cast(dtype) - 1j * paddle.sin(theta / 2).cast(dtype), _zero(dtype),
        _zero(dtype), paddle.cos(theta / 2).cast(dtype) + 1j * paddle.sin(theta / 2).cast(dtype),
    ]
    return paddle.reshape(paddle.concat(gate_matrix), [2, 2])


def u3_gate(theta: paddle.Tensor) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            U_3(\theta, \phi, \lambda) =
                \begin{bmatrix}
                    \cos\frac\theta2&-e^{i\lambda}\sin\frac\theta2\\
                    e^{i\phi}\sin\frac\theta2&e^{i(\phi+\lambda)}\cos\frac\theta2
                \end{bmatrix}
        \end{align}

    Args:
        theta: the parameter of this matrix.
        
    Returns:
        the matrix of U_3 gate.
    
    """
    dtype = __get_complex_dtype(theta.dtype)
    gate_real = [
        paddle.cos(theta[0] / 2),
        -paddle.cos(theta[2]) * paddle.sin(theta[0] / 2),
        paddle.cos(theta[1]) * paddle.sin(theta[0] / 2),
        paddle.cos(theta[1] + theta[2]) * paddle.cos(theta[0] / 2),
    ]
    gate_real = paddle.reshape(paddle.concat(gate_real), [2, 2])

    gate_imag = [
        paddle.to_tensor(0, dtype=theta.dtype),
        -paddle.sin(theta[2]) * paddle.sin(theta[0] / 2),
        paddle.sin(theta[1]) * paddle.sin(theta[0] / 2),
        paddle.sin(theta[1] + theta[2]) * paddle.cos(theta[0] / 2),
    ]
    gate_imag = paddle.reshape(paddle.concat(gate_imag), [2, 2])

    return gate_real + 1j * gate_imag


# ------------------------------------------------- Split line -------------------------------------------------
r"""
    Belows are multi-qubit matrices.
"""


def cnot_gate(dtype: Optional[str] = None) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{CNOT} &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes X\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 1 \\
                    0 & 0 & 1 & 0
                \end{bmatrix}
        \end{align}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.
        
    Returns:
        the matrix of CNOT gate.
    
    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ]
    return paddle.to_tensor(gate_matrix, dtype=dtype)


def cy_gate(dtype: Optional[str] = None) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{CY} &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Y\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & -i \\
                    0 & 0 & i & 0
                \end{bmatrix}
        \end{align}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.
        
    Returns:
        the matrix of CY gate.
    
    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, -1j],
        [0, 0, 1j, 0],
    ]
    return paddle.to_tensor(gate_matrix, dtype=dtype)


def cz_gate(dtype: Optional[str] = None) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{CZ} &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Z\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & -1
                \end{bmatrix}
        \end{align}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.
        
    Returns:
        the matrix of CZ gate.
    
    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ]
    return paddle.to_tensor(gate_matrix, dtype=dtype)


def swap_gate(dtype: Optional[str] = None) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{SWAP} =
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 1
                \end{bmatrix}
        \end{align}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.
        
    Returns:
        the matrix of SWAP gate.
    
    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
    return paddle.to_tensor(gate_matrix, dtype=dtype)


def cp_gate(theta: paddle.Tensor) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{CP}(\theta) =
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & e^{i\theta}
                \end{bmatrix}
        \end{align}

    Args:
        theta: the parameter of this matrix.
        
    Returns:
        the matrix of CP gate.
    
    """
    dtype = __get_complex_dtype(theta.dtype)
    gate_matrix = [
        _one(dtype), _zero(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _one(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _zero(dtype), _one(dtype), _zero(dtype),
        _zero(dtype), _zero(dtype), _zero(dtype), paddle.cos(theta).cast(dtype) + 1j * paddle.sin(theta).cast(dtype),
    ]
    return paddle.reshape(paddle.concat(gate_matrix), [4, 4])


def crx_gate(theta: paddle.Tensor) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{CR_X} &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes R_X\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
                    0 & 0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
                \end{bmatrix}
        \end{align}

    Args:
        theta: the parameter of this matrix.
        
    Returns:
        the matrix of CR_X gate.
    
    """
    dtype = __get_complex_dtype(theta.dtype)
    gate_matrix = [
        _one(dtype), _zero(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _one(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _zero(dtype), paddle.cos(theta / 2).cast(dtype), -1j * paddle.sin(theta / 2).cast(dtype),
        _zero(dtype), _zero(dtype), -1j * paddle.sin(theta / 2).cast(dtype), paddle.cos(theta / 2).cast(dtype),
    ]
    return paddle.reshape(paddle.concat(gate_matrix), [4, 4])


def cry_gate(theta: paddle.Tensor) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{CR_Y} &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes R_Y\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
                    0 & 0 & \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
                \end{bmatrix}
        \end{align}

    Args:
        theta: the parameter of this matrix.
        
    Returns:
        the matrix of CR_Y gate.
    
    """
    dtype = __get_complex_dtype(theta.dtype)
    gate_matrix = [
        _one(dtype), _zero(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _one(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _zero(dtype), paddle.cos(theta / 2).cast(dtype), (-paddle.sin(theta / 2)).cast(dtype),
        _zero(dtype), _zero(dtype), paddle.sin(theta / 2).cast(dtype), paddle.cos(theta / 2).cast(dtype),
    ]
    return paddle.reshape(paddle.concat(gate_matrix), [4, 4])


def crz_gate(theta: paddle.Tensor) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{CR_Z} &= |0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes R_Z\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & e^{-i\frac{\theta}{2}} & 0 \\
                    0 & 0 & 0 & e^{i\frac{\theta}{2}}
                \end{bmatrix}
        \end{align}

    Args:
        theta: the parameter of this matrix.
        
    Returns:
        the matrix of CR_Z gate.
    
    """
    dtype = __get_complex_dtype(theta.dtype)
    param1 = paddle.cos(theta / 2).cast(dtype) - 1j * paddle.sin(theta / 2).cast(dtype)
    param2 = paddle.cos(theta / 2).cast(dtype) + 1j * paddle.sin(theta / 2).cast(dtype)
    gate_matrix = [
        _one(dtype), _zero(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _one(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _zero(dtype), param1, _zero(dtype),
        _zero(dtype), _zero(dtype), _zero(dtype), param2,
    ]
    return paddle.reshape(paddle.concat(gate_matrix), [4, 4])


def cu_gate(theta: paddle.Tensor) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{CU}
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & \cos\frac\theta2 &-e^{i\lambda}\sin\frac\theta2 \\
                    0 & 0 & e^{i\phi}\sin\frac\theta2&e^{i(\phi+\lambda)}\cos\frac\theta2
                \end{bmatrix}
        \end{align}

    Args:
        theta: the parameter of this matrix.
        
    Returns:
        the matrix of CU gate.
    
    """
    dtype = __get_complex_dtype(theta.dtype)
    param1 = paddle.cos(theta[0] / 2).cast(dtype)
    param2 = (paddle.cos(theta[2]).cast(dtype) + 1j * paddle.sin(theta[2]).cast(dtype)) * \
             (-paddle.sin(theta[0] / 2)).cast(dtype)
    param3 = (paddle.cos(theta[1]).cast(dtype) + 1j * paddle.sin(theta[1]).cast(dtype)) * \
        paddle.sin(theta[0] / 2).cast(dtype)
    param4 = (paddle.cos(theta[1] + theta[2]).cast(dtype) + 1j * paddle.sin(theta[1] + theta[2]).cast(dtype)) * \
        paddle.cos(theta[0] / 2).cast(dtype)
    gate_matrix = [
        _one(dtype), _zero(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _one(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _zero(dtype), param1, param2,
        _zero(dtype), _zero(dtype), param3, param4,
    ]
    return paddle.reshape(paddle.concat(gate_matrix), [4, 4])

def rxx_gate(theta: paddle.Tensor) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{R_{XX}}(\theta) =
                    \begin{bmatrix}
                        \cos\frac{\theta}{2} & 0 & 0 & -i\sin\frac{\theta}{2} \\
                        0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\
                        0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
                        -i\sin\frac{\theta}{2} & 0 & 0 & \cos\frac{\theta}{2}
                    \end{bmatrix}
        \end{align}

    Args:
        theta: the parameter of this matrix.
        
    Returns:
        the matrix of RXX gate.
    
    """
    dtype = __get_complex_dtype(theta.dtype)
    param1 = paddle.cos(theta / 2).cast(dtype)
    param2 = -1j * paddle.sin(theta / 2).cast(dtype)
    gate_matrix = [
        param1, _zero(dtype), _zero(dtype), param2,
        _zero(dtype), param1, param2, _zero(dtype),
        _zero(dtype), param2, param1, _zero(dtype),
        param2, _zero(dtype), _zero(dtype), param1,
    ]
    return paddle.reshape(paddle.concat(gate_matrix), [4, 4])


def ryy_gate(theta: paddle.Tensor) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{R_{YY}}(\theta) =
                    \begin{bmatrix}
                        \cos\frac{\theta}{2} & 0 & 0 & i\sin\frac{\theta}{2} \\
                        0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\
                        0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
                        i\sin\frac{\theta}{2} & 0 & 0 & cos\frac{\theta}{2}
                    \end{bmatrix}
        \end{align}

    Args:
        theta: the parameter of this matrix.
        
    Returns:
        the matrix of RYY gate.
    
    """
    dtype = __get_complex_dtype(theta.dtype)
    param1 = paddle.cos(theta / 2).cast(dtype)
    param2 = -1j * paddle.sin(theta / 2).cast(dtype)
    param3 = 1j * paddle.sin(theta / 2).cast(dtype)
    gate_matrix = [
        param1, _zero(dtype), _zero(dtype), param3,
        _zero(dtype), param1, param2, _zero(dtype),
        _zero(dtype), param2, param1, _zero(dtype),
        param3, _zero(dtype), _zero(dtype), param1,
    ]
    return paddle.reshape(paddle.concat(gate_matrix), [4, 4])


def rzz_gate(theta: paddle.Tensor) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{R_{ZZ}}(\theta) =
                    \begin{bmatrix}
                        e^{-i\frac{\theta}{2}} & 0 & 0 & 0 \\
                        0 & e^{i\frac{\theta}{2}} & 0 & 0 \\
                        0 & 0 & e^{i\frac{\theta}{2}} & 0 \\
                        0 & 0 & 0 & e^{-i\frac{\theta}{2}}
                    \end{bmatrix}
        \end{align}

    Args:
        theta: the parameter of this matrix.
        
    Returns:
        the matrix of RZZ gate.
    
    """
    dtype = __get_complex_dtype(theta.dtype)
    param1 = paddle.cos(theta / 2).cast(dtype) - 1j * paddle.sin(theta / 2).cast(dtype)
    param2 = paddle.cos(theta / 2).cast(dtype) + 1j * paddle.sin(theta / 2).cast(dtype)
    gate_matrix = [
        param1, _zero(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), param2, _zero(dtype), _zero(dtype),
        _zero(dtype), _zero(dtype), param2, _zero(dtype),
        _zero(dtype), _zero(dtype), _zero(dtype), param1,
    ]
    return paddle.reshape(paddle.concat(gate_matrix), [4, 4])


def ms_gate(dtype: Optional[str] = None) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{MS} = \mathit{R_{XX}}(-\frac{\pi}{2}) = \frac{1}{\sqrt{2}}
                    \begin{bmatrix}
                        1 & 0 & 0 & i \\
                        0 & 1 & i & 0 \\
                        0 & i & 1 & 0 \\
                        i & 0 & 0 & 1
                    \end{bmatrix}
        \end{align}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.
        
    Returns:
        the matrix of MS gate.
    
    """
    dtype = get_dtype() if dtype is None else dtype
    val1 = math.sqrt(2) / 2
    val2 = 1j / math.sqrt(2)
    gate_matrix = [
        [val1, 0, 0, val2],
        [0, val1, val2, 0],
        [0, val2, val1, 0],
        [val2, 0, 0, val1],
    ]
    return paddle.to_tensor(gate_matrix, dtype=dtype)


def cswap_gate(dtype: Optional[str] = None) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{CSWAP} =
                \begin{bmatrix}
                    1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
                \end{bmatrix}
        \end{align}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.
        
    Returns:
        the matrix of CSWAP gate.
    
    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
    return paddle.to_tensor(gate_matrix, dtype=dtype)


def toffoli_gate(dtype: Optional[str] = None) -> paddle.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{CSWAP} =
                \begin{bmatrix}
                    1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
                \end{bmatrix}
        \end{align}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.
        
    Returns:
        the matrix of Toffoli gate.
    
    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ]
    return paddle.to_tensor(gate_matrix, dtype=dtype)


def universal2_gate(theta: paddle.Tensor) -> paddle.Tensor:
    r"""Generate the matrix

    Args:
        theta: the parameter of this matrix.
        
    Returns:
        the matrix of universal two qubits gate.
    
    """
    dtype = __get_complex_dtype(theta.dtype)
    unitary = paddle.eye(2 ** 2).cast(dtype)
    _cnot_gate = cnot_gate(dtype)
    
    unitary = unitary_transformation(unitary, u3_gate(theta[[0, 1, 2]]), qubit_idx=0, num_qubits=2)
    unitary = unitary_transformation(unitary, u3_gate(theta[[3, 4, 5]]), qubit_idx=1, num_qubits=2)
    unitary = unitary_transformation(unitary, _cnot_gate, qubit_idx=[1, 0], num_qubits=2)
    
    unitary = unitary_transformation(unitary, rz_gate(theta[[6]]), qubit_idx=0, num_qubits=2)
    unitary = unitary_transformation(unitary, ry_gate(theta[[7]]), qubit_idx=1, num_qubits=2)
    unitary = unitary_transformation(unitary, _cnot_gate, qubit_idx=[0, 1], num_qubits=2)
    
    unitary = unitary_transformation(unitary, ry_gate(theta[[8]]), qubit_idx=1, num_qubits=2)
    unitary = unitary_transformation(unitary, _cnot_gate, qubit_idx=[1, 0], num_qubits=2)
    
    unitary = unitary_transformation(unitary, u3_gate(theta[[9, 10, 11]]), qubit_idx=0, num_qubits=2)
    unitary = unitary_transformation(unitary, u3_gate(theta[[12, 13, 14]]), qubit_idx=1, num_qubits=2)
    
    return unitary


def universal3_gate(theta: paddle.Tensor) -> paddle.Tensor:
    r"""Generate the matrix

    Args:
        theta: the parameter of this matrix.
        
    Returns:
        the matrix of universal three qubits gate.
    
    """
    dtype = __get_complex_dtype(theta.dtype)
    unitary = paddle.eye(2 ** 3).cast(dtype)
    _h_gate, _s_gate, _cnot_gate = h_gate(dtype), s_gate(dtype), cnot_gate(dtype)
    
    psi = paddle.reshape(theta[:60], shape=[4, 15])
    phi = paddle.reshape(theta[60:], shape=[7, 3])
    
    def __block_u(_unitary, _theta):
        _unitary = unitary_transformation(_unitary, _cnot_gate, qubit_idx=[1, 2], num_qubits=3)
        _unitary = unitary_transformation(_unitary, ry_gate(_theta[0]), qubit_idx=1, num_qubits=3)
        _unitary = unitary_transformation(_unitary, _cnot_gate, qubit_idx=[0, 1], num_qubits=3)
        _unitary = unitary_transformation(_unitary, ry_gate(_theta[1]), qubit_idx=1, num_qubits=3)
        _unitary = unitary_transformation(_unitary, _cnot_gate, qubit_idx=[0, 1], num_qubits=3)
        
        _unitary = unitary_transformation(_unitary, _cnot_gate, qubit_idx=[1, 2], num_qubits=3)
        _unitary = unitary_transformation(_unitary, _h_gate, qubit_idx=2, num_qubits=3)
        _unitary = unitary_transformation(_unitary, _cnot_gate, qubit_idx=[1, 0], num_qubits=3)
        
        _unitary = unitary_transformation(_unitary, _cnot_gate, qubit_idx=[0, 2], num_qubits=3)
        _unitary = unitary_transformation(_unitary, _cnot_gate, qubit_idx=[1, 2], num_qubits=3)
        _unitary = unitary_transformation(_unitary, rz_gate(_theta[2]), qubit_idx=2, num_qubits=3)
        _unitary = unitary_transformation(_unitary, _cnot_gate, qubit_idx=[1, 2], num_qubits=3)
        _unitary = unitary_transformation(_unitary, _cnot_gate, qubit_idx=[0, 2], num_qubits=3)
        return _unitary
    
    def __block_v(_unitary, _theta):
        _unitary = unitary_transformation(_unitary, _cnot_gate, qubit_idx=[2, 0], num_qubits=3)
        _unitary = unitary_transformation(_unitary, _cnot_gate, qubit_idx=[1, 2], num_qubits=3)
        _unitary = unitary_transformation(_unitary, _cnot_gate, qubit_idx=[2, 1], num_qubits=3)
        
        _unitary = unitary_transformation(_unitary, ry_gate(_theta[0]), qubit_idx=2, num_qubits=3)
        _unitary = unitary_transformation(_unitary, _cnot_gate, qubit_idx=[1, 2], num_qubits=3)
        _unitary = unitary_transformation(_unitary, ry_gate(_theta[1]), qubit_idx=2, num_qubits=3)
        _unitary = unitary_transformation(_unitary, _cnot_gate, qubit_idx=[1, 2], num_qubits=3)
        
        _unitary = unitary_transformation(_unitary, _s_gate, qubit_idx=2, num_qubits=3)
        _unitary = unitary_transformation(_unitary, _cnot_gate, qubit_idx=[2, 0], num_qubits=3)
        _unitary = unitary_transformation(_unitary, _cnot_gate, qubit_idx=[0, 1], num_qubits=3)
        _unitary = unitary_transformation(_unitary, _cnot_gate, qubit_idx=[1, 0], num_qubits=3)
        
        _unitary = unitary_transformation(_unitary, _h_gate, qubit_idx=2, num_qubits=3)
        _unitary = unitary_transformation(_unitary, _cnot_gate, qubit_idx=[0, 2], num_qubits=3)
        _unitary = unitary_transformation(_unitary, rz_gate(_theta[2]), qubit_idx=2, num_qubits=3)
        _unitary = unitary_transformation(_unitary, _cnot_gate, qubit_idx=[0, 2], num_qubits=3)
        return _unitary
    
    unitary = unitary_transformation(unitary, universal2_gate(psi[0]), qubit_idx=[0, 1], num_qubits=3)
    unitary = unitary_transformation(unitary, u3_gate(phi[0, 0:3]), qubit_idx=2, num_qubits=3)
    unitary = __block_u(unitary, phi[1])
    
    unitary = unitary_transformation(unitary, universal2_gate(psi[1]), qubit_idx=[0, 1], num_qubits=3)
    unitary = unitary_transformation(unitary, u3_gate(phi[2, 0:3]), qubit_idx=2, num_qubits=3)
    unitary = __block_v(unitary, phi[3])
    
    unitary = unitary_transformation(unitary, universal2_gate(psi[2]), qubit_idx=[0, 1], num_qubits=3)
    unitary = unitary_transformation(unitary, u3_gate(phi[4, 0:3]), qubit_idx=2, num_qubits=3)
    unitary = __block_u(unitary, phi[5])
    
    unitary = unitary_transformation(unitary, universal2_gate(psi[3]), qubit_idx=[0, 1], num_qubits=3)
    unitary = unitary_transformation(unitary, u3_gate(phi[6, 0:3]), qubit_idx=2, num_qubits=3)
    return unitary
