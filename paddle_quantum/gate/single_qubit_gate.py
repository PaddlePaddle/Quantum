# !/usr/bin/env python3
# Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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
The source file of the classes for single-qubit gates.
"""

from typing import Optional, Union, Iterable

import paddle
from .base import Gate, ParamGate
from .matrix import (
    h_gate, s_gate, sdg_gate, t_gate, tdg_gate,
    x_gate, y_gate, z_gate, p_gate,
    rx_gate, ry_gate, rz_gate, u3_gate,
)
from ..backend import Backend
from ..intrinsic import _format_qubits_idx, _get_float_dtype


class H(Gate):
    r"""A collection of single-qubit Hadamard gates.

    The matrix form of such a gate is:

    .. math::

        H = \frac{1}{\sqrt{2}}
            \begin{bmatrix}
                1&1\\
                1&-1
            \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    __matrix = h_gate('complex128')

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1
    ):
        gate_info = {
            'gatename': 'h',
            'texname': r'$H$',
            'plot_width': 0.4,
        }
        super().__init__(
            None, qubits_idx, depth, gate_info, num_qubits, check_legality=False, num_acted_qubits=1)

    @property
    def matrix(self) -> paddle.Tensor:
        if self.dtype == 'complex64':
            return H.__matrix.cast('complex64')
        return H.__matrix


class S(Gate):
    r"""A collection of single-qubit S gates.

    The matrix form of such a gate is:

    .. math::

        S =
            \begin{bmatrix}
                1&0\\
                0&i
            \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    __matrix = s_gate('complex128')

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1
    ):
        gate_info = {
            'gatename': 's',
            'texname': r'$S$',
            'plot_width': 0.4,
        }
        super().__init__(
            None, qubits_idx, depth, gate_info, num_qubits, check_legality=False, num_acted_qubits=1)

    @property
    def matrix(self) -> paddle.Tensor:
        if self.dtype == 'complex64':
            return S.__matrix.cast('complex64')
        return S.__matrix


class Sdg(Gate):
    r"""A collection of single-qubit S dagger (S inverse) gates.

    The matrix form of such a gate is:

    .. math::

        S^\dagger =
            \begin{bmatrix}
                1&0\\
                0&-i
            \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    __matrix = sdg_gate('complex128')

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1
    ):
        gate_info = {
            'gatename': 'sdg',
            'texname': r'$S^\dagger$',
            'plot_width': 0.4,
        }
        super().__init__(
            None, qubits_idx, depth, gate_info, num_qubits, check_legality=False, num_acted_qubits=1)

    @property
    def matrix(self) -> paddle.Tensor:
        if self.dtype == 'complex64':
            return Sdg.__matrix.cast('complex64')
        return Sdg.__matrix


class T(Gate):
    r"""A collection of single-qubit T gates.

    The matrix form of such a gate is:

    .. math::

        T =
            \begin{bmatrix}
                1&0\\
                0&e^\frac{i\pi}{4}
            \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    __matrix = t_gate('complex128')

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1
    ):
        gate_info = {
            'gatename': 't',
            'texname': r'$T$',
            'plot_width': 0.4,
        }
        super().__init__(
            None, qubits_idx, depth, gate_info, num_qubits, check_legality=False, num_acted_qubits=1)

    @property
    def matrix(self) -> paddle.Tensor:
        if self.dtype == 'complex64':
            return T.__matrix.cast('complex64')
        return T.__matrix


class Tdg(Gate):
    r"""A collection of single-qubit T dagger (T inverse) gates.

    The matrix form of such a gate is:

    .. math::

        T^\dagger =
            \begin{bmatrix}
                1&0\\
                0&e^{-\frac{i\pi}{4}}
            \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    __matrix = tdg_gate('complex128')

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1
    ):
        gate_info = {
            'gatename': 'tdg',
            'texname': r'$T^\dagger$',
            'plot_width': 0.4,
        }
        super().__init__(
            None, qubits_idx, depth, gate_info, num_qubits, check_legality=False, num_acted_qubits=1)

    @property
    def matrix(self) -> paddle.Tensor:
        if self.dtype == 'complex64':
            return Tdg.__matrix.cast('complex64')
        return Tdg.__matrix


class X(Gate):
    r"""A collection of single-qubit X gates.

    The matrix form of such a gate is:

    .. math::

        \begin{bmatrix}
            0 & 1 \\
            1 & 0
        \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """

    __matrix = x_gate('complex128')

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1
    ):
        gate_info = {
            'gatename': 'x',
            'texname': r'$X$',
            'plot_width': 0.4,
        }
        super().__init__(
            None, qubits_idx, depth, gate_info, num_qubits, check_legality=False, num_acted_qubits=1)

    @property
    def matrix(self) -> paddle.Tensor:
        if self.dtype == 'complex64':
            return X.__matrix.cast('complex64')
        return X.__matrix


class Y(Gate):
    r"""A collection of single-qubit Y gates.

    The matrix form of such a gate is:

    .. math::

        \begin{bmatrix}
            0 & -i \\
            i & 0
        \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    __matrix = y_gate('complex128')

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1
    ):
        gate_info = {
            'gatename': 'y',
            'texname': r'$Y$',
            'plot_width': 0.4,
        }
        super().__init__(
            None, qubits_idx, depth, gate_info, num_qubits, check_legality=False, num_acted_qubits=1)

    @property
    def matrix(self) -> paddle.Tensor:
        if self.dtype == 'complex64':
            return Y.__matrix.cast('complex64')
        return Y.__matrix


class Z(Gate):
    r"""A collection of single-qubit Z gates.

    The matrix form of such a gate is:

    .. math::

        \begin{bmatrix}
            1 & 0 \\
            0 & -1
        \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    __matrix = z_gate('complex128')

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1
    ):
        gate_info = {
            'gatename': 'z',
            'texname': r'$Z$',
            'plot_width': 0.4,
        }
        super().__init__(
            None, qubits_idx, depth, gate_info, num_qubits, check_legality=False, num_acted_qubits=1)

    @property
    def matrix(self) -> paddle.Tensor:
        if self.dtype == 'complex64':
            return Z.__matrix.cast('complex64')
        return Z.__matrix


class P(ParamGate):
    r"""A collection of single-qubit P gates.

    The matrix form of such a gate is:

    .. math::

        \begin{bmatrix}
            1 & 0 \\
            0 & e^{i\theta}
        \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``paddle.Tensor`` or ``float``.
    """
    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            'gatename': 'p',
            'texname': r'$P$',
            'plot_width': 0.9,
        }

        super().__init__(
            p_gate, param, depth, 1, param_sharing, qubits_idx, gate_info, num_qubits, False, num_acted_qubits=1)


class RX(ParamGate):
    r"""A collection of single-qubit rotation gates about the x-axis.

    The matrix form of such a gate is:

    .. math::

        \begin{bmatrix}
            \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
            -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
        \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``paddle.Tensor`` or ``float``.
    """
    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            'gatename': 'rx',
            'texname': r'$R_{x}$',
            'plot_width': 0.9,
        }

        super().__init__(
            rx_gate, param, depth, 1, param_sharing, qubits_idx, gate_info, num_qubits, False, num_acted_qubits=1)


class RY(ParamGate):
    r"""A collection of single-qubit rotation gates about the y-axis.

    The matrix form of such a gate is:

    .. math::

        \begin{bmatrix}
            \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
            \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
        \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``paddle.Tensor`` or ``float``.
    """
    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            'gatename': 'ry',
            'texname': r'$R_{y}$',
            'plot_width': 0.9,
        }
        super().__init__(
            ry_gate, param, depth, 1, param_sharing, qubits_idx, gate_info, num_qubits, False, num_acted_qubits=1)


class RZ(ParamGate):
    r"""A collection of single-qubit rotation gates about the z-axis.

    The matrix form of such a gate is:

    .. math::

        \begin{bmatrix}
            e^{-i\frac{\theta}{2}} & 0 \\
            0 & e^{i\frac{\theta}{2}}
        \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``paddle.Tensor`` or ``float``.
    """
    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            'gatename': 'rz',
            'texname': r'$R_{z}$',
            'plot_width': 0.9,
        }

        super().__init__(
            rz_gate, param, depth, 1, param_sharing, qubits_idx, gate_info, num_qubits, False, num_acted_qubits=1)


class U3(ParamGate):
    r"""A collection of single-qubit rotation gates.

    The matrix form of such a gate is:

    .. math::

        \begin{align}
            U3(\theta, \phi, \lambda) =
                \begin{bmatrix}
                    \cos\frac\theta2&-e^{i\lambda}\sin\frac\theta2\\
                    e^{i\phi}\sin\frac\theta2&e^{i(\phi+\lambda)}\cos\frac\theta2
                \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``paddle.Tensor`` or ``float``.
    """
    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, Iterable[float]]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            'gatename': 'u',
            'texname': r'$U$',
            'plot_width': 1.65,
        }

        super().__init__(
            u3_gate, param, depth, 3, param_sharing, qubits_idx, gate_info, num_qubits, False, num_acted_qubits=1)
