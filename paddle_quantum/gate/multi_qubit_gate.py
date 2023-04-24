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
The source file of the classes for multi-qubit gates.
"""

import matplotlib
from typing import Optional, Union, Iterable, List, Tuple

import numpy as np
import math
import paddle_quantum as pq
import paddle
from .base import Gate, ParamGate
from .functional.visual import (
    _cnot_display, _cswap_display, _tofolli_display, _swap_display,
    _cx_like_display, _crx_like_display, _oracle_like_display, _rxx_like_display
)
from .matrix import (
    cnot_gate,
    cy_gate,
    cz_gate,
    swap_gate,
    cp_gate,
    crx_gate,
    cry_gate,
    crz_gate,
    cu_gate,
    rxx_gate,
    ryy_gate,
    rzz_gate,
    ms_gate,
    cswap_gate,
    toffoli_gate,
    universal2_gate,
    universal3_gate
)
from ..backend import Backend
from ..intrinsic import _cnot_idx_fetch, _inverse_gather_for_dm, _paddle_gather


class CNOT(Gate):
    r"""A collection of CNOT gates.

    For a 2-qubit quantum circuit, when ``qubits_idx`` is ``[0, 1]``, the matrix form of such a gate is:

    .. math::

        \begin{align}
            CNOT &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes X\\
            &=
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1 \\
                0 & 0 & 1 & 0
            \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
        cnot_idx: CNOT gate index. Defaults to ``None``.
    """

    __matrix = cnot_gate('complex128')

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            cnot_idx: Optional[List[int]] = None
    ):
        gate_info = {
            'gatename': 'cnot',
            'texname': r'$CNOT$',
            'plot_width': 0.2,
        }
        super().__init__(
            None, qubits_idx, depth, gate_info, num_qubits, check_legality=False, num_acted_qubits=2)

        self.cnot_idx = cnot_idx

    @property
    def matrix(self) -> paddle.Tensor:
        if self.dtype == 'complex64':
            return CNOT.__matrix.cast('complex64')
        return CNOT.__matrix

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _cnot_display(self, ax, x, )

    def forward(self, state: pq.State) -> pq.State:
        if state.backend == Backend.QuLeaf or state.backend == Backend.UnitaryMatrix:
            return super().forward(state=state)

        if self.cnot_idx is None:
            self.cnot_idx = _cnot_idx_fetch(num_qubits=state.num_qubits, qubits_idx=self.qubits_idx)

        if state.is_swap_back:
            state = state.clone()

        state.reset_sequence()
        data = state.data

        num_qubits = int(math.log2(data.shape[-1]))

        if state.backend == Backend.StateVector:
            for _ in range(self.depth):
                # whether to use batch in state_vector backend. len(state.shape) equals 1 means not using batch
                if len(data.shape) == 1:
                    data = _paddle_gather(data, index=paddle.to_tensor(self.cnot_idx))
                else:
                    data = paddle.reshape(data, [-1, 2 ** num_qubits]).T
                    data = _paddle_gather(data, index=paddle.to_tensor(self.cnot_idx)).T

        elif state.backend == Backend.DensityMatrix:
            for _ in range(self.depth):
                # left swap
                # whether to use batch in density_matrix backend. len(state.shape) is greater than 2 means using batch
                if len(data.shape) > 2:
                    data = paddle.reshape(data, [-1, 2 ** num_qubits, 2 ** num_qubits])
                    data = paddle.transpose(data, perm=[1, 2, 0])

                data = _paddle_gather(data, index=paddle.to_tensor(self.cnot_idx))

                # right swap
                data = _inverse_gather_for_dm(data, base_idx=paddle.to_tensor(self.cnot_idx))

        state.data = data
        return state


CX = CNOT


class CY(Gate):
    r"""A collection of controlled Y gates.

    For a 2-qubit quantum circuit, when ``qubits_idx`` is ``[0, 1]``, the matrix form of such a gate is:

    .. math::

        \begin{align}
            CY &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Y\\
            &=
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & -1j \\
                0 & 0 & 1j & 0
            \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """

    __matrix = cy_gate('complex128')

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1
    ):
        gate_info = {
            'gatename': 'cy',
            'texname': r'$Y$',
            'plot_width': 0.4,
        }
        super().__init__(
            None, qubits_idx, depth, gate_info, num_qubits, check_legality=False, num_acted_qubits=2)

    @property
    def matrix(self) -> paddle.Tensor:
        if self.dtype == 'complex64':
            return CY.__matrix.cast('complex64')
        return CY.__matrix

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _cx_like_display(self, ax, x, )


class CZ(Gate):
    r"""A collection of controlled Z gates.

    For a 2-qubit quantum circuit, when ``qubits_idx`` is ``[0, 1]``, the matrix form of such a gate is:

    .. math::

        \begin{align}
            CZ &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Z\\
            &=
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & -1
            \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """

    __matrix = cz_gate('complex128')

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1
    ):
        gate_info = {
            'gatename': 'cz',
            'texname': r'$Z$',
            'plot_width': 0.4,
        }
        super().__init__(
            None, qubits_idx, depth, gate_info, num_qubits, check_legality=False, num_acted_qubits=2)

    @property
    def matrix(self) -> paddle.Tensor:
        if self.dtype == 'complex64':
            return CZ.__matrix.cast('complex64')
        return CZ.__matrix

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _cx_like_display(self, ax, x)


class SWAP(Gate):
    r"""A collection of SWAP gates.

    The matrix form of such a gate is:

    .. math::

        \begin{align}
            SWAP =
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1
            \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """

    __matrix = swap_gate('complex128')

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1
    ):
        gate_info = {
            'gatename': 'swap',
            'texname': r'$SWAP$',
            'plot_width': 0.2,
        }
        super().__init__(
            None, qubits_idx, depth, gate_info, num_qubits, check_legality=False, num_acted_qubits=2)

    @property
    def matrix(self) -> paddle.Tensor:
        if self.dtype == 'complex64':
            return SWAP.__matrix.cast('complex64')
        return SWAP.__matrix

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _swap_display(self, ax, x, )


class CP(ParamGate):
    r"""A collection of controlled P gates.

    For a 2-qubit quantum circuit, when ``qubits_idx`` is ``[0, 1]``, the matrix form of such a gate is:

    .. math::

        \begin{bmatrix}
            1 & 0 & 0 & 0\\
            0 & 1 & 0 & 0\\
            0 & 0 & 1 & 0\\
            0 & 0 & 0 & e^{i\theta}
        \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``paddle.Tensor`` or ``float``.
    """

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            'gatename': 'cp',
            'texname': r'$P$',
            'plot_width': 0.9,
        }

        super().__init__(
            cp_gate, param, depth, 1, param_sharing, qubits_idx, gate_info, num_qubits, False, num_acted_qubits=2)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _crx_like_display(self, ax, x)


class CRX(ParamGate):
    r"""A collection of controlled rotation gates about the x-axis.

    For a 2-qubit quantum circuit, when ``qubits_idx`` is ``[0, 1]``, the matrix form of such a gate is:

    .. math::

        \begin{align}
            CRx &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Rx\\
            &=
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
                0 & 0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``paddle.Tensor`` or ``float``.
    """

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            'gatename': 'crx',
            'texname': r'$R_{x}$',
            'plot_width': 0.9,
        }

        super().__init__(
            crx_gate, param, depth, 1, param_sharing, qubits_idx, gate_info, num_qubits, False, num_acted_qubits=2)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _crx_like_display(self, ax, x)


class CRY(ParamGate):
    r"""A collection of controlled rotation gates about the y-axis.

    For a 2-qubit quantum circuit, when ``qubits_idx`` is ``[0, 1]``, the matrix form of such a gate is:

    .. math::

        \begin{align}
            CRy &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Ry\\
            &=
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
                0 & 0 & \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``paddle.Tensor`` or ``float``.
    """

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            'gatename': 'cry',
            'texname': r'$R_{y}$',
            'plot_width': 0.9,
        }

        super().__init__(
            cry_gate, param, depth, 1, param_sharing, qubits_idx, gate_info, num_qubits, False, num_acted_qubits=2)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _crx_like_display(self, ax, x, )


class CRZ(ParamGate):
    r"""A collection of controlled rotation gates about the z-axis.

    For a 2-qubit quantum circuit, when ``qubits_idx`` is ``[0, 1]``, the matrix form of such a gate is:

    .. math::

        \begin{align}
            CRz &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Rz\\
            &=
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & e^{-i\frac{\theta}{2}} & 0 \\
                0 & 0 & 0 & e^{i\frac{\theta}{2}}
            \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``paddle.Tensor`` or ``float``.
    """

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            'gatename': 'crz',
            'texname': r'$R_{z}$',
            'plot_width': 0.9,
        }

        super().__init__(
            crz_gate, param, depth, 1, param_sharing, qubits_idx, gate_info, num_qubits, False, num_acted_qubits=2)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _crx_like_display(self, ax, x)


class CU(ParamGate):
    r"""A collection of controlled single-qubit rotation gates.

    For a 2-qubit quantum circuit, when ``qubits_idx`` is ``[0, 1]``, the matrix form of such a gate is:

    .. math::

        \begin{align}
            CU
            &=
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \cos\frac\theta2 &-e^{i\lambda}\sin\frac\theta2 \\
                0 & 0 & e^{i\phi}\sin\frac\theta2&e^{i(\phi+\lambda)}\cos\frac\theta2
            \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``paddle.Tensor`` or ``float``.
    """

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            'gatename': 'cu',
            'texname': r'$U$',
            'plot_width': 1.65,
        }
        super().__init__(
            cu_gate, param, depth, 3, param_sharing, qubits_idx, gate_info, num_qubits, False, num_acted_qubits=2)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _crx_like_display(self, ax, x)


class RXX(ParamGate):
    r"""A collection of RXX gates.

    The matrix form of such a gate is:

    .. math::

        \begin{align}
            RXX(\theta) =
                \begin{bmatrix}
                    \cos\frac{\theta}{2} & 0 & 0 & -i\sin\frac{\theta}{2} \\
                    0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\
                    0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
                    -i\sin\frac{\theta}{2} & 0 & 0 & \cos\frac{\theta}{2}
                \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``paddle.Tensor`` or ``float``.
    """

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            'gatename': 'rxx',
            'texname': r'$R_{xx}$',
            'plot_width': 1.0,
        }
        super().__init__(
            rxx_gate, param, depth, 1, param_sharing, qubits_idx, gate_info, num_qubits, False, num_acted_qubits=2)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _rxx_like_display(self, ax, x)


class RYY(ParamGate):
    r"""A collection of RYY gates.

    The matrix form of such a gate is:

    .. math::

        \begin{align}
            RYY(\theta) =
                \begin{bmatrix}
                    \cos\frac{\theta}{2} & 0 & 0 & i\sin\frac{\theta}{2} \\
                    0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\
                    0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
                    i\sin\frac{\theta}{2} & 0 & 0 & cos\frac{\theta}{2}
                \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``paddle.Tensor`` or ``float``.
    """

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            'gatename': 'ryy',
            'texname': r'$R_{yy}$',
            'plot_width': 1.0,
        }

        super().__init__(
            ryy_gate, param, depth, 1, param_sharing, qubits_idx, gate_info, num_qubits, False, num_acted_qubits=2)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _rxx_like_display(self, ax, x)


class RZZ(ParamGate):
    r"""A collection of RZZ gates.

    The matrix form of such a gate is:

    .. math::

        \begin{align}
            RZZ(\theta) =
                \begin{bmatrix}
                    e^{-i\frac{\theta}{2}} & 0 & 0 & 0 \\
                    0 & e^{i\frac{\theta}{2}} & 0 & 0 \\
                    0 & 0 & e^{i\frac{\theta}{2}} & 0 \\
                    0 & 0 & 0 & e^{-i\frac{\theta}{2}}
                \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``paddle.Tensor`` or ``float``.
    """

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            'gatename': 'rzz',
            'texname': r'$R_{zz}$',
            'plot_width': 1.0,
        }
        super().__init__(
            rzz_gate, param, depth, 1, param_sharing, qubits_idx, gate_info, num_qubits, False, num_acted_qubits=2)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _rxx_like_display(self, ax, x)


class MS(Gate):
    r"""A collection of Mølmer-Sørensen (MS) gates for trapped ion devices.

    The matrix form of such a gate is:

    .. math::

        \begin{align}
            MS = RXX(-\frac{\pi}{2}) = \frac{1}{\sqrt{2}}
                \begin{bmatrix}
                    1 & 0 & 0 & i \\
                    0 & 1 & i & 0 \\
                    0 & i & 1 & 0 \\
                    i & 0 & 0 & 1
                \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """

    __matrix = ms_gate('complex128')

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1
    ):
        gate_info = {
            'gatename': 'ms',
            'texname': r'$MS$',
            'plot_width': 0.6,
        }
        super().__init__(
            None, qubits_idx, depth, gate_info, num_qubits, check_legality=False, num_acted_qubits=2)

    @property
    def matrix(self) -> paddle.Tensor:
        if self.dtype == 'complex64':
            return MS.__matrix.cast('complex64')
        return MS.__matrix

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _oracle_like_display(self, ax, x)


class CSWAP(Gate):
    r"""A collection of CSWAP (Fredkin) gates.

    The matrix form of such a gate is:

    .. math::

        \begin{align}
            CSWAP =
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
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    __matrix = cswap_gate('complex128')

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1
    ):
        gate_info = {
            'gatename': 'cswap',
            'texname': r'$CSWAP$',
            'plot_width': 0.2,
        }
        super().__init__(
            None, qubits_idx, depth, gate_info, num_qubits, check_legality=False, num_acted_qubits=3)

    @property
    def matrix(self) -> paddle.Tensor:
        if self.dtype == 'complex64':
            return CSWAP.__matrix.cast('complex64')
        return CSWAP.__matrix

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _cswap_display(self, ax, x)


class CCX(Gate):
    r"""A collection of CCX (Toffoli) gates.

    The matrix form of such a gate is:

    .. math::

        \begin{align}
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
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    __matrix = toffoli_gate('complex128')

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1
    ):
        gate_info = {
            'gatename': 'ccx',
            'texname': r'$Toffoli$',
            'plot_width': 0.2,
        }

        super().__init__(
            None, qubits_idx, depth, gate_info, num_qubits, check_legality=False, num_acted_qubits=3)

    @property
    def matrix(self) -> paddle.Tensor:
        if self.dtype == 'complex64':
            return CCX.__matrix.cast('complex64')
        return CCX.__matrix

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _tofolli_display(self, ax, x, )


Toffoli = CCX


class UniversalTwoQubits(ParamGate):
    r"""A collection of universal two-qubit gates. One of such a gate requires 15 parameters.

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``paddle.Tensor`` or ``float``.
    """

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            'gatename': 'uni2',
            'texname': r'$U$',
            'plot_width': 0.6,
        }
        super().__init__(
            universal2_gate, param, depth, 15, param_sharing, qubits_idx, gate_info, num_qubits, False,
            num_acted_qubits=2)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _oracle_like_display(self, ax, x)


class UniversalThreeQubits(ParamGate):
    r"""A collection of universal three-qubit gates. One of such a gate requires 81 parameters.

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``paddle.Tensor`` or ``float``.
    """

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, str]] = None,
            num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            'gatename': 'uni3',
            'texname': r'$U$',
            'plot_width': 0.6,
        }
        super().__init__(
            universal3_gate, param, depth, 81, param_sharing, qubits_idx, gate_info, num_qubits, False,
            num_acted_qubits=3)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _oracle_like_display(self, ax, x)
