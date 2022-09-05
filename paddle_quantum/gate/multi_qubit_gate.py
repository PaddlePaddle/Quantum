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

import copy
import math
import paddle
import paddle_quantum
from . import functional
from .base import Gate, ParamGate
from ..backend import Backend
from ..intrinsic import _format_qubits_idx, _get_float_dtype
from typing import Optional, Union, Iterable


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
    """
    def __init__(self, qubits_idx: Optional[Union[Iterable, str]] = 'cycle', num_qubits: Optional[int] = None, depth: Optional[int] = 1):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits=2)
        self.gate_name = 'cnot'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            state.gate_history.append({
                'gate_name': 'cnot',
                'qubits_idx': copy.deepcopy(self.qubits_idx),
                'depth': self.depth,
            })
            return state
        for _ in range(0, self.depth):
            for qubits_idx in self.qubits_idx:
                state = functional.cnot(state, qubits_idx, self.dtype, self.backend)
        return state


class CX(Gate):
    r"""Same as CNOT.

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    def __init__(self, qubits_idx: Optional[Union[Iterable, str]] = 'cycle', 
                 num_qubits: Optional[int] = None, depth: Optional[int] = 1):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits=2)
        self.gate_name = 'cnot'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            state.gate_history.append({
                'gate_name': 'cx',
                'qubits_idx': copy.deepcopy(self.qubits_idx),
                'depth': self.depth,
            })
            return state
        for _ in range(0, self.depth):
            for qubits_idx in self.qubits_idx:
                state = functional.cx(state, qubits_idx, self.dtype, self.backend)
        return state


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
    def __init__(self, qubits_idx: Optional[Union[Iterable, str]] = 'cycle', 
                 num_qubits: Optional[int] = None, depth: Optional[int] = 1):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits=2)
        self.gate_name = 'cy'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            state.gate_history.append({
                'gate_name': 'cy',
                'qubits_idx': copy.deepcopy(self.qubits_idx),
                'depth': self.depth,
            })
            return state
        for _ in range(0, self.depth):
            for qubits_idx in self.qubits_idx:
                state = functional.cy(state, qubits_idx, self.dtype, self.backend)
        return state


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
    def __init__(self, qubits_idx: Optional[Union[Iterable, str]] = 'cycle', 
                 num_qubits: Optional[int] = None, depth: Optional[int] = 1):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits=2)
        self.gate_name = 'cz'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            state.gate_history.append({
                'gate_name': 'cz',
                'qubits_idx': copy.deepcopy(self.qubits_idx),
                'depth': self.depth,
            })
            return state
        for _ in range(0, self.depth):
            for qubits_idx in self.qubits_idx:
                state = functional.cz(state, qubits_idx, self.dtype, self.backend)
        return state


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
    def __init__(self, qubits_idx: Optional[Union[Iterable, str]] = 'cycle', 
                 num_qubits: Optional[int] = None, depth: Optional[int] = 1):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits=2)
        self.gate_name = 'swap'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            state.gate_history.append({
                'gate_name': 'swap',
                'qubits_idx': copy.deepcopy(self.qubits_idx),
                'depth': self.depth,
            })
            return state
        for _ in range(0, self.depth):
            for qubits_idx in self.qubits_idx:
                state = functional.swap(state, qubits_idx, self.dtype, self.backend)
        return state


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
            self, qubits_idx: Optional[Union[Iterable, str]] = 'cycle', num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits=2)
        self.param_sharing = param_sharing
        
        param_shape = [depth, 1 if param_sharing else len(self.qubits_idx)]
        self.theta_generation(param, param_shape)
        self.gate_name = 'cp'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            raise NotImplementedError
        for depth_idx in range(0, self.depth):
            if self.param_sharing:
                for qubit_idx in self.qubits_idx:
                    state = functional.cp(state, self.theta[depth_idx], qubit_idx, self.dtype, self.backend)
            else:
                for param_idx, qubit_idx in enumerate(self.qubits_idx):
                    state = functional.cp(
                        state, self.theta[depth_idx, param_idx], qubit_idx, self.dtype, self.backend)
        return state


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
            self, qubits_idx: Optional[Union[Iterable, str]] = 'cycle', num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits=2)
        self.param_sharing = param_sharing
        
        param_shape = [depth, 1 if param_sharing else len(self.qubits_idx)]
        self.theta_generation(param, param_shape)
        self.gate_name = 'crx'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            state.param_list.extend(self.theta)
            if self.param_sharing:
                param_idx_list = [[state.num_param]]
                state.num_param += 1
            else:
                param_idx_list = []
                for _ in range(0, self.depth):
                    param_idx_list.append(list(range(state.num_param, state.num_param + len(self.qubits_idx))))
                    state.num_param += self.theta.size
            state.gate_history.append({
                'gate_name': 'crx',
                'qubits_idx': copy.deepcopy(self.qubits_idx),
                'depth': self.depth,
                'param': param_idx_list,
                'param_sharing': self.param_sharing,
            })
            return state
        for depth_idx in range(0, self.depth):
            if self.param_sharing:
                for qubit_idx in self.qubits_idx:
                    state = functional.crx(state, self.theta[depth_idx], qubit_idx, self.dtype, self.backend)
            else:
                for param_idx, qubit_idx in enumerate(self.qubits_idx):
                    state = functional.crx(
                        state, self.theta[depth_idx, param_idx], qubit_idx, self.dtype, self.backend)
        return state


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
            self, qubits_idx: Optional[Union[Iterable, str]] = 'cycle', num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits=2)
        self.param_sharing = param_sharing
        
        param_shape = [depth, 1 if param_sharing else len(self.qubits_idx)]
        self.theta_generation(param, param_shape)
        self.gate_name = 'cry'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            state.param_list.extend(self.theta)
            if self.param_sharing:
                param_idx_list = [[state.num_param]]
                state.num_param += 1
            else:
                param_idx_list = []
                for _ in range(0, self.depth):
                    param_idx_list.append(list(range(state.num_param, state.num_param + len(self.qubits_idx))))
                    state.num_param += self.theta.size
            state.gate_history.append({
                'gate_name': 'cry',
                'qubits_idx': copy.deepcopy(self.qubits_idx),
                'depth': self.depth,
                'param': param_idx_list,
                'param_sharing': self.param_sharing,
            })
            return state
        for depth_idx in range(0, self.depth):
            if self.param_sharing:
                for qubit_idx in self.qubits_idx:
                    state = functional.cry(state, self.theta[depth_idx], qubit_idx, self.dtype, self.backend)
            else:
                for param_idx, qubit_idx in enumerate(self.qubits_idx):
                    state = functional.cry(
                        state, self.theta[depth_idx, param_idx], qubit_idx, self.dtype, self.backend)
        return state


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
            self, qubits_idx: Optional[Union[Iterable, str]] = 'cycle', num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits=2)
        self.param_sharing = param_sharing
        
        param_shape = [depth, 1 if param_sharing else len(self.qubits_idx)]
        self.theta_generation(param, param_shape)
        self.gate_name = 'crz'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            state.param_list.extend(self.theta)
            if self.param_sharing:
                param_idx_list = [[state.num_param]]
                state.num_param += 1
            else:
                param_idx_list = []
                for _ in range(0, self.depth):
                    param_idx_list.append(list(range(state.num_param, state.num_param + len(self.qubits_idx))))
                    state.num_param += self.theta.size
            state.gate_history.append({
                'gate_name': 'crz',
                'qubits_idx': copy.deepcopy(self.qubits_idx),
                'depth': self.depth,
                'param': param_idx_list,
                'param_sharing': self.param_sharing,
            })
            return state
        for depth_idx in range(0, self.depth):
            if self.param_sharing:
                for qubit_idx in self.qubits_idx:
                    state = functional.crz(state, self.theta[depth_idx], qubit_idx, self.dtype, self.backend)
            else:
                for param_idx, qubit_idx in enumerate(self.qubits_idx):
                    state = functional.crz(
                        state, self.theta[depth_idx, param_idx], qubit_idx, self.dtype, self.backend)
        return state


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
            self, qubits_idx: Optional[Union[Iterable, str]] = 'cycle', num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits=2)
        self.param_sharing = param_sharing
        
        if param_sharing:
            param_shape = [depth, 3]
        else:
            param_shape = [depth, len(self.qubits_idx), 3]
        self.theta_generation(param, param_shape)
        self.gate_name = 'cu'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            state.param_list.extend(self.theta)
            if self.param_sharing:
                param_idx_list = [range(state.num_param, state.num_param + 3)]
                state.num_param += 3
            else:
                param_idx_list = []
                for _ in range(0, self.depth):
                    param_idx_list.append(list(range(state.num_param, state.num_param + len(self.qubits_idx))))
                    state.num_param += self.theta.size
            state.gate_history.append({
                'gate_name': 'cu',
                'qubits_idx': copy.deepcopy(self.qubits_idx),
                'depth': self.depth,
                'param': param_idx_list,
                'param_sharing': self.param_sharing,
            })
            return state
        for depth_idx in range(0, self.depth):
            if self.param_sharing:
                for qubit_idx in self.qubits_idx:
                    state = functional.cu(state, self.theta[depth_idx], qubit_idx, self.dtype, self.backend)
            else:
                for param_idx, qubit_idx in enumerate(self.qubits_idx):
                    state = functional.cu(
                        state, self.theta[depth_idx, param_idx], qubit_idx, self.dtype, self.backend)
        return state


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
            self, qubits_idx: Optional[Union[Iterable, str]] = 'cycle', num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits=2)
        self.param_sharing = param_sharing
        
        param_shape = [depth, 1 if param_sharing else len(self.qubits_idx)]
        self.theta_generation(param, param_shape)
        self.gate_name = 'rxx'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            raise NotImplementedError
        for depth_idx in range(0, self.depth):
            if self.param_sharing:
                for qubit_idx in self.qubits_idx:
                    state = functional.rxx(state, self.theta[depth_idx], qubit_idx, self.dtype, self.backend)
            else:
                for param_idx, qubit_idx in enumerate(self.qubits_idx):
                    state = functional.rxx(
                        state, self.theta[depth_idx, param_idx], qubit_idx, self.dtype, self.backend)
        return state


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
            self, qubits_idx: Optional[Union[Iterable, str]] = 'cycle', num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits=2)
        self.param_sharing = param_sharing
        
        param_shape = [depth, 1 if param_sharing else len(self.qubits_idx)]
        self.theta_generation(param, param_shape)
        self.gate_name = 'ryy'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            raise NotImplementedError
        for depth_idx in range(0, self.depth):
            if self.param_sharing:
                for qubit_idx in self.qubits_idx:
                    state = functional.ryy(state, self.theta[depth_idx], qubit_idx, self.dtype, self.backend)
            else:
                for param_idx, qubit_idx in enumerate(self.qubits_idx):
                    state = functional.ryy(
                        state, self.theta[depth_idx, param_idx], qubit_idx, self.dtype, self.backend)
        return state


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
            self, qubits_idx: Optional[Union[Iterable, str]] = 'cycle', num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits=2)
        self.param_sharing = param_sharing
        
        param_shape = [depth, 1 if param_sharing else len(self.qubits_idx)]
        self.theta_generation(param, param_shape)
        self.gate_name = 'rzz'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            raise NotImplementedError
        for depth_idx in range(0, self.depth):
            if self.param_sharing:
                for qubit_idx in self.qubits_idx:
                    state = functional.rzz(state, self.theta[depth_idx], qubit_idx, self.dtype, self.backend)
            else:
                for param_idx, qubit_idx in enumerate(self.qubits_idx):
                    state = functional.rzz(
                        state, self.theta[depth_idx, param_idx], qubit_idx, self.dtype, self.backend)
        return state


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
    def __init__(self, qubits_idx: Optional[Union[Iterable, str]] = 'cycle', num_qubits: Optional[int] = None, depth: Optional[int] = 1):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits=2)
        self.gate_name = 'ms'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            raise NotImplementedError
        for _ in range(0, self.depth):
            for qubits_idx in self.qubits_idx:
                state = functional.ms(state, qubits_idx, self.dtype, self.backend)
        return state


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
    def __init__(self, qubits_idx: Optional[Union[Iterable, str]] = 'cycle', num_qubits: Optional[int] = None, depth: Optional[int] = 1):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits=3)
        self.gate_name = 'cswap'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            state.gate_history.append({
                'gate_name': 'cswap',
                'qubits_idx': copy.deepcopy(self.qubits_idx),
                'depth': self.depth,
            })
            return state
        for _ in range(0, self.depth):
            for qubits_idx in self.qubits_idx:
                state = functional.cswap(state, qubits_idx, self.dtype, self.backend)
        return state


class Toffoli(Gate):
    r"""A collection of Toffoli gates.

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
    def __init__(self, qubits_idx: Optional[Union[Iterable, str]] = 'cycle', num_qubits: Optional[int] = None, depth: Optional[int] = 1):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits=3)
        self.gate_name = 'ccx'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            state.gate_history.append({
                'gate_name': 'toffoli',
                'qubits_idx': copy.deepcopy(self.qubits_idx),
                'depth': self.depth,
            })
            return state
        for _ in range(0, self.depth):
            for qubits_idx in self.qubits_idx:
                state = functional.toffoli(state, qubits_idx, self.dtype, self.backend)
        return state


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
            self, qubits_idx: Optional[Union[Iterable, str]] = 'cycle', num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits=2)
        self.param_sharing = param_sharing
        
        if param_sharing:
            param_shape = [depth, 15]
        else:
           param_shape = [depth, len(self.qubits_idx), 15]
        self.theta_generation(param, param_shape)
        self.gate_name = 'uni2'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            raise NotImplementedError
        for depth_idx in range(0, self.depth):
            if self.param_sharing:
                for qubit_idx in self.qubits_idx:
                    state = functional.universal_two_qubits(
                        state, self.theta[depth_idx], qubit_idx, self.dtype, self.backend)
            else:
                for param_idx, qubit_idx in enumerate(self.qubits_idx):
                    state = functional.universal_two_qubits(
                        state, self.theta[depth_idx, param_idx], qubit_idx, self.dtype, self.backend)
        return state


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
            self, qubits_idx: Optional[Union[Iterable, str]] = 'cycle', num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits=3)
        self.param_sharing = param_sharing
        
        if param_sharing:
            param_shape = [depth, 81]
        else:
           param_shape = [depth, len(self.qubits_idx), 81]
        self.theta_generation(param, param_shape)
        self.gate_name = 'uni3'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            raise NotImplementedError
        for depth_idx in range(0, self.depth):
            if self.param_sharing:
                for qubit_idx in self.qubits_idx:
                    state = functional.universal_three_qubits(
                        state, self.theta[depth_idx], qubit_idx, self.dtype, self.backend)
            else:
                for param_idx, qubit_idx in enumerate(self.qubits_idx):
                    state = functional.universal_three_qubits(
                        state, self.theta[depth_idx, param_idx], qubit_idx, self.dtype, self.backend)
        return state
