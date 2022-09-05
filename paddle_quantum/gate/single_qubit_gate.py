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

import copy
import math
import numpy as np
import paddle.nn
import paddle_quantum
from . import functional
from .base import Gate, ParamGate
from ..backend import Backend
from paddle_quantum.intrinsic import _format_qubits_idx, _get_float_dtype
from typing import Optional, List, Union, Iterable


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
    def __init__(self, qubits_idx: Optional[Union[Iterable, int, str]] = 'full', num_qubits: Optional[int] = None, depth: Optional[int] = 1):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        self.gate_name = 'h'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            state.gate_history.append({
                'gate_name': 'h',
                'qubits_idx': copy.deepcopy(self.qubits_idx),
                'depth': self.depth,
            })
            return state
        for _ in range(0, self.depth):
            for qubit_idx in self.qubits_idx:
                state = functional.h(state, qubit_idx, self.dtype, self.backend)
        return state


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
    def __init__(self, qubits_idx: Optional[Union[Iterable, int, str]] = 'full', num_qubits: Optional[int] = None, depth: Optional[int] = 1):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        self.gate_name = 's'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            state.gate_history.append({
                'gate_name': 's',
                'qubits_idx': copy.deepcopy(self.qubits_idx),
                'depth': self.depth,
            })
            return state
        for _ in range(0, self.depth):
            for qubit_idx in self.qubits_idx:
                state = functional.s(state, qubit_idx, self.dtype, self.backend)
        return state


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
    def __init__(self, qubits_idx: Optional[Union[Iterable, int, str]] = 'full', num_qubits: Optional[int] = None, depth: Optional[int] = 1):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        self.gate_name = 't'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            state.gate_history.append({
                'gate_name': 't',
                'qubits_idx': copy.deepcopy(self.qubits_idx),
                'depth': self.depth,
            })
            return state
        for _ in range(0, self.depth):
            for qubit_idx in self.qubits_idx:
                state = functional.t(state, qubit_idx, self.dtype, self.backend)
        return state


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
    def __init__(self, qubits_idx: Optional[Union[Iterable, int, str]] = 'full', num_qubits: Optional[int] = None, depth: Optional[int] = 1):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        self.gate_name = 'x'
        
    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            state.gate_history.append({
                'gate_name': 'x',
                'qubits_idx': copy.deepcopy(self.qubits_idx),
                'depth': self.depth,
            })
            return state
        for _ in range(0, self.depth):
            for qubit_idx in self.qubits_idx:
                state = functional.x(state, qubit_idx, self.dtype, self.backend)
        return state


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
    def __init__(self, qubits_idx: Optional[Union[Iterable, int, str]] = 'full', num_qubits: Optional[int] = None, depth: Optional[int] = 1):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        self.gate_name = 'y'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            state.gate_history.append({
                'gate_name': 'y',
                'qubits_idx': copy.deepcopy(self.qubits_idx),
                'depth': self.depth,
            })
            return state
        for _ in range(0, self.depth):
            for qubit_idx in self.qubits_idx:
                state = functional.y(state, qubit_idx, self.dtype, self.backend)
        return state


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
    def __init__(self, qubits_idx: Optional[Union[Iterable, int, str]] = 'full', num_qubits: Optional[int] = None, depth: Optional[int] = 1):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        self.gate_name = 'z'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            state.gate_history.append({
                'gate_name': 'z',
                'qubits_idx': copy.deepcopy(self.qubits_idx),
                'depth': self.depth,
            })
            return state
        for _ in range(0, self.depth):
            for qubit_idx in self.qubits_idx:
                state = functional.z(state, qubit_idx, self.dtype, self.backend)
        return state


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
            self, qubits_idx: Optional[Union[Iterable, int, str]] = 'full', num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        self.param_sharing = param_sharing
        
        param_shape = [depth, 1 if param_sharing else len(self.qubits_idx)]
        self.theta_generation(param, param_shape)
        self.gate_name = 'p'

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        if self.backend == Backend.QuLeaf and state.backend == Backend.QuLeaf:
            raise NotImplementedError
        for depth_idx in range(0, self.depth):
            if self.param_sharing:
                for qubit_idx in self.qubits_idx:
                    state = functional.p(state, self.theta[depth_idx], qubit_idx, self.dtype, self.backend)
            else:
                for param_idx, qubit_idx in enumerate(self.qubits_idx):
                    state = functional.p(
                        state, self.theta[depth_idx, param_idx], qubit_idx, self.dtype, self.backend)
        return state


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
            self, qubits_idx: Optional[Union[Iterable, int, str]] = 'full', num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        self.param_sharing = param_sharing
        
        param_shape = [depth, 1 if param_sharing else len(self.qubits_idx)]
        self.theta_generation(param, param_shape)
        self.gate_name = 'rx'

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
                'gate_name': 'rx',
                'qubits_idx': copy.deepcopy(self.qubits_idx),
                'depth': self.depth,
                'param': param_idx_list,
                'param_sharing': self.param_sharing,
            })
            return state
        for depth_idx in range(0, self.depth):
            if self.param_sharing:
                for qubit_idx in self.qubits_idx:
                    state = functional.rx(state, self.theta[depth_idx], qubit_idx, self.dtype, self.backend)
            else:
                for param_idx, qubit_idx in enumerate(self.qubits_idx):
                    state = functional.rx(
                        state, self.theta[depth_idx, param_idx], qubit_idx, self.dtype, self.backend)
        return state


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
            self, qubits_idx: Optional[Union[Iterable, int, str]] = 'full', num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        self.param_sharing = param_sharing
        
        param_shape = [depth, 1 if param_sharing else len(self.qubits_idx)]
        self.theta_generation(param, param_shape)
        self.gate_name = 'ry'

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
                state.num_param += len(self.qubits_idx)
            state.gate_history.append({
                'gate_name': 'ry',
                'qubits_idx': copy.deepcopy(self.qubits_idx),
                'depth': self.depth,
                'param': param_idx_list,
                'param_sharing': self.param_sharing,
            })
            return state
        for depth_idx in range(0, self.depth):
            if self.param_sharing:
                for qubit_idx in self.qubits_idx:
                    state = functional.ry(state, self.theta[depth_idx], qubit_idx, self.dtype, self.backend)
            else:
                for param_idx, qubit_idx in enumerate(self.qubits_idx):
                    state = functional.ry(
                        state, self.theta[depth_idx, param_idx], qubit_idx, self.dtype, self.backend)
        return state


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
            self, qubits_idx: Optional[Union[Iterable, int, str]] = 'full', num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        self.param_sharing = param_sharing
        
        param_shape = [depth, 1 if param_sharing else len(self.qubits_idx)]
        self.theta_generation(param, param_shape)
        self.gate_name = 'rz'

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
                'gate_name': 'rz',
                'qubits_idx': copy.deepcopy(self.qubits_idx),
                'depth': self.depth,
                'param': param_idx_list,
                'param_sharing': self.param_sharing,
            })
            return state
        for depth_idx in range(0, self.depth):
            if self.param_sharing:
                for qubit_idx in self.qubits_idx:
                    state = functional.rz(state, self.theta[depth_idx], qubit_idx, self.dtype, self.backend)
            else:
                for param_idx, qubit_idx in enumerate(self.qubits_idx):
                    state = functional.rz(
                        state, self.theta[depth_idx, param_idx], qubit_idx, self.dtype, self.backend)
        return state


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
            self, qubits_idx: Optional[Union[Iterable, int, str]] = 'full', num_qubits: Optional[int] = None, depth: Optional[int] = 1,
            param: Optional[Union[paddle.Tensor, Iterable[float]]] = None, param_sharing: Optional[bool] = False
    ):
        super().__init__(depth)
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        self.param_sharing = param_sharing
        
        if param_sharing:
            param_shape = [depth, 3]
        else:
            param_shape = [depth, len(self.qubits_idx), 3]
        self.theta_generation(param, param_shape)
        self.gate_name = 'u'

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
                'gate_name': 'u3',
                'qubits_idx': copy.deepcopy(self.qubits_idx),
                'depth': self.depth,
                'param': param_idx_list,
                'param_sharing': self.param_sharing,
            })
            return state
        for depth_idx in range(0, self.depth):
            if self.param_sharing:
                for qubit_idx in self.qubits_idx:
                    state = functional.u3(state, self.theta[depth_idx], qubit_idx, self.dtype, self.backend)
            else:
                for param_idx, qubit_idx in enumerate(self.qubits_idx):
                    state = functional.u3(
                        state, self.theta[depth_idx, param_idx], qubit_idx, self.dtype, self.backend)
        return state
