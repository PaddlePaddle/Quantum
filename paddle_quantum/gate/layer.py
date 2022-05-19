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
The source file of the class for quantum circuit templates.
"""

import math
import numpy as np
import paddle
import paddle_quantum
from paddle_quantum.gate import functional
from paddle_quantum.intrinsic import _get_float_dtype
from .base import Gate
from typing import Iterable, List, Optional, Union


def qubits_idx_filter(qubits_idx: Union[Iterable[int], str], num_qubits: int) -> List[Iterable[int]]:
    r"""Check the validity of ``qubits_idx`` and ``num_qubits``.

    Args:
        qubits_idx: Indices of qubits.
        num_qubits: Total number of qubits.

    Raises:
        RuntimeError: You must specify ``qubits_idx`` or ``num_qubits`` to instantiate the class.
        ValueError: The ``qubits_idx`` must be ``Iterable`` or ``'full'``.

    Returns:
        Checked indices of qubits.
    """
    if qubits_idx == 'full':
        if num_qubits is None:
            raise RuntimeError("You must specify qubits_idx or num_qubits to instantiate the class.")
        return list(range(0, num_qubits))
    elif isinstance(qubits_idx, Iterable):
        return list(qubits_idx)
    else:
        raise ValueError("The param qubits_idx must be iterable or full")


class SuperpositionLayer(Gate):
    r"""Layers of Hadamard gates.

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    def __init__(
            self, qubits_idx: Union[Iterable[int], str] = 'full', num_qubits: int = None, depth: int = 1
    ):
        super().__init__(depth)
        self.qubits_idx = qubits_idx_filter(qubits_idx, num_qubits)

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        for _ in range(0, self.depth):
            for qubit_idx in self.qubits_idx:
                state = functional.h(state, qubit_idx, self.dtype, self.backend)
        return state


class WeakSuperpositionLayer(Gate):
    r"""Layers of Ry gates with a rotation angle :math:`\pi/4`.

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    def __init__(
            self, qubits_idx: Union[Iterable[int], str] = 'full', num_qubits: int = None, depth: int = 1
    ):
        super().__init__(depth)
        self.qubits_idx = qubits_idx_filter(qubits_idx, num_qubits)

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        theta = paddle.to_tensor([np.pi / 4])
        for _ in range(0, self.depth):
            for qubit_idx in self.qubits_idx:
                state = functional.ry(state, theta, qubit_idx, self.dtype, self.backend)
        return state


class LinearEntangledLayer(Gate):
    r"""Linear entangled layers consisting of Ry gates, Rz gates, and CNOT gates.

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    def __init__(
            self, qubits_idx: Union[Iterable[int], str] = 'full', num_qubits: int = None, depth: int = 1
    ):
        super().__init__(depth)
        self.qubits_idx = qubits_idx_filter(qubits_idx, num_qubits)

        float_dtype = _get_float_dtype(self.dtype)
        param_shape = [self.depth] + list(np.shape(self.qubits_idx)) + [2]
        initializer = paddle.nn.initializer.Uniform(low=0, high=2 * math.pi)
        theta = self.create_parameter(
            shape=param_shape,
            dtype=float_dtype,
            default_initializer=initializer
        )
        self.add_parameter('theta', theta)

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        for depth_idx in range(0, self.depth):
            for param_idx, qubit_idx in enumerate(self.qubits_idx):
                state = functional.ry(
                    state, self.theta[depth_idx, param_idx, 0], qubit_idx, self.dtype, self.backend)
            for idx in range(0, len(self.qubits_idx) - 1):
                state = functional.cnot(
                    state, [self.qubits_idx[idx], self.qubits_idx[idx + 1]], self.dtype, self.backend)
            for param_idx, qubit_idx in enumerate(self.qubits_idx):
                state = functional.rz(
                    state, self.theta[depth_idx, param_idx, 1], qubit_idx, self.dtype, self.backend)
            for idx in range(0, len(self.qubits_idx) - 1):
                state = functional.cnot(
                    state, [self.qubits_idx[idx], self.qubits_idx[idx + 1]], self.dtype, self.backend)
        return state


class RealEntangledLayer(Gate):
    r"""Strongly entangled layers consisting of Ry gates and CNOT gates.

    Note:
        The mathematical representation of this layer of quantum gates is a real unitary matrix.
        This ansatz is from the following paper: https://arxiv.org/pdf/1905.10876.pdf.

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    def __init__(
            self, qubits_idx: Union[Iterable[int], str] = 'full', num_qubits: int = None, depth: int = 1
    ):
        super().__init__(depth)
        self.qubits_idx = qubits_idx_filter(qubits_idx, num_qubits)
        assert len(self.qubits_idx) > 1, 'you need at least 2 qubits'

        float_dtype = _get_float_dtype(self.dtype)
        param_shape = [self.depth] + list(np.shape(self.qubits_idx))
        initializer = paddle.nn.initializer.Uniform(low=0, high=2 * math.pi)
        theta = self.create_parameter(
            shape=param_shape,
            dtype=float_dtype,
            default_initializer=initializer
        )
        self.add_parameter('theta', theta)

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        for depth_idx in range(0, self.depth):
            for param_idx, qubit_idx in enumerate(self.qubits_idx):
                state = functional.ry(
                    state, self.theta[depth_idx, param_idx], qubit_idx, self.dtype, self.backend)
            for qubit_idx in range(0, len(self.qubits_idx)):
                state = functional.cnot(
                    state, [self.qubits_idx[qubit_idx], self.qubits_idx[(qubit_idx + 1) % len(self.qubits_idx)]],
                    self.dtype, self.backend)
        return state


class ComplexEntangledLayer(Gate):
    r"""Strongly entangled layers consisting of single-qubit rotation gates and CNOT gates.

    Note:
        The mathematical representation of this layer of quantum gates is a complex unitary matrix.
        This ansatz is from the following paper: https://arxiv.org/abs/1804.00633.

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    def __init__(
            self, qubits_idx: Union[Iterable[int], str] = 'full', num_qubits: int = None, depth: int = 1
    ):
        super().__init__(depth)
        self.qubits_idx = qubits_idx_filter(qubits_idx, num_qubits)
        assert len(self.qubits_idx) > 1, 'you need at least 2 qubits'

        float_dtype = _get_float_dtype(self.dtype)
        param_shape = [self.depth] + list(np.shape(self.qubits_idx)) + [3]
        initializer = paddle.nn.initializer.Uniform(low=0, high=2 * math.pi)
        theta = self.create_parameter(
            shape=param_shape, dtype=float_dtype,
            default_initializer=initializer
        )
        self.add_parameter('theta', theta)

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        for depth_idx in range(0, self.depth):
            for param_idx, qubit_idx in enumerate(self.qubits_idx):
                state = functional.u3(
                    state, self.theta[depth_idx, param_idx], qubit_idx, self.dtype, self.backend)
            for qubit_idx in range(0, len(self.qubits_idx)):
                state = functional.cnot(
                    state, [self.qubits_idx[qubit_idx], self.qubits_idx[(qubit_idx + 1) % len(self.qubits_idx)]],
                    self.dtype, self.backend)
        return state


class RealBlockLayer(Gate):
    r"""Weakly entangled layers consisting of Ry gates and CNOT gates.

    Note:
        The mathematical representation of this layer of quantum gates is a real unitary matrix.

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    def __init__(
            self, qubits_idx: Union[Iterable[int], str] = 'full', num_qubits: int = None, depth: int = 1
    ):
        super().__init__(depth)
        self.qubits_idx = qubits_idx_filter(qubits_idx, num_qubits)
        assert len(self.qubits_idx) > 1, 'you need at least 2 qubits'

        float_dtype = _get_float_dtype(self.dtype)
        # TODO: currently do not support multiple dimensions of qubits_idx
        param_shape = [self.depth] + [len(self.qubits_idx) - 1] + [4]
        initializer = paddle.nn.initializer.Uniform(low=0, high=2 * math.pi)
        theta = self.create_parameter(
            shape=param_shape,
            dtype=float_dtype,
            default_initializer=initializer
        )
        self.add_parameter('theta', theta)

    def __add_real_block(self, theta: paddle.Tensor, position: List[int]) -> None:
        assert len(theta) == 4, 'the length of theta is not right'

        position[0] = self.qubits_idx[position[0]]
        position[1] = self.qubits_idx[position[1]]

        state = functional.ry(self.state, theta[0], position[0], self.dtype, self.backend)
        state = functional.ry(state, theta[1], position[1], self.dtype, self.backend)

        state = functional.cnot(state, [position[0], position[1]], self.dtype, self.backend)

        state = functional.ry(state, theta[2], position[0], self.dtype, self.backend)
        state = functional.ry(state, theta[3], position[1], self.dtype, self.backend)

        self.state = state

    def __add_real_layer(self, theta: paddle.Tensor, position: List) -> None:
        assert theta.shape[1] == 4 and theta.shape[0] == (position[1] - position[0] + 1) / 2, \
            'the shape of theta is not right'

        for i in range(position[0], position[1], 2):
            self.__add_real_block(theta[int((i - position[0]) / 2)], [i, i + 1])

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        self.state = state
        n = len(self.qubits_idx)

        if n % 2 == 0:
            for depth_idx in range(self.depth):
                self.__add_real_layer(self.theta[depth_idx, :int(n / 2)], [0, n - 1])
                self.__add_real_layer(self.theta[depth_idx, int(n / 2):], [1, n - 2]) if n > 2 else None
        else:
            for depth_idx in range(self.depth):
                self.__add_real_layer(self.theta[depth_idx, :int((n - 1) / 2)], [0, n - 2])
                self.__add_real_layer(self.theta[depth_idx, int((n - 1) / 2):], [1, n - 1])

        return self.state


class ComplexBlockLayer(Gate):
    r"""Weakly entangled layers consisting of single-qubit rotation gates and CNOT gates.

    Note:
        The mathematical representation of this layer of quantum gates is a complex unitary matrix.

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    def __init__(self, qubits_idx: Optional[Union[Iterable[int], str]] = 'full', 
                 num_qubits: Optional[int] = None, depth: Optional[int] = 1) -> None:
        super().__init__(depth)
        self.qubits_idx = qubits_idx_filter(qubits_idx, num_qubits)
        assert len(self.qubits_idx) > 1, 'you need at least 2 qubits'

        float_dtype = _get_float_dtype(self.dtype)
        # TODO: currently do not support multiple dimensions of qubits_idx
        param_shape = [self.depth] + [len(self.qubits_idx) - 1] + [12]
        initializer = paddle.nn.initializer.Uniform(low=0, high=2 * math.pi)
        theta = self.create_parameter(
            shape=param_shape,
            dtype=float_dtype,
            default_initializer=initializer
        )
        self.add_parameter('theta', theta)
        self.state = None

    def __add_complex_block(self, theta: paddle.Tensor, position: List[int]) -> None:
        assert len(theta) == 12, 'the length of theta is not right'

        position[0] = self.qubits_idx[position[0]]
        position[1] = self.qubits_idx[position[1]]

        state = functional.u3(self.state, [theta[0], theta[1], theta[2]], position[0], self.dtype, self.backend)
        state = functional.u3(state, [theta[3], theta[4], theta[5]], position[1], self.dtype, self.backend)

        state = functional.cnot(state, [position[0], position[1]], self.dtype, self.backend)

        state = functional.u3(state, [theta[6], theta[7], theta[8]], position[0], self.dtype, self.backend)
        state = functional.u3(state, [theta[9], theta[10], theta[11]], position[1], self.dtype, self.backend)

        self.state = state

    def __add_complex_layer(self, theta: paddle.Tensor, position: List[int]) -> None:
        assert theta.shape[1] == 12 and theta.shape[0] == (position[1] - position[0] + 1) / 2, \
            'the shape of theta is not right'
        for i in range(position[0], position[1], 2):
            self.__add_complex_block(theta[int((i - position[0]) / 2)], [i, i + 1])

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        self.state = state

        num_acted_qubits = len(self.qubits_idx)

        if num_acted_qubits % 2 == 0:
            for depth_idx in range(self.depth):
                self.__add_complex_layer(self.theta[depth_idx, :num_acted_qubits // 2], [0, num_acted_qubits - 1])
                if num_acted_qubits > 2:
                    self.__add_complex_layer(self.theta[depth_idx, num_acted_qubits // 2:], [1, num_acted_qubits - 2])
        else:
            for depth_idx in range(0, self.depth):
                self.__add_complex_layer(self.theta[depth_idx, :(num_acted_qubits - 1) // 2], [0, num_acted_qubits - 2])
                self.__add_complex_layer(self.theta[depth_idx, (num_acted_qubits - 1) // 2:], [1, num_acted_qubits - 1])

        return self.state


class QAOALayer(Gate):
    # TODO: only maxcut now
    def __init__(
            self, edges: Iterable, nodes: Iterable, depth: int = 1
    ):
        super().__init__(depth)
        float_dtype = _get_float_dtype(self.dtype)
        initializer = paddle.nn.initializer.Uniform(low=0, high=2 * math.pi)
        self.edges = edges
        self.nodes = nodes
        gamma = self.create_parameter(
            shape=[self.depth],
            dtype=float_dtype,
            default_initializer=initializer
        )
        beta = self.create_parameter(
            shape=[self.depth],
            dtype=float_dtype,
            default_initializer=initializer
        )
        self.add_parameter('gamma', gamma)
        self.add_parameter('beta', beta)

    def forward(self, state: paddle_quantum.State) -> paddle_quantum.State:
        for depth_idx in range(0, self.depth):
            for node0, node1 in self.edges:
                state = functional.cnot(state, [node0, node1], self.dtype, self.backend)
                state = functional.rz(state, self.gamma[depth_idx], node1, self.dtype, self.backend)
                state = functional.cnot(state, [node0, node1], self.dtype, self.backend)
            for node in self.nodes:
                state = functional.rx(state, self.beta[depth_idx], node, self.dtype, self.backend)
        return state
