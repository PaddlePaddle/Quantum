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
The source file of the oracle class and the control oracle class.
"""

import math
import matplotlib
import paddle

import warnings
import paddle_quantum as pq
from . import functional
from .base import Gate
from ..intrinsic import _format_qubits_idx
from typing import Union, Iterable
from .functional.visual import _c_oracle_like_display, _oracle_like_display


class Oracle(Gate):
    """An oracle as a gate.

    Args:
        oracle: Unitary oracle to be implemented.
        qubits_idx: Indices of the qubits on which the gates are applied.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    def __init__(
            self, oracle: paddle.Tensor, qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int],
            num_qubits: int = None, depth: int = 1, gate_info: dict = None
    ):
        super().__init__(depth)
        complex_dtype = pq.get_dtype()
        oracle = oracle.cast(complex_dtype)

        dimension = oracle.shape[0]
        err = paddle.norm(paddle.abs(oracle @ paddle.conj(oracle.T) - paddle.cast(paddle.eye(dimension), complex_dtype))).item()
        if err > min(1e-6 * dimension, 0.01):
            warnings.warn(
                f"\nThe input oracle may not be a unitary: norm(U * U^d - I) = {err}.", UserWarning)

        num_acted_qubits = int(math.log2(dimension))
        self.oracle = oracle
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits)

        self.gate_info = {
            'gatename': 'O',
            'texname': r'$O$',
            'plot_width': 0.6,
        }
        if gate_info:
            self.gate_info.update(gate_info)

    def forward(self, state: pq.State) -> pq.State:
        for _ in range(self.depth):
            for qubits_idx in self.qubits_idx:
                state = functional.oracle(state, self.oracle, qubits_idx, self.backend)
        return state
    
    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float,) -> float:
        return _oracle_like_display(self, ax, x)


class ControlOracle(Gate):
    """A controlled oracle as a gate.

    Args:
        oracle: Unitary oracle to be implemented.
        qubits_idx: Indices of the qubits on which the gates are applied.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    def __init__(
            self, oracle: paddle.Tensor, qubits_idx: Union[Iterable[Iterable[int]], Iterable[int]],
            num_qubits: int = None, depth: int = 1, gate_info: dict = None
    ) -> None:
        super().__init__(depth)
        complex_dtype = pq.get_dtype()
        oracle = oracle.cast(complex_dtype)
        
        dimension = oracle.shape[0]
        err = paddle.norm(paddle.abs(oracle @ paddle.conj(oracle.T) - paddle.cast(paddle.eye(dimension), complex_dtype))).item()
        if  err > min(1e-6 * dimension, 0.01):
            warnings.warn(
                f"\nThe input oracle may not be a unitary: norm(U * U^d - I) = {err}.", UserWarning)

        num_acted_qubits = int(math.log2(dimension))
        oracle = (
            paddle.kron(paddle.to_tensor([[1.0, 0], [0, 0]], dtype=complex_dtype), paddle.eye(2 ** num_acted_qubits)) +
            paddle.kron(paddle.to_tensor([[0.0, 0], [0, 1]], dtype=complex_dtype), oracle)
        )
        num_acted_qubits += 1
        self.oracle = oracle
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits)

        self.gate_info = {
            'gatename': 'cO',
            'texname': r'$O$',
            'plot_width': 0.6,
        }
        if gate_info:
            self.gate_info.update(gate_info)

    def forward(self, state: pq.State) -> pq.State:
        for _ in range(self.depth):
            for qubits_idx in self.qubits_idx:
                state = functional.oracle(state, self.oracle, qubits_idx, self.backend)
        return state

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float,) -> float:
        return _c_oracle_like_display(self, ax, x)
