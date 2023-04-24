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
from typing import Callable, List, Union, Iterable

import warnings
import paddle_quantum as pq
from . import functional
from .base import Gate, ParamGate
from .functional.visual import _c_oracle_like_display, _oracle_like_display
from ..intrinsic import _format_qubits_idx


class Oracle(Gate):
    """An oracle as a gate.

    Args:
        oracle: Unitary oracle to be implemented.
        qubits_idx: Indices of the qubits on which the gates are applied.
        num_qubits: Total number of qubits. Defaults to ``None``.
        depth: Number of layers. Defaults to ``1``.
    """
    def __init__(
            self, oracle: paddle.Tensor, qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int] = None,
            num_qubits: int = None, depth: int = 1, gate_info: dict = None
    ):
        super().__init__(oracle, qubits_idx, depth, gate_info, num_qubits)
    
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
        complex_dtype = oracle.dtype
        
        dimension = oracle.shape[0]
        oracle = (
            paddle.kron(paddle.to_tensor([[1.0, 0], [0, 0]], dtype=complex_dtype), paddle.eye(dimension).cast(complex_dtype)) +
            paddle.kron(paddle.to_tensor([[0.0, 0], [0, 1]], dtype=complex_dtype), oracle)
        )

        default_gate_info = {
            'gatename': 'cO',
            'texname': r'$O$',
            'plot_width': 0.6,
        }
        if gate_info is not None:
            default_gate_info.update(gate_info)
        super().__init__(oracle, qubits_idx, depth, gate_info, num_qubits)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float,) -> float:
        return _c_oracle_like_display(self, ax, x)


class ParamOracle(ParamGate):
    """An parameterized oracle as a gate

    Args:
        generator: function that generates the oracle.
        param: input parameters of quantum parameterized gates. Defaults to ``None`` i.e. randomized.
        qubits_idx: indices of the qubits on which this gate acts on. Defaults to ``None`` i.e. list(range(num_qubits)).
        depth: number of layers. Defaults to ``1``.
        num_acted_param: the number of parameters required for a single operation.
        param_sharing: whether all operations are shared by the same parameter set.
        gate_info: information of this gate that will be placed into the gate history or plotted by a Circuit. 
        Defaults to ``None``.
        num_qubits: total number of qubits. Defaults to ``None``.

    """
    def __init__(
            self, generator: Callable[[paddle.Tensor], paddle.Tensor],
            param: Union[paddle.Tensor, float, List[float]] = None,
            depth: int = 1, num_acted_param: int = 1, param_sharing: bool = False,
            qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int] = None,
            gate_info: dict = None, num_qubits: int = None
    ):
        super().__init__(generator, param, depth, num_acted_param, param_sharing, qubits_idx, gate_info, num_qubits)
