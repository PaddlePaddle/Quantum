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
The source file of the classes for quantum encoding.
"""

import paddle
import paddle_quantum
from paddle_quantum.gate import functional
from .base import Gate
from paddle_quantum.intrinsic import _get_float_dtype, _format_qubits_idx
from typing import Iterable, Optional, Union


class BasisEncoding(Gate):
    r"""Basis encoding gate for encoding input classical data into quantum states.

    In basis encoding, the input classical data can only consist of 0's and 1's. If the input data are 1101,
    then the quantum state after encoding is :math:`|1101\rangle`. Note that the quantum state before encoding is
    assumed to be :math:`|00\ldots 0\rangle`.

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
    """
    def __init__(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        self.gate_name = 'BasisEnc'

    def forward(self, feature: paddle.Tensor, state: 'paddle_quantum.State' = None) -> 'paddle_quantum.State':
        if state is None:
            state = paddle_quantum.state.zero_state(self.num_qubits)
        feature = paddle.cast(feature, 'int32')
        gate_history = []
        for idx, element in enumerate(feature):
            if element:
                state = functional.x(state, self.qubits_idx[idx], self.dtype, self.backend)
                gate_history.append({'gate': 'x', 'which_qubits': self.qubits_idx[idx], 'theta': None})
        self.gate_history = gate_history
        return state
    
    def gate_history_generation(self) -> None:
        if self.gate_history is None:
            raise RuntimeError("you must forward the encoding to receive the gate history")
        pass
        


class AmplitudeEncoding(Gate):
    r"""Amplitude encoding gate for encoding input classical data into quantum states.

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
    """
    def __init__(
            self, qubits_idx: Optional[Union[Iterable[int], int, str]] = 'full', num_qubits: Optional[int] = None
    ) -> None:
        if num_qubits is None:
            num_qubits = max(qubits_idx)
        super().__init__()
        self.num_qubits = num_qubits
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        self.gate_name = 'AmpEnc'

    def forward(self, feature: paddle.Tensor) -> 'paddle_quantum.State':
        def calc_location(location_of_bits_list):
            if len(location_of_bits_list) <= 1:
                result_list = [0, location_of_bits_list[0]]
            else:
                current_tmp = location_of_bits_list[0]
                inner_location_of_qubits_list = calc_location(location_of_bits_list[1:])
                current_list_len = len(inner_location_of_qubits_list)
                for each in range(0, current_list_len):
                    inner_location_of_qubits_list.append(inner_location_of_qubits_list[each] + current_tmp)
                result_list = inner_location_of_qubits_list
            return result_list

        def encoding_location_list(which_qubits):
            location_of_bits_list = []
            for qubit_idx in which_qubits:
                tmp = 2 ** (self.num_qubits - qubit_idx - 1)
                location_of_bits_list.append(tmp)
            result_list = calc_location(location_of_bits_list)

            return sorted(result_list)

        # Get the specific position of the code, denoted by sequence number (list)
        location_of_qubits_list = encoding_location_list(self.qubits_idx)
        # Classical data preprocessing
        feature = paddle.cast(feature, _get_float_dtype(paddle_quantum.get_dtype()))
        feature = paddle.flatten(feature)
        length = paddle.norm(feature, p=2)
        # Normalization
        feature = paddle.divide(feature, length)
        # Create a quantum state with all zero amplitudes
        data = paddle.zeros((2 ** self.num_qubits,), feature.dtype)
        # The value of the encoded amplitude is filled into the specified qubits
        for idx in range(0, len(feature)):
            data[location_of_qubits_list[idx]] = feature[idx]
        data = paddle.cast(data, dtype=paddle_quantum.get_dtype())
        if self.backend == paddle_quantum.Backend.DensityMatrix:
            data = paddle.unsqueeze(data, axis=1)
            data = paddle.matmul(data, paddle_quantum.linalg.dagger(data))
        elif self.backend != paddle_quantum.Backend.StateVector:
            raise ValueError("the mode should be state_vector or density_matrix")
        encoding_state = paddle_quantum.state.to_state(data, self.num_qubits)
        return encoding_state


class AngleEncoding(Gate):
    r"""Angle encoding gate for encoding input classical data into quantum states.

    Args:
        feature: Vector to be encoded.
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        encoding_gate: The type of quantum gates used for encoding, which should be one of ``"rx"``, ``"ry"``,
            and ``"rz"``. Defaults to ``None``.
    """
    def __init__(
            self, feature: paddle.Tensor, qubits_idx: Optional[Union[Iterable[int], int, str]] = 'full', 
            num_qubits: Optional[int] = None, encoding_gate: Optional[str] = None,
    ) -> None:
        if num_qubits is None:
            num_qubits = max(qubits_idx)
        super().__init__()
        self.num_qubits = num_qubits
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        
        if encoding_gate == 'rx':
            self.encoding_gate = functional.rx
        elif encoding_gate == 'ry':
            self.encoding_gate = functional.ry
        elif encoding_gate == 'rz':
            self.encoding_gate = functional.rz
        self.encoding_gate_name = encoding_gate
        
        feature = paddle.cast(feature, _get_float_dtype(paddle_quantum.get_dtype()))
        feature = paddle.flatten(feature)
        self.feature = feature
        
        self.gate_name = 'AngleEnc'

    def forward(
            self, state: 'paddle_quantum.State' = None, invert: bool = False
    ) -> 'paddle_quantum.State':
        gate_history = []
        if state is None:
            state = paddle_quantum.state.zero_state(self.num_qubits)
        if invert:
            feature = -1 * self.feature
        else:
            feature = self.feature
        for idx, element in enumerate(feature):
            state = self.encoding_gate(
                state, element[0], self.qubits_idx[idx],
                dtype=self.dtype, backend=self.backend
            )
            gate_history.append({'gate': self.encoding_gate_name, 'which_qubits': self.qubits_idx[idx], 'theta': element[0]})
        self.gate_history = gate_history
        return state
    
    def gate_history_generation(self) -> None:
        if self.gate_history is None:
            raise RuntimeError("you must forward the encoding to receive the gate history")
        pass


class IQPEncoding(Gate):
    r"""IQP style encoding gate for encoding input classical data into quantum states.

    Args:
        feature: Vector to be encoded.
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``None``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        num_repeat: Number of encoding layers. Defaults to ``1``.
    """
    def __init__(
            self, feature: paddle.Tensor, qubits_idx: Optional[Iterable[Iterable[int]]] = None,
            num_qubits: Optional[int] = None, num_repeat: Optional[int] = 1,
    ) -> None:
        super().__init__()
        self.qubits_idx = [list(idx) for idx in qubits_idx]
        self.num_repeat = num_repeat
        self.num_qubits = num_qubits
        if feature is not None:
            feature = paddle.cast(feature, _get_float_dtype(paddle_quantum.get_dtype()))
            feature = paddle.flatten(feature)
            self.feature = feature
            
        self.gate_name = 'IQPEnc'

    def forward(
            self, state: paddle_quantum.State = None, invert: Optional[bool] = False
    ) -> paddle_quantum.State:
        gate_history = []
        if state is None:
            state = paddle_quantum.state.zero_state(self.num_qubits)
        for _ in range(0, self.num_repeat):
            if invert:
                for qubits_idx in self.qubits_idx:
                    state = functional.cnot(state, qubits_idx, dtype=self.dtype, backend=self.backend)
                    state = functional.rz(
                        state, -self.feature[qubits_idx[0]] * self.feature[qubits_idx[1]], qubits_idx[1],
                        dtype=self.dtype, backend=self.backend
                    )
                    state = functional.cnot(state, qubits_idx, dtype=self.dtype, backend=self.backend)
                    
                    gate_history.append({'gate': 'cnot', 'which_qubits': qubits_idx, 'theta': None})
                    gate_history.append({'gate': 'rz', 'which_qubits': qubits_idx[1], 
                                         'theta': -self.feature[qubits_idx[0]] * self.feature[qubits_idx[1]]})
                    gate_history.append({'gate': 'cnot', 'which_qubits': qubits_idx, 'theta': None})
                    
                for idx in range(0, self.feature.size):
                    state = functional.rz(state, -self.feature[idx], idx, dtype=self.dtype, backend=self.backend)
                    gate_history.append({'gate': 'rz', 'which_qubits': idx, 'theta': -self.feature[idx]})
                    
                for idx in range(0, self.feature.size):
                    state = functional.h(state, idx, dtype=self.dtype, backend=self.backend)
                    gate_history.append({'gate': 'h', 'which_qubits': idx, 'theta': None})
            else:
                for idx in range(0, self.feature.size):
                    state = functional.h(state, idx, dtype=self.dtype, backend=self.backend)
                    gate_history.append({'gate': 'h', 'which_qubits': idx, 'theta': None})
                    
                for idx in range(0, self.feature.size):
                    state = functional.rz(state, self.feature[idx], idx, dtype=self.dtype, backend=self.backend)
                    gate_history.append({'gate': 'rz', 'which_qubits': idx, 'theta': self.feature[idx]})
                    
                for qubits_idx in self.qubits_idx:
                    state = functional.cnot(state, qubits_idx, dtype=self.dtype, backend=self.backend)
                    state = functional.rz(
                        state, self.feature[qubits_idx[0]] * self.feature[qubits_idx[1]], qubits_idx[1],
                        dtype=self.dtype, backend=self.backend
                    )
                    state = functional.cnot(state, qubits_idx, dtype=self.dtype, backend=self.backend)
                    
                    gate_history.append({'gate': 'cnot', 'which_qubits': qubits_idx, 'theta': None})
                    gate_history.append({'gate': 'rz', 'which_qubits': qubits_idx[1], 
                                         'theta': self.feature[qubits_idx[0]] * self.feature[qubits_idx[1]]})
                    gate_history.append({'gate': 'cnot', 'which_qubits': qubits_idx, 'theta': None})
                    
        self.gate_history = gate_history
        return state

    def gate_history_generation(self) -> None:
        if self.gate_history is None:
            raise RuntimeError("you must forward the encoding to receive the gate history")
        pass