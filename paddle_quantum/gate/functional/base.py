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
The source file of the basic function for quantum gates.
"""

import copy
import paddle
import paddle_quantum as pq
from ...backend import state_vector
from ...backend import density_matrix
from ...backend import unitary_matrix
from typing import Union, List


def simulation(state: pq.State, gate: List[paddle.Tensor], qubit_idx: Union[int, List[int]]) -> pq.State:
    r"""Apply the gate on the input state.

    Args:
        state: Input state.
        gate: List of Gates to be executed.
        qubit_idx: Indices of the qubits on which the gate is applied.

    Returns:
        Output state.
    """
    if isinstance(qubit_idx, int):
        qubit_idx = [qubit_idx]
    
    if state.is_swap_back:
        if len(gate) > 1:
            for gate_i in range(len(gate)):
                state = __simulation_with_swapback(
                    state, gate[gate_i], qubit_idx[gate_i])
        else:
            state = __simulation_with_swapback(state, gate[0], qubit_idx)
        return state

    return __simulation_without_swapback(state, gate, qubit_idx)


def __simulation_with_swapback(
        state: pq.State, gate: paddle.Tensor, qubit_idx: List[int]
) -> pq.State:
    r"""Apply the gate on the input state (old logic).

    Args:
        state: Input state.
        gate: Gate to be executed.
        qubit_idx: Indices of the qubits on which the gate is applied.

    Returns:
        Output state.
    """
    num_qubits, backend = state.num_qubits, state.backend
    data = state.data

    if backend == pq.Backend.StateVector:
        data = state_vector.unitary_transformation(
            data, gate, qubit_idx, num_qubits)
    elif backend == pq.Backend.DensityMatrix:
        data = density_matrix.unitary_transformation(
            data, gate, qubit_idx, num_qubits)
    elif backend == pq.Backend.UnitaryMatrix:
        data = unitary_matrix.unitary_transformation(
            data, gate, qubit_idx, num_qubits)
    else:
        raise NotImplementedError

    state = copy.copy(state)
    state.data = data
    return state


def __simulation_without_swapback(
        state: pq.State, gate: List[paddle.Tensor], qubit_idx: List[int]
) -> pq.State:
    r"""Apply the gate on the input state (new logic).

    Args:
        state: Input state.
        gate: List of Gates to be executed.
        qubit_idx: Indices of the qubits on which the gate is applied.

    Returns:
        Output state.
    """
    num_qubits, backend = state.num_qubits, state.backend
    data = state.data
    qubit_sequence = state.qubit_sequence

    if backend == pq.Backend.StateVector:
        data, qubit_sequence = state_vector.unitary_transformation_without_swapback(
            data, gate, qubit_idx, num_qubits, qubit_sequence)
    elif backend == pq.Backend.DensityMatrix:
        data, qubit_sequence = density_matrix.unitary_transformation_without_swapback(
            data, gate, qubit_idx, num_qubits, qubit_sequence)
    else:
        if len(gate) > 1:
            for gate_i in range(len(gate)):
                data = unitary_matrix.unitary_transformation(
                    data, gate[gate_i], qubit_idx[gate_i], num_qubits)
        else:
            data = unitary_matrix.unitary_transformation(
                data, gate[0], qubit_idx, num_qubits)

    state.data = data
    state.qubit_sequence = qubit_sequence
    return state
