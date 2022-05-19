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

import paddle
import paddle_quantum
from ...backend import state_vector
from ...backend import density_matrix
from ...backend import unitary_matrix
from typing import Union, List


def simulation(
        state: paddle_quantum.State, gate: paddle.Tensor, qubit_idx: Union[int, List[int]],
        num_qubits: int, backend: paddle_quantum.Backend
) -> 'paddle.Tensor':
    r"""Apply the gate on the input state.

    Args:
        state: Input state.
        gate: Gate to be executed.
        qubit_idx: Indices of the qubits on which the gate is applied.
        num_qubits: Total number of qubits.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    if isinstance(qubit_idx, int):
        qubit_idx = [qubit_idx]
    if backend == paddle_quantum.Backend.StateVector:
        return state_vector.unitary_transformation(state.data, gate, qubit_idx, num_qubits)
    elif backend == paddle_quantum.Backend.DensityMatrix:
        return density_matrix.unitary_transformation(state.data, gate, qubit_idx, num_qubits)
    elif backend == paddle_quantum.Backend.UnitaryMatrix:
        return unitary_matrix.unitary_transformation(state.data, gate, qubit_idx, num_qubits)
    else:
        return NotImplementedError
