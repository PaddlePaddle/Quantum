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
The source file of the functions for single-qubit quantum gates.
"""

import math
import paddle
import paddle_quantum
from .base import simulation
from paddle_quantum.intrinsic import _get_float_dtype, _zero, _one


def h(state: paddle_quantum.State, qubit_idx: int, dtype: str, backend: paddle_quantum.Backend) -> paddle_quantum.State:
    r"""Apply a Hadamard gate on the input state.

    Args:
        state: Input state.
        qubit_idx: Index of the qubit on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    element = 1 / math.sqrt(2)
    gate = [
        [element, element],
        [element, -element],
    ]
    gate = paddle.to_tensor(gate, dtype=dtype)
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def s(state: paddle_quantum.State, qubit_idx: int, dtype: str, backend: paddle_quantum.Backend) -> paddle_quantum.State:
    r"""Apply an S gate on the input state.

    Args:
        state: Input state.
        qubit_idx: Index of the qubit on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    gate = [
        [1, 0],
        [0, 1j],
    ]
    gate = paddle.to_tensor(gate, dtype=dtype)
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def t(state: paddle_quantum.State, qubit_idx: int, dtype: str, backend: paddle_quantum.Backend) -> paddle_quantum.State:
    r"""Apply a T gate on the input state.

    Args:
        state: Input state.
        qubit_idx: Index of the qubit on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    gate = [
        [1, 0],
        [0, math.cos(math.pi / 4) + math.sin(math.pi / 4) * 1j],
    ]
    gate = paddle.to_tensor(gate, dtype=dtype)
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def x(state: paddle_quantum.State, qubit_idx: int, dtype: str, backend: paddle_quantum.Backend) -> paddle_quantum.State:
    r"""Apply an X gate on the input state.

    Args:
        state: Input state.
        qubit_idx: Index of the qubit on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    gate = [
        [0, 1],
        [1, 0],
    ]
    gate = paddle.to_tensor(gate, dtype=dtype)
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def y(state: paddle_quantum.State, qubit_idx: int, dtype: str, backend: paddle_quantum.Backend) -> paddle_quantum.State:
    r"""Apply a Y gate on the input state.

    Args:
        state: Input state.
        qubit_idx: Index of the qubit on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    gate = [
        [0, -1j],
        [1j, 0],
    ]
    gate = paddle.to_tensor(gate, dtype=dtype)
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def z(state: paddle_quantum.State, qubit_idx: int, dtype: str, backend: paddle_quantum.Backend) -> paddle_quantum.State:
    r"""Apply a Z gate on the input state.

    Args:
        state: Input state.
        qubit_idx: Index of the qubit on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    gate = [
        [1, 0],
        [0, -1],
    ]
    gate = paddle.to_tensor(gate, dtype=dtype)
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def p(
        state: paddle_quantum.State, theta: paddle.Tensor, qubit_idx: int, dtype: str, backend: paddle_quantum.Backend
) -> paddle_quantum.State:
    r"""Apply a P gate on the input state.

    Args:
        state: Input state.
        theta: Parameter of the gate.
        qubit_idx: Index of the qubit on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    gate = [
        _one(dtype), _zero(dtype),
        _zero(dtype), paddle.cos(theta).cast(dtype) + 1j * paddle.sin(theta).cast(dtype),
    ]
    gate = paddle.reshape(paddle.concat(gate), [2, 2])
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def rx(
        state: paddle_quantum.State, theta: paddle.Tensor, qubit_idx: int, dtype: str, backend: paddle_quantum.Backend
) -> paddle_quantum.State:
    r"""Apply a single-qubit rotation gate about the x-axis on the input state.

    Args:
        state: Input state.
        theta: Parameter of the gate.
        qubit_idx: Index of the qubit on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    gate = [
        paddle.cos(theta / 2).cast(dtype), -1j * paddle.sin(theta / 2).cast(dtype),
        -1j * paddle.sin(theta / 2).cast(dtype), paddle.cos(theta / 2).cast(dtype),
    ]
    gate = paddle.reshape(paddle.concat(gate), [2, 2])
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def ry(
        state: paddle_quantum.State, theta: paddle.Tensor, qubit_idx: int, dtype: str, backend: paddle_quantum.Backend
) -> paddle_quantum.State:
    r"""Apply a single-qubit rotation gate about the y-axis on the input state.

    Args:
        state: Input state.
        theta: Parameter of the gate.
        qubit_idx: Index of the qubit on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    gate = [
        paddle.cos(theta / 2).cast(dtype), (-paddle.sin(theta / 2)).cast(dtype),
        paddle.sin(theta / 2).cast(dtype), paddle.cos(theta / 2).cast(dtype),
    ]
    gate = paddle.reshape(paddle.concat(gate), [2, 2])
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def rz(
        state: paddle_quantum.State, theta: paddle.Tensor, qubit_idx: int, dtype: str, backend: paddle_quantum.Backend
) -> paddle_quantum.State:
    r"""Apply a single-qubit rotation gate about the z-axis on the input state.

    Args:
        state: Input state.
        theta: Parameter of the gate.
        qubit_idx: Index of the qubit on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    gate = [
        paddle.cos(theta / 2).cast(dtype) - 1j * paddle.sin(theta / 2).cast(dtype), _zero(dtype),
        _zero(dtype), paddle.cos(theta / 2).cast(dtype) + 1j * paddle.sin(theta / 2).cast(dtype),
    ]
    gate = paddle.reshape(paddle.concat(gate), [2, 2])
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def u3(
        state: paddle_quantum.State, theta: paddle.Tensor, qubit_idx: int, dtype: str, backend: paddle_quantum.Backend
) -> paddle_quantum.State:
    r"""Apply a single-qubit rotation gate on the input state.

    Args:
        state: Input state.
        theta: Parameters of the gate.
        qubit_idx: Index of the qubit on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    # theta is theta[0], phi is theta[1], lambda is theta[2]
    gate_real = [
        paddle.cos(theta[0] / 2),
        -paddle.cos(theta[2]) * paddle.sin(theta[0] / 2),
        paddle.cos(theta[1]) * paddle.sin(theta[0] / 2),
        paddle.cos(theta[1] + theta[2]) * paddle.cos(theta[0] / 2),
    ]
    gate_real = paddle.reshape(paddle.concat(gate_real), [2, 2])

    gate_imag = [
        paddle.to_tensor(0, dtype=_get_float_dtype(dtype)),
        -paddle.sin(theta[2]) * paddle.sin(theta[0] / 2),
        paddle.sin(theta[1]) * paddle.sin(theta[0] / 2),
        paddle.sin(theta[1] + theta[2]) * paddle.cos(theta[0] / 2),
    ]
    gate_imag = paddle.reshape(paddle.concat(gate_imag), [2, 2])

    gate = gate_real + 1j * gate_imag
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state
