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
The source file of the functions for multi-qubit quantum gates.
"""

import math
import paddle
import paddle_quantum
from paddle_quantum.intrinsic import _zero, _one
from .single_qubit_gate import h, s, ry, rz, u3
from .base import simulation


def cnot(state: paddle_quantum.State, qubit_idx: list, dtype: str, backend: paddle_quantum.Backend) -> paddle_quantum.State:
    r"""Apply a CNOT gate on the input state.

    Args:
        state: Input state.
        qubit_idx: Indices of the qubits on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    gate = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ]
    gate = paddle.to_tensor(gate, dtype=dtype)
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


cx = cnot


def cy(state: paddle_quantum.State, qubit_idx: list, dtype: str, backend: paddle_quantum.Backend) -> paddle_quantum.State:
    r"""Apply a controlled Y gate on the input state.

    Args:
        state: Input state.
        qubit_idx: Indices of the qubits on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    gate = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, -1j],
        [0, 0, 1j, 0],
    ]
    gate = paddle.to_tensor(gate, dtype=dtype)
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def cz(state: paddle_quantum.State, qubit_idx: list, dtype: str, backend: paddle_quantum.Backend) -> paddle_quantum.State:
    r"""Apply a controlled Z gate on the input state.

    Args:
        state: Input state.
        qubit_idx: Indices of the qubits on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    gate = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ]
    gate = paddle.to_tensor(gate, dtype=dtype)
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def swap(state: paddle_quantum.State, qubit_idx: list, dtype: str, backend: paddle_quantum.Backend) -> paddle_quantum.State:
    r"""Apply a SWAP gate on the input state.

    Args:
        state: Input state.
        qubit_idx: Indices of the qubits on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    gate = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
    gate = paddle.to_tensor(gate, dtype=dtype)
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def cp(state: paddle_quantum.State, theta: paddle.Tensor, qubit_idx: list, dtype: str, backend: paddle_quantum.Backend) -> paddle_quantum.State:
    r"""Apply a controlled P gate on the input state.

    Args:
        state: Input state.
        theta: Parameter of the gate.
        qubit_idx: Indices of the qubits on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    gate = [
        _one(dtype), _zero(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _one(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _zero(dtype), _one(dtype), _zero(dtype),
        _zero(dtype), _zero(dtype), _zero(dtype), paddle.cos(theta).cast(dtype) + 1j * paddle.sin(theta).cast(dtype),
    ]
    gate = paddle.reshape(paddle.concat(gate), [4, 4])
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def crx(state: paddle_quantum.State, theta: paddle.Tensor, qubit_idx: list, dtype: str, backend: paddle_quantum.Backend) -> paddle_quantum.State:
    r"""Apply a controlled rotation gate about the x-axis on the input state.

    Args:
        state: Input state.
        theta: Parameter of the gate.
        qubit_idx: Indices of the qubits on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    gate = [
        _one(dtype), _zero(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _one(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _zero(dtype), paddle.cos(theta / 2).cast(dtype), -1j * paddle.sin(theta / 2).cast(dtype),
        _zero(dtype), _zero(dtype), -1j * paddle.sin(theta / 2).cast(dtype), paddle.cos(theta / 2).cast(dtype),
    ]
    gate = paddle.reshape(paddle.concat(gate), [4, 4])
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def cry(state: paddle_quantum.State, theta: paddle.Tensor, qubit_idx: list, dtype: str, backend: paddle_quantum.Backend) -> paddle_quantum.State:
    r"""Apply a controlled rotation gate about the y-axis on the input state.

    Args:
        state: Input state.
        theta: Parameter of the gate.
        qubit_idx: Indices of the qubits on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    gate = [
        _one(dtype), _zero(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _one(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _zero(dtype), paddle.cos(theta / 2).cast(dtype), (-paddle.sin(theta / 2)).cast(dtype),
        _zero(dtype), _zero(dtype), paddle.sin(theta / 2).cast(dtype), paddle.cos(theta / 2).cast(dtype),
    ]
    gate = paddle.reshape(paddle.concat(gate), [4, 4])
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def crz(state: paddle_quantum.State, theta: paddle.Tensor, qubit_idx: list, dtype: str, backend: paddle_quantum.Backend) -> paddle_quantum.State:
    r"""Apply a controlled rotation gate about the z-axis on the input state.

    Args:
        state: Input state.
        theta: Parameter of the gate.
        qubit_idx: Indices of the qubits on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    param1 = paddle.cos(theta / 2).cast(dtype) - 1j * paddle.sin(theta / 2).cast(dtype)
    param2 = paddle.cos(theta / 2).cast(dtype) + 1j * paddle.sin(theta / 2).cast(dtype)
    gate = [
        _one(dtype), _zero(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _one(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _zero(dtype), param1, _zero(dtype),
        _zero(dtype), _zero(dtype), _zero(dtype), param2,
    ]
    gate = paddle.reshape(paddle.concat(gate), [4, 4])
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def cu(state: paddle_quantum.State, theta: paddle.Tensor, qubit_idx: list, dtype: str, backend: paddle_quantum.Backend) -> paddle_quantum.State:
    r"""Apply a controlled single-qubit rotation gate on the input state.

    Args:
        state: Input state.
        theta: Parameters of the gate.
        qubit_idx: Indices of the qubits on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    # theta is theta[0], phi is theta[1], lambda is theta[2]
    param1 = paddle.cos(theta[0] / 2).cast(dtype)
    param2 = (paddle.cos(theta[2]).cast(dtype) + 1j * paddle.sin(theta[2]).cast(dtype)) * \
             (-paddle.sin(theta[0] / 2)).cast(dtype)
    param3 = (paddle.cos(theta[1]).cast(dtype) + 1j * paddle.sin(theta[1]).cast(dtype)) * \
             paddle.sin(theta[0] / 2).cast(dtype)
    param4 = (paddle.cos(theta[1] + theta[2]).cast(dtype) + 1j * paddle.sin(theta[1] + theta[2]).cast(dtype)) * \
             paddle.cos(theta[0] / 2).cast(dtype)
    gate = [
        _one(dtype), _zero(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _one(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _zero(dtype), param1, param2,
        _zero(dtype), _zero(dtype), param3, param4,
    ]
    gate = paddle.reshape(paddle.concat(gate), [4, 4])
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def rxx(state: paddle_quantum.State, theta: paddle.Tensor, qubit_idx: list, dtype: str, backend: paddle_quantum.Backend) -> paddle_quantum.State:
    r"""Apply an RXX gate on the input state.

    Args:
        state: Input state.
        theta: Parameter of the gate.
        qubit_idx: Indices of the qubits on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    param1 = paddle.cos(theta / 2).cast(dtype)
    param2 = -1j * paddle.sin(theta / 2).cast(dtype)
    gate = [
        param1, _zero(dtype), _zero(dtype), param2,
        _zero(dtype), param1, param2, _zero(dtype),
        _zero(dtype), param2, param1, _zero(dtype),
        param2, _zero(dtype), _zero(dtype), param1,
    ]
    gate = paddle.reshape(paddle.concat(gate), [4, 4])
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def ryy(state: paddle_quantum.State, theta: paddle.Tensor, qubit_idx: list, dtype: str, backend: paddle_quantum.Backend) -> paddle_quantum.State:
    r"""Apply an RYY gate on the input state.

    Args:
        state: Input state.
        theta: Parameter of the gate.
        qubit_idx: Indices of the qubits on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    param1 = paddle.cos(theta / 2).cast(dtype)
    param2 = -1j * paddle.sin(theta / 2).cast(dtype)
    param3 = 1j * paddle.sin(theta / 2).cast(dtype)
    gate = [
        param1, _zero(dtype), _zero(dtype), param3,
        _zero(dtype), param1, param2, _zero(dtype),
        _zero(dtype), param2, param1, _zero(dtype),
        param3, _zero(dtype), _zero(dtype), param1,
    ]
    gate = paddle.reshape(paddle.concat(gate), [4, 4])
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def rzz(state: paddle_quantum.State, theta: paddle.Tensor, qubit_idx: list, dtype: str, backend: paddle_quantum.Backend) -> paddle_quantum.State:
    r"""Apply an RZZ gate on the input state.

    Args:
        state: Input state.
        theta: Parameter of the gate.
        qubit_idx: Indices of the qubits on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    param1 = paddle.cos(theta / 2).cast(dtype) - 1j * paddle.sin(theta / 2).cast(dtype)
    param2 = paddle.cos(theta / 2).cast(dtype) + 1j * paddle.sin(theta / 2).cast(dtype)
    gate = [
        param1, _zero(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), param2, _zero(dtype), _zero(dtype),
        _zero(dtype), _zero(dtype), param2, _zero(dtype),
        _zero(dtype), _zero(dtype), _zero(dtype), param1,
    ]
    gate = paddle.reshape(paddle.concat(gate), [4, 4])
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def ms(state: paddle_quantum.State, qubit_idx: list, dtype: str, backend: paddle_quantum.Backend) -> paddle_quantum.State:
    r"""Apply a Mølmer-Sørensen (MS) gate on the input state.

    Args:
        state: Input state.
        qubit_idx: Indices of the qubits on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    val1 = 1 / math.sqrt(2)
    val2 = 1j / math.sqrt(2)
    gate = [
        [val1, 0, 0, val2],
        [0, val1, val2, 0],
        [0, val2, val1, 0],
        [val2, 0, 0, val1],
    ]
    gate = paddle.to_tensor(gate, dtype=dtype)
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def cswap(state: paddle_quantum.State, qubit_idx: list, dtype: str, backend: paddle_quantum.Backend) -> paddle_quantum.State:
    r"""Apply a CSWAP (Fredkin) gate on the input state.

    Args:
        state: Input state.
        qubit_idx: Indices of the qubits on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    gate = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
    gate = paddle.to_tensor(gate, dtype=dtype)
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def toffoli(state: paddle_quantum.State, qubit_idx: list, dtype: str, backend: paddle_quantum.Backend) -> paddle_quantum.State:
    r"""Apply a Toffoli gate on the input state.

    Args:
        state: Input state.
        qubit_idx: Indices of the qubits on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    gate = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ]
    gate = paddle.to_tensor(gate, dtype=dtype)
    state_data = simulation(state, gate, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def universal_two_qubits(
        state: paddle_quantum.State, theta: paddle.Tensor, qubit_idx: list, dtype: str, backend: paddle_quantum.Backend
) -> paddle_quantum.State:
    r"""Apply a universal two-qubit gate on the input state.

    Args:
        state: Input state.
        theta: Parameters of the gate.
        qubit_idx: Indices of the qubits on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    idx0, idx1 = qubit_idx
    state = u3(state, theta[[0, 1, 2]], idx0, dtype, backend)
    state = u3(state, theta[[3, 4, 5]], idx1, dtype, backend)
    state = cnot(state, [idx1, idx0], dtype, backend)
    state = rz(state, theta[6], idx0, dtype, backend)
    state = ry(state, theta[7], idx1, dtype, backend)
    state = cnot(state, [idx0, idx1], dtype, backend)
    state = ry(state, theta[8], idx1, dtype, backend)
    state = cnot(state, [idx1, idx0], dtype, backend)
    state = u3(state, theta[[9, 10, 11]], idx0, dtype, backend)
    state = u3(state, theta[[12, 13, 14]], idx1, dtype, backend)
    return state


def universal_three_qubits(
        state: paddle_quantum.State, theta: paddle.Tensor, qubit_idx: list, dtype: str, backend: paddle_quantum.Backend
) -> paddle_quantum.State:
    r"""Apply a universal three-qubit gate on the input state.

    Args:
        state: Input state.
        theta: Parameters of the gate.
        qubit_idx: Indices of the qubits on which the gate is applied.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    idx0, idx1, idx2 = qubit_idx
    psi = paddle.reshape(theta[:60], shape=[4, 15])
    phi = paddle.reshape(theta[60:], shape=[7, 3])

    def __block_u(_state: paddle_quantum.State, _theta: paddle.Tensor) -> paddle_quantum.State:
        _state = cnot(_state, [idx1, idx2], dtype, backend)
        _state = ry(_state, _theta[0], idx1, dtype, backend)
        _state = cnot(_state, [idx0, idx1], dtype, backend)
        _state = ry(_state, _theta[1], idx1, dtype, backend)
        _state = cnot(_state, [idx0, idx1], dtype, backend)
        _state = cnot(_state, [idx1, idx2], dtype, backend)
        _state = h(_state, idx2, dtype, backend)
        _state = cnot(_state, [idx1, idx0], dtype, backend)
        _state = cnot(_state, [idx0, idx2], dtype, backend)
        _state = cnot(_state, [idx1, idx2], dtype, backend)
        _state = rz(_state, _theta[2], idx2, dtype, backend)
        _state = cnot(_state, [idx1, idx2], dtype, backend)
        _state = cnot(_state, [idx0, idx2], dtype, backend)
        return _state

    def __block_v(_state: paddle_quantum.State, _theta: paddle.Tensor) -> paddle_quantum.State:
        _state = cnot(_state, [idx2, idx0], dtype, backend)
        _state = cnot(_state, [idx1, idx2], dtype, backend)
        _state = cnot(_state, [idx2, idx1], dtype, backend)
        _state = ry(_state, _theta[0], idx2, dtype, backend)
        _state = cnot(_state, [idx1, idx2], dtype, backend)
        _state = ry(_state, _theta[1], idx2, dtype, backend)
        _state = cnot(_state, [idx1, idx2], dtype, backend)
        _state = s(_state, idx2, dtype, backend)
        _state = cnot(_state, [idx2, idx0], dtype, backend)
        _state = cnot(_state, [idx0, idx1], dtype, backend)
        _state = cnot(_state, [idx1, idx0], dtype, backend)
        _state = h(_state, idx2, dtype, backend)
        _state = cnot(_state, [idx0, idx2], dtype, backend)
        _state = rz(_state, _theta[2], idx2, dtype, backend)
        _state = cnot(_state, [idx0, idx2], dtype, backend)
        return _state

    state = universal_two_qubits(state, psi[0], [idx0, idx1], dtype, backend)
    state = u3(state, phi[0, 0:3], idx2, dtype, backend)
    state = __block_u(state, phi[1])
    state = universal_two_qubits(state, psi[1], [idx0, idx1], dtype, backend)
    state = u3(state, phi[2, 0:3], idx2, dtype, backend)
    state = __block_v(state, phi[3])
    state = universal_two_qubits(state, psi[2], [idx0, idx1], dtype, backend)
    state = u3(state, phi[4, 0:3], idx2, dtype, backend)
    state = __block_u(state, phi[5])
    state = universal_two_qubits(state, psi[3], [idx0, idx1], dtype, backend)
    state = u3(state, phi[6, 0:3], idx2, dtype, backend)
    return state


def oracle(
        state: paddle_quantum.State, oracle: 'paddle.Tensor', qubit_idx: list, backend: paddle_quantum.Backend
) -> paddle_quantum.State:
    r"""Apply an oracle gate on the input state.

    Args:
        state: Input state.
        oracle: Oracle to be executed.
        qubit_idx: Indices of the qubits on which the gate is applied.
        backend: Backend on which the simulation is run.

    Returns:
        Output state.
    """
    state_data = simulation(state, oracle, qubit_idx, state.num_qubits, backend)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state
