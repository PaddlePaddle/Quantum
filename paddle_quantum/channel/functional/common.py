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
The source file of the various quantum channels.
"""

import functools
import paddle
import paddle_quantum
from ...backend import density_matrix
from ...intrinsic import _zero, _one
from typing import Iterable


def bit_flip(
        state: paddle_quantum.State, prob: paddle.Tensor, qubit_idx: int, dtype: str, backend: paddle_quantum.Backend
) -> paddle_quantum.State:
    r"""Apply a bit flip channel on the input state.

    Args:
        state: Input state.
        prob: Probability of a bit flip.
        qubit_idx: Index of the qubit on which the channel acts.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Raises:
        RuntimeError: The noisy channel can only run in density matrix mode.

    Returns:
        Output state.
    """
    if backend != paddle_quantum.Backend.DensityMatrix:
        raise RuntimeError("The noisy channel can only run in density matrix mode.")
    kraus_oper = [
        [
            paddle.sqrt(1 - prob).cast(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(1 - prob).cast(dtype),
        ],
        [
            _zero(dtype), paddle.sqrt(prob).cast(dtype),
            paddle.sqrt(prob).cast(dtype), _zero(dtype),
        ]
    ]
    for idx, oper in enumerate(kraus_oper):
        kraus_oper[idx] = paddle.reshape(paddle.concat(oper), [2, 2])
    state_data = [
        density_matrix.unitary_transformation(state.data, oper, [qubit_idx], state.num_qubits) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def phase_flip(
        state: paddle_quantum.State, prob: paddle.Tensor, qubit_idx: int, dtype: str, backend: paddle_quantum.Backend
) -> paddle_quantum.State:
    r"""Apply a phase flip channel on the input state.

    Args:
        state: Input state.
        prob: Probability of a phase flip.
        qubit_idx: Index of the qubit on which the channel acts.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Raises:
        RuntimeError: The noisy channel can only run in density matrix mode.

    Returns:
        Output state.
    """
    if backend != paddle_quantum.Backend.DensityMatrix:
        raise RuntimeError("The noisy channel can only run in density matrix mode.")
    kraus_oper = [
        [
            paddle.sqrt(1 - prob).cast(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(1 - prob).cast(dtype),
        ],
        [
            paddle.sqrt(prob).cast(dtype), _zero(dtype),
            _zero(dtype), (-paddle.sqrt(prob)).cast(dtype),
        ]
    ]
    for idx, oper in enumerate(kraus_oper):
        kraus_oper[idx] = paddle.reshape(paddle.concat(oper), [2, 2])
    state_data = [
        density_matrix.unitary_transformation(state.data, oper, [qubit_idx], state.num_qubits) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def bit_phase_flip(
        state: paddle_quantum.State, prob: paddle.Tensor, qubit_idx: int, dtype: str, backend: paddle_quantum.Backend
) -> paddle_quantum.State:
    r"""Apply a bit phase flip channel on the input state.

    Args:
        state: Input state.
        prob: Probability of a bit phase flip.
        qubit_idx: Index of the qubit on which the channel acts.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Raises:
        RuntimeError: The noisy channel can only run in density matrix mode.

    Returns:
        Output state.
    """
    if backend != paddle_quantum.Backend.DensityMatrix:
        raise RuntimeError("The noisy channel can only run in density matrix mode.")
    kraus_oper = [
        [
            paddle.sqrt(1 - prob).cast(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(1 - prob).cast(dtype),
        ],
        [
            _zero(dtype), -1j * paddle.sqrt(prob),
            1j * -paddle.sqrt(prob), _zero(dtype),
        ]
    ]
    for idx, oper in enumerate(kraus_oper):
        kraus_oper[idx] = paddle.reshape(paddle.concat(oper), [2, 2])
    state_data = [
        density_matrix.unitary_transformation(state.data, oper, [qubit_idx], state.num_qubits) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def amplitude_damping(
        state: paddle_quantum.State, gamma: paddle.Tensor, qubit_idx: int, dtype: str, backend: paddle_quantum.Backend
) -> paddle_quantum.State:
    r"""Apply an amplitude damping channel on the input state.

    Args:
        state: Input state.
        gamma: Damping probability.
        qubit_idx: Index of the qubit on which the channel acts.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Raises:
        RuntimeError: The noisy channel can only run in density matrix mode.

    Returns:
        Output state.
    """
    if backend != paddle_quantum.Backend.DensityMatrix:
        raise RuntimeError("The noisy channel can only run in density matrix mode.")
    kraus_oper = [
        [
            _one(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(1 - gamma).cast(dtype),
        ],
        [
            _zero(dtype), paddle.sqrt(gamma).cast(dtype),
            _zero(dtype), _zero(dtype)],

    ]
    for idx, oper in enumerate(kraus_oper):
        kraus_oper[idx] = paddle.reshape(paddle.concat(oper), [2, 2])
    state_data = [
        density_matrix.unitary_transformation(state.data, oper, [qubit_idx], state.num_qubits) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def generalized_amplitude_damping(
        state: paddle_quantum.State, gamma: paddle.Tensor, prob: paddle.Tensor,
        qubit_idx: int, dtype: str, backend: paddle_quantum.Backend
) -> paddle_quantum.State:
    r"""Apply a generalized amplitude damping channel on the input state.

    Args:
        state: Input state.
        gamma: Damping probability.
        prob: Excitation probability.
        qubit_idx: Index of the qubit on which the channel acts.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Raises:
        RuntimeError: The noisy channel can only run in density matrix mode.

    Returns:
        Output state.
    """
    if backend != paddle_quantum.Backend.DensityMatrix:
        raise RuntimeError("The noisy channel can only run in density matrix mode.")
    kraus_oper = [
        [
            paddle.sqrt(prob).cast(dtype), _zero(dtype),
            _zero(dtype), (paddle.sqrt(prob).cast(dtype) * paddle.sqrt(1 - gamma)).cast(dtype),
        ],
        [
            _zero(dtype), (paddle.sqrt(prob) * paddle.sqrt(gamma)).cast(dtype),
            _zero(dtype), _zero(dtype),
        ],
        [
            (paddle.sqrt(1 - prob) * paddle.sqrt(1 - gamma)).cast(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(1 - prob).cast(dtype),
        ],
        [
            _zero(dtype), _zero(dtype),
            (paddle.sqrt(1 - prob) * paddle.sqrt(gamma)).cast(dtype), _zero(dtype),
        ],
    ]
    for idx, oper in enumerate(kraus_oper):
        kraus_oper[idx] = paddle.reshape(paddle.concat(oper), [2, 2])
    state_data = [
        density_matrix.unitary_transformation(state.data, oper, [qubit_idx], state.num_qubits) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def phase_damping(
        state: paddle_quantum.State, gamma: paddle.Tensor, qubit_idx: int, dtype: str, backend: paddle_quantum.Backend
) -> paddle_quantum.State:
    r"""Apply a phase damping channel on the input state.

    Args:
        state: Input state.
        gamma: Parameter of the phase damping channel.
        qubit_idx: Index of the qubit on which the channel acts.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Raises:
        RuntimeError: The noisy channel can only run in density matrix mode.

    Returns:
        Output state.
    """
    if backend != paddle_quantum.Backend.DensityMatrix:
        raise RuntimeError("The noisy channel can only run in density matrix mode.")
    kraus_oper = [
        [
            _one(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(1 - gamma).cast(dtype),
        ],
        [
            _zero(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(gamma).cast(dtype),
        ]
    ]
    for idx, oper in enumerate(kraus_oper):
        kraus_oper[idx] = paddle.reshape(paddle.concat(oper), [2, 2])
    state_data = [
        density_matrix.unitary_transformation(state.data, oper, [qubit_idx], state.num_qubits) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def depolarizing(
        state: paddle_quantum.State, prob: paddle.Tensor, qubit_idx: int, dtype: str, backend: paddle_quantum.Backend
) -> paddle_quantum.State:
    r"""Apply a depolarizing channel on the input state.

    Args:
        state: Input state.
        prob: Parameter of the depolarizing channel.
        qubit_idx: Index of the qubit on which the channel acts.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Raises:
        RuntimeError: The noisy channel can only run in density matrix mode.

    Returns:
        Output state.
    """
    if backend != paddle_quantum.Backend.DensityMatrix:
        raise RuntimeError("The noisy channel can only run in density matrix mode.")
    kraus_oper = [
        [
            paddle.sqrt(1 - prob).cast(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(1 - prob).cast(dtype),
        ],
        [
            _zero(dtype), paddle.sqrt(prob / 3).cast(dtype),
            paddle.sqrt(prob / 3).cast(dtype), _zero(dtype),
        ],
        [
            _zero(dtype), -1j * paddle.sqrt(prob / 3).cast(dtype),
            1j * paddle.sqrt(prob / 3).cast(dtype), _zero(dtype),
        ],
        [
            paddle.sqrt(prob / 3).cast(dtype), _zero(dtype),
            _zero(dtype), (-1 * paddle.sqrt(prob / 3)).cast(dtype),
        ],
    ]
    for idx, oper in enumerate(kraus_oper):
        kraus_oper[idx] = paddle.reshape(paddle.concat(oper), [2, 2])
    state_data = [
        density_matrix.unitary_transformation(state.data, oper, [qubit_idx], state.num_qubits) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def pauli_channel(
        state: paddle_quantum.State, prob: paddle.Tensor, qubit_idx: int,
        dtype: str, backend: paddle_quantum.Backend
) -> paddle_quantum.State:
    r"""Apply a Pauli channel on the input state.

    Args:
        state: Input state.
        prob: Probabilities corresponding to the Pauli X, Y, and Z operators.
        qubit_idx: Index of the qubit on which the channel acts.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Raises:
        RuntimeError: The noisy channel can only run in density matrix mode.

    Returns:
        Output state.
    """
    if backend != paddle_quantum.Backend.DensityMatrix:
        raise RuntimeError("The noisy channel can only run in density matrix mode.")
    prob_x, prob_y, prob_z = prob
    prob_i = paddle.sqrt(1 - paddle.sum(prob))
    kraus_oper = [
        [
            paddle.sqrt(prob_i).cast(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(prob_i).cast(dtype),
        ],
        [
            _zero(dtype), paddle.sqrt(prob_x).cast(dtype),
            paddle.sqrt(prob_x).cast(dtype), _zero(dtype),
        ],
        [
            _zero(dtype), -1j * paddle.sqrt(prob_y).cast(dtype),
            1j * paddle.sqrt(prob_y).cast(dtype), _zero(dtype),
        ],
        [
            paddle.sqrt(prob_z).cast(dtype), _zero(dtype),
            _zero(dtype), (-paddle.sqrt(prob_z)).cast(dtype),
        ],
    ]
    for idx, oper in enumerate(kraus_oper):
        kraus_oper[idx] = paddle.reshape(paddle.concat(oper), [2, 2])
    state_data = [
        density_matrix.unitary_transformation(state.data, oper, [qubit_idx], state.num_qubits) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def reset_channel(
        state: paddle_quantum.State, prob: paddle.Tensor, qubit_idx: int,
        dtype: str, backend: paddle_quantum.Backend
) -> paddle_quantum.State:
    r"""Apply a reset channel on the input state.

    Args:
        state: Input state.
        prob: Probabilities of resetting to :math:`|0\rangle` and to :math:`|1\rangle`.
        qubit_idx: Index of the qubit on which the channel acts.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Raises:
        RuntimeError: The noisy channel can only run in density matrix mode.

    Returns:
        Output state.
    """
    if backend != paddle_quantum.Backend.DensityMatrix:
        raise RuntimeError("The noisy channel can only run in density matrix mode.")
    prob_0, prob_1 = prob
    prob_i = 1 - paddle.sum(prob)
    kraus_oper = [
        [
            paddle.sqrt(prob_0).cast(dtype), _zero(dtype),
            _zero(dtype), _zero(dtype),
        ],
        [
            _zero(dtype), paddle.sqrt(prob_0).cast(dtype),
            _zero(dtype), _zero(dtype),
        ],
        [
            _zero(dtype), _zero(dtype),
            paddle.sqrt(prob_1).cast(dtype), _zero(dtype),
        ],
        [
            _zero(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(prob_1).cast(dtype),
        ],
        [
            paddle.sqrt(prob_i).cast(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(prob_i).cast(dtype),
        ],
    ]
    for idx, oper in enumerate(kraus_oper):
        kraus_oper[idx] = paddle.reshape(paddle.concat(oper), [2, 2])
    state_data = [
        density_matrix.unitary_transformation(state.data, oper, [qubit_idx], state.num_qubits) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def thermal_relaxation(
        state: paddle_quantum.State, const_t: paddle.Tensor, exec_time: paddle.Tensor,
        qubit_idx: int, dtype: str, backend: paddle_quantum.Backend
) -> paddle_quantum.State:
    r"""Apply a thermal relaxation channel on the input state.

    Args:
        state: Input state.
        const_t: :math:`T_1` and :math:`T_2` relaxation time in microseconds.
        exec_time: Quantum gate execution time in the process of relaxation in nanoseconds.
        qubit_idx: Index of the qubit on which the channel acts.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Raises:
        RuntimeError: The noisy channel can only run in density matrix mode.

    Returns:
        Output state.
    """
    if backend != paddle_quantum.Backend.DensityMatrix:
        raise RuntimeError("The noisy channel can only run in density matrix mode.")
    t1, t2 = const_t
    exec_time = exec_time / 1000
    prob_reset = 1 - paddle.exp(-exec_time / t1)
    prob_z = (1 - prob_reset) * (1 - paddle.exp(-exec_time / t2) * paddle.exp(exec_time / t1)) / 2
    prob_i = 1 - prob_reset - prob_z
    kraus_oper = [
        [
            paddle.sqrt(prob_i).cast(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(prob_i).cast(dtype),
        ],
        [
            paddle.sqrt(prob_z).cast(dtype), _zero(dtype),
            _zero(dtype), (-paddle.sqrt(prob_z)).cast(dtype),
        ],
        [
            paddle.sqrt(prob_reset).cast(dtype), _zero(dtype),
            _zero(dtype), _zero(dtype),
        ],
        [
            _zero(dtype), paddle.sqrt(prob_reset).cast(dtype),
            _zero(dtype), _zero(dtype),
        ],
    ]
    for idx, oper in enumerate(kraus_oper):
        kraus_oper[idx] = paddle.reshape(paddle.concat(oper), [2, 2])
    state_data = [
        density_matrix.unitary_transformation(state.data, oper, [qubit_idx], state.num_qubits) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def kraus_repr(
        state: paddle_quantum.State, kraus_oper: Iterable[paddle.Tensor], qubit_idx: int,
        dtype: str, backend: paddle_quantum.Backend
) -> paddle_quantum.State:
    r"""Apply a custom channel in the Kraus representation on the input state.

    Args:
        state: Input state.
        kraus_oper: Kraus operators of this channel.
        qubit_idx: Index of the qubit on which the channel acts.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Raises:
        RuntimeError: The noisy channel can only run in density matrix mode.

    Returns:
        Output state.
    """
    if backend != paddle_quantum.Backend.DensityMatrix:
        raise RuntimeError("The noisy channel can only run in density matrix mode.")
    kraus_oper = [paddle.cast(oper, dtype) for oper in kraus_oper]
    state_data = [
        density_matrix.unitary_transformation(state.data, oper, [qubit_idx], state.num_qubits) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def choi_repr():
    raise NotImplementedError
