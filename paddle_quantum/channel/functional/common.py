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
from typing import Iterable, List, Tuple, Union


def bit_flip(
        state: paddle_quantum.State, prob: paddle.Tensor, qubit_idx: Union[List[int], int],
        dtype: str, backend: paddle_quantum.Backend
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
        density_matrix.unitary_transformation(
            state.data,
            oper,
            qubit_idx if isinstance(qubit_idx, list) else [qubit_idx],
            state.num_qubits
        ) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def phase_flip(
        state: paddle_quantum.State, prob: paddle.Tensor, qubit_idx: Union[List[int], int],
        dtype: str, backend: paddle_quantum.Backend
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
        density_matrix.unitary_transformation(
            state.data,
            oper,
            qubit_idx if isinstance(qubit_idx, list) else [qubit_idx],
            state.num_qubits
        ) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def bit_phase_flip(
        state: paddle_quantum.State, prob: paddle.Tensor, qubit_idx: Union[List[int], int],
        dtype: str, backend: paddle_quantum.Backend
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
        density_matrix.unitary_transformation(
            state.data,
            oper,
            qubit_idx if isinstance(qubit_idx, list) else [qubit_idx],
            state.num_qubits
        ) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def amplitude_damping(
        state: paddle_quantum.State, gamma: paddle.Tensor, qubit_idx: Union[List[int], int],
        dtype: str, backend: paddle_quantum.Backend
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
        density_matrix.unitary_transformation(
            state.data,
            oper,
            qubit_idx if isinstance(qubit_idx, list) else [qubit_idx],
            state.num_qubits
        ) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def generalized_amplitude_damping(
        state: paddle_quantum.State, gamma: paddle.Tensor, prob: paddle.Tensor,
        qubit_idx: Union[List[int], int], dtype: str, backend: paddle_quantum.Backend
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
        density_matrix.unitary_transformation(
            state.data,
            oper,
            qubit_idx if isinstance(qubit_idx, list) else [qubit_idx],
            state.num_qubits
        ) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def phase_damping(
        state: paddle_quantum.State, gamma: paddle.Tensor, qubit_idx: Union[List[int], int],
        dtype: str, backend: paddle_quantum.Backend
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
        density_matrix.unitary_transformation(
            state.data,
            oper,
            qubit_idx if isinstance(qubit_idx, list) else [qubit_idx],
            state.num_qubits
        ) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def depolarizing(
        state: paddle_quantum.State, prob: paddle.Tensor, qubit_idx: Union[List[int], int],
        dtype: str, backend: paddle_quantum.Backend
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
            paddle.sqrt(1 - 3 * prob / 4).cast(dtype), _zero(dtype),
            _zero(dtype), paddle.sqrt(1 - 3 * prob / 4).cast(dtype),
        ],
        [
            _zero(dtype), paddle.sqrt(prob / 4).cast(dtype),
            paddle.sqrt(prob / 4).cast(dtype), _zero(dtype),
        ],
        [
            _zero(dtype), -1j * paddle.sqrt(prob / 4).cast(dtype),
            1j * paddle.sqrt(prob / 4).cast(dtype), _zero(dtype),
        ],
        [
            paddle.sqrt(prob / 4).cast(dtype), _zero(dtype),
            _zero(dtype), (-1 * paddle.sqrt(prob / 4)).cast(dtype),
        ],
    ]
    for idx, oper in enumerate(kraus_oper):
        kraus_oper[idx] = paddle.reshape(paddle.concat(oper), [2, 2])
    state_data = [
        density_matrix.unitary_transformation(
            state.data,
            oper,
            qubit_idx if isinstance(qubit_idx, list) else [qubit_idx],
            state.num_qubits
        ) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def pauli_channel(
        state: paddle_quantum.State, prob: paddle.Tensor, qubit_idx: Union[List[int], int],
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
        density_matrix.unitary_transformation(
            state.data,
            oper,
            qubit_idx if isinstance(qubit_idx, list) else [qubit_idx],
            state.num_qubits
        ) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def reset_channel(
        state: paddle_quantum.State, prob: paddle.Tensor, qubit_idx: Union[List[int], int],
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
        density_matrix.unitary_transformation(
            state.data,
            oper,
            qubit_idx if isinstance(qubit_idx, list) else [qubit_idx],
            state.num_qubits
        ) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def thermal_relaxation(
        state: paddle_quantum.State, const_t: paddle.Tensor, exec_time: paddle.Tensor,
        qubit_idx: Union[List[int], int], dtype: str, backend: paddle_quantum.Backend
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
        density_matrix.unitary_transformation(
            state.data,
            oper,
            qubit_idx if isinstance(qubit_idx, list) else [qubit_idx],
            state.num_qubits
        ) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def kraus_repr(
        state: paddle_quantum.State, kraus_oper: Iterable[paddle.Tensor], qubit_idx: Union[List[int], int],
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
    state_data = [
        density_matrix.unitary_transformation(
            state.data,
            oper,
            qubit_idx if isinstance(qubit_idx, list) else [qubit_idx],
            state.num_qubits
        ) for oper in kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def choi_repr(
        state: paddle_quantum.State, choi_oper: paddle.Tensor, qubit_idx: Union[List[int], int],
        dtype: str, backend: paddle_quantum.Backend
) -> paddle_quantum.State:
    r"""choi_repr implement

    Assume the choi state has the shape of sum :math:`|i\rangle\langle j|` :math:`N(|i\rangle\langle j|)` .

    Args:
        state: input quantum state
        choi_oper: choi representation for the channel
        qubit_idx: which qubits the channel acts on
        dtype: data dtype
        backend: data backend

    Raises:
        RuntimeError: _description_

    Returns:
        paddle_quantum.State: output from the channel
    """
    qubit_idx = qubit_idx if isinstance(qubit_idx, list) else [qubit_idx]

    def genSwapList(origin: List[int], target: List[int]) -> List[Tuple[int, int]]:
        assert len(origin) == len(target)
        swapped = [False] * len(origin)
        swap_ops = []

        origin_pos_dict = {v: pos for pos, v in enumerate(origin)}

        def positionOfValueAt(idx):
            # return the position of value `target[idx]` in origin array
            return origin_pos_dict[target[idx]]

        ref = origin.copy()
        for idx in range(len(origin)):
            if not swapped[idx]:
                next_idx = idx
                swapped[next_idx] = True
                while not swapped[positionOfValueAt(next_idx)]:
                    swapped[positionOfValueAt(next_idx)] = True
                    if next_idx < positionOfValueAt(next_idx):
                        swap_ops.append((next_idx, positionOfValueAt(next_idx)))
                    else:
                        swap_ops.append((positionOfValueAt(next_idx), next_idx))

                    x, y = swap_ops[-1]
                    ref[x], ref[y] = ref[y], ref[x]
                    # print(idx, (x,y), ref)
                    next_idx = positionOfValueAt(next_idx)

        return swap_ops

    if backend != paddle_quantum.Backend.DensityMatrix:
        raise RuntimeError("The noisy channel can only run in density matrix mode.")
    assert len(choi_oper) == 2 ** (2 * len(qubit_idx))

    num_qubits = state.num_qubits
    num_acted_qubits = len(qubit_idx)

    # make partial transpose on the ancilla of choi repr, this leads to choi_mat as `sum |j><i| N(|i><j|)`
    choi_mat = paddle.reshape(choi_oper, [2 ** num_acted_qubits, 2 ** num_acted_qubits,
                                          2 ** num_acted_qubits, 2 ** num_acted_qubits])
    choi_mat = paddle.transpose(choi_mat, [2, 1, 0, 3])
    choi_mat = paddle.reshape(choi_mat, [2 ** (2 * num_acted_qubits), 2 ** (2 * num_acted_qubits)])
    ext_state = paddle.kron(state.data, paddle.eye(2 ** num_acted_qubits))

    ext_qubit_idx = qubit_idx + [num_qubits + x for x in range(num_acted_qubits)]
    ext_num_qubits = num_qubits + num_acted_qubits
    higher_dims = ext_state.shape[:-2]
    num_higher_dims = len(higher_dims)

    swap_ops = genSwapList(list(range(ext_num_qubits)), ext_qubit_idx +
                           [x for x in range(ext_num_qubits) if x not in ext_qubit_idx])

    # make swap for left
    for swap_op in swap_ops:
        shape = higher_dims.copy()
        last_idx = -1
        for idx in swap_op:
            shape.append(2 ** (idx - last_idx - 1))
            shape.append(2)
            last_idx = idx
        shape.append(2 ** (2 * ext_num_qubits - last_idx - 1))
        ext_state = paddle.reshape(ext_state, shape)
        ext_state = paddle.transpose(
            ext_state, list(range(num_higher_dims)) + [item + num_higher_dims for item in [0, 3, 2, 1, 4]]
        )

    # multiply the choi_matrix
    ext_state = paddle.reshape(
        ext_state, higher_dims.copy() + [2 ** (2 * num_acted_qubits), 2 ** (2 * ext_num_qubits - 2 * num_acted_qubits)]
    )
    ext_state = paddle.reshape(
        paddle.matmul(choi_mat, ext_state), higher_dims.copy() + [2 ** ext_num_qubits, 2 ** ext_num_qubits]
    )

    # make swap for right
    for swap_op in swap_ops:
        shape = higher_dims.copy()
        last_idx = -1
        shape.append(2 ** ext_num_qubits)
        for idx in swap_op:
            shape.append(2 ** (idx - last_idx - 1))
            shape.append(2)
            last_idx = idx
        shape.append(2 ** (ext_num_qubits - last_idx - 1))
        ext_state = paddle.reshape(ext_state, shape)
        ext_state = paddle.transpose(
            ext_state, list(range(num_higher_dims)) + [item + num_higher_dims for item in [0, 1, 4, 3, 2, 5]]
        )

    # implement partial trace on ext_state
    new_state = paddle.trace(
        paddle.reshape(
            ext_state,
            higher_dims.copy() + [2 ** num_acted_qubits, 2 ** num_qubits, 2 ** num_acted_qubits, 2 ** num_qubits]
        ),
        axis1=len(higher_dims),
        axis2=2+len(higher_dims)
    )

    # swap back
    revert_swap_ops = genSwapList(qubit_idx + [x for x in range(num_qubits) if x not in qubit_idx],
                                  list(range(num_qubits)))
    for swap_op in revert_swap_ops:
        shape = higher_dims.copy()
        last_idx = -1
        for idx in swap_op:
            shape.append(2 ** (idx - last_idx - 1))
            shape.append(2)
            last_idx = idx
        shape.append(2 ** (2 * num_qubits - last_idx - 1))
        new_state = paddle.reshape(new_state, shape)
        new_state = paddle.transpose(
            new_state, list(range(num_higher_dims)) + [item + num_higher_dims for item in [0, 3, 2, 1, 4]]
        )
    for swap_op in revert_swap_ops:
        shape = higher_dims.copy()
        last_idx = -1
        shape.append(2 ** num_qubits)
        for idx in swap_op:
            shape.append(2 ** (idx - last_idx - 1))
            shape.append(2)
            last_idx = idx
        shape.append(2 ** (num_qubits - last_idx - 1))
        new_state = paddle.reshape(new_state, shape)
        new_state = paddle.transpose(
            new_state, list(range(num_higher_dims)) + [item + num_higher_dims for item in [0, 1, 4, 3, 2, 5]]
        )

    new_state = paddle.reshape(new_state, higher_dims.copy() + [2 ** num_qubits, 2 ** num_qubits])
    return paddle_quantum.State(new_state, dtype=dtype, backend=backend)


def stinespring_repr(
    state: paddle_quantum.State,
    stinespring_mat: paddle.Tensor,
    qubit_idx: Union[List[int], int],
    dtype: str,
    backend: paddle_quantum.Backend
):
    """stinespring representation for quantum channel

    assuming stinespring_mat being the rectangle matrix of shape (dim1 * dim2, dim1)
    where dim1 is the dimension of qubit_idx, dim2 needs to be partial traced. With
    Dirac notation we have the elements

        stinespring_mat.reshape([dim1, dim2, dim1])[i, j, k] = <i, j|A|k>

    with A being the stinespring operator, the channel acts as rho -> Tr_2 A rho A^dagger.

    Args:
        state: input quantum state
        stinespring_mat: Stinespring representation for the channel
        qubit_idx: which qubits the channel acts on
        dtype: data dtype
        backend: data backend

    Returns:
        paddle_quantum.State: output from the channel
    """
    qubit_idx = qubit_idx if isinstance(qubit_idx, list) else [qubit_idx]

    def genSwapList(origin: List[int], target: List[int]) -> List[Tuple[int, int]]:
        assert len(origin) == len(target)
        swapped = [False] * len(origin)
        swap_ops = []

        origin_pos_dict = {v: pos for pos, v in enumerate(origin)}

        def positionOfValueAt(idx):
            # return the position of value `target[idx]` in origin array
            return origin_pos_dict[target[idx]]

        ref = origin.copy()
        for idx in range(len(origin)):
            if not swapped[idx]:
                next_idx = idx
                swapped[next_idx] = True
                while not swapped[positionOfValueAt(next_idx)]:
                    swapped[positionOfValueAt(next_idx)] = True
                    if next_idx < positionOfValueAt(next_idx):
                        swap_ops.append((next_idx, positionOfValueAt(next_idx)))
                    else:
                        swap_ops.append((positionOfValueAt(next_idx), next_idx))

                    x, y = swap_ops[-1]
                    ref[x], ref[y] = ref[y], ref[x]
                    # print(idx, (x,y), ref)
                    next_idx = positionOfValueAt(next_idx)

        return swap_ops

    num_qubits = state.num_qubits
    num_acted_qubits = len(qubit_idx)
    dim_ancilla = stinespring_mat.shape[0] // (2 ** num_acted_qubits)
    dim_main = 2 ** num_acted_qubits
    dim_extended = dim_ancilla * 2 ** num_qubits

    # transpose the stinespring matrix such that it has the shape of (dim_ancilla, dim_main, dim_main)
    # assuming the input form is (dim_main * dim_ancilla, dim_main)
    stine_m = stinespring_mat.reshape([dim_main, dim_ancilla, dim_main]).transpose([1, 0, 2]).reshape(
        [dim_main * dim_ancilla, dim_main])

    # rotate the density matrix such that the acted_qubits are at the head
    state_data = state.data
    higher_dims = state_data.shape[:-2]
    num_higher_dims = len(higher_dims)

    swap_ops = genSwapList(list(range(num_qubits)), qubit_idx + [x for x in range(num_qubits) if x not in qubit_idx])

    # make swap for left
    for swap_op in swap_ops:
        shape = higher_dims.copy()
        last_idx = -1
        for idx in swap_op:
            shape.append(2 ** (idx - last_idx - 1))
            shape.append(2)
            last_idx = idx
        shape.append(2 ** (2 * num_qubits - last_idx - 1))
        state_data = paddle.reshape(state_data, shape)
        state_data = paddle.transpose(
            state_data, list(range(num_higher_dims)) + [item + num_higher_dims for item in [0, 3, 2, 1, 4]]
        )

    # make swap for right
    for swap_op in swap_ops:
        shape = higher_dims.copy()
        last_idx = -1
        shape.append(2 ** num_qubits)
        for idx in swap_op:
            shape.append(2 ** (idx - last_idx - 1))
            shape.append(2)
            last_idx = idx
        shape.append(2 ** (num_qubits - last_idx - 1))
        state_data = paddle.reshape(state_data, shape)
        state_data = paddle.transpose(
            state_data, list(range(num_higher_dims)) + [item + num_higher_dims for item in [0, 1, 4, 3, 2, 5]]
        )

    # multiply the stinespring matrix
    state_data = paddle.reshape(
        state_data, higher_dims.copy() + [dim_main, -1]
    )
    state_data = paddle.reshape(
        paddle.matmul(
            stine_m, state_data
        ), higher_dims.copy() + [dim_extended, 2 ** num_qubits]
    )
    state_data = paddle.reshape(
        state_data, higher_dims.copy() + [dim_extended, dim_main, -1]
    )
    state_data = paddle.reshape(
        paddle.matmul(
            stine_m.conj(), state_data
        ), higher_dims.copy() + [dim_extended, dim_extended]
    )

    # make partial trace
    state_data = paddle.trace(
        paddle.reshape(
            state_data,
            higher_dims.copy() + [dim_ancilla, 2 ** num_qubits, dim_ancilla, 2 ** num_qubits]
        ),
        axis1=len(higher_dims),
        axis2=2 + len(higher_dims)
    )

    # swap back
    revert_swap_ops = genSwapList(qubit_idx + [x for x in range(num_qubits) if x not in qubit_idx],
                                  list(range(num_qubits)))
    for swap_op in revert_swap_ops:
        shape = higher_dims.copy()
        last_idx = -1
        for idx in swap_op:
            shape.append(2 ** (idx - last_idx - 1))
            shape.append(2)
            last_idx = idx
        shape.append(2 ** (2 * num_qubits - last_idx - 1))
        state_data = paddle.reshape(state_data, shape)
        state_data = paddle.transpose(
            state_data, list(range(num_higher_dims)) + [item + num_higher_dims for item in [0, 3, 2, 1, 4]]
        )
    for swap_op in revert_swap_ops:
        shape = higher_dims.copy()
        last_idx = -1
        shape.append(2 ** num_qubits)
        for idx in swap_op:
            shape.append(2 ** (idx - last_idx - 1))
            shape.append(2)
            last_idx = idx
        shape.append(2 ** (num_qubits - last_idx - 1))
        state_data = paddle.reshape(state_data, shape)
        state_data = paddle.transpose(
            state_data, list(range(num_higher_dims)) + [item + num_higher_dims for item in [0, 1, 4, 3, 2, 5]]
        )

    state_data = paddle.reshape(state_data, higher_dims.copy() + [2 ** num_qubits, 2 ** num_qubits])
    return paddle_quantum.State(state_data, dtype=dtype, backend=backend)
