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
The underlying logic operations of quantum channels.
"""

import functools
import paddle
from ...backend import Backend, density_matrix
from ...state import State
from typing import Iterable, List, Tuple, Union


def kraus_repr(
        state: State, list_kraus_oper: Iterable[paddle.Tensor], qubit_idx: Union[List[int], int],
        dtype: str, backend: Backend
) -> State:
    r"""Apply a custom channel in the Kraus representation on the input state.

    Args:
        state: Input state.
        list_kraus_oper: Kraus operators of this channel.
        qubit_idx: Index of the qubit on which the channel acts.
        dtype: Type of data.
        backend: Backend on which the simulation is run.

    Raises:
        RuntimeError: The noisy channel can only run in density matrix mode.

    Returns:
        Output state.
    """
    if state.backend != Backend.DensityMatrix or backend != Backend.DensityMatrix:
        raise RuntimeError("The noisy channel can only run in density matrix mode.")
    state_data = [
        density_matrix.unitary_transformation(
            state.data,
            oper,
            qubit_idx if isinstance(qubit_idx, list) else [qubit_idx],
            state.num_qubits
        ) for oper in list_kraus_oper
    ]
    state_data = functools.reduce(lambda x, y: x + y, state_data)
    transformed_state = state.clone()
    transformed_state.data = state_data
    return transformed_state


def choi_repr(state: State, choi_oper: paddle.Tensor, qubit_idx: Union[List[int], int],
              dtype: str, backend: Backend) -> State:
    r"""Apply a custom channel in the Choi representation on the input state. 
    The choi operator is with form
    
    .. math::
    
        \sum_{i, j} |i\rangle\langle j| \otimes N(|i\rangle\langle j|)

    Args:
        state: Input quantum state
        choi_oper: Choi representation for the channel :math:`N`
        qubit_idx: Index of the qubit on which the channel acts on
        dtype: Type of data
        backend: Backend on which the simulation is run

    Returns:
        Output state.
        
    """
    if state.backend != Backend.DensityMatrix or backend != Backend.DensityMatrix:
        raise RuntimeError("The noisy channel can only run in density matrix mode.")
    
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
    return State(new_state, dtype=dtype, backend=backend)


def stinespring_repr(state: State, stinespring_mat: paddle.Tensor, qubit_idx: Union[List[int], int], 
                     dtype: str, backend: Backend) -> State:
    r"""Apply a custom channel in the Stinespring representation on the input state. 
    ``stinespring_mat`` is a :math:`(d_1 * d_2) \times d_1`  rectangular matrix.
    Here :math:`d_1` is the dimension of space where ``qubit_idx`` locates, 
    :math:`d_2` is the dimension of auxillary system. 
    With Dirac notation ``stinespring_mat`` can be defined as

    .. math::
    
        \text{stinespring_mat.reshape}([d_1, d_2, d_1])[i, j, k] = \langle i, j| A |k \rangle

    with :math:`A` being the Stinespring operator and the channel acting as :math:`\rho \mapsto \text{tr}_2 (A \rho A^\dagger)`.

    Args:
        state: Input quantum state
        stinespring_mat: Stinespring representation :math:`A` for the channel
        qubit_idx: Index of the qubit on which the channel acts on
        dtype: Type of data
        backend: Backend on which the simulation is run

    Returns:
        Output state.
        
    """
    if state.backend != Backend.DensityMatrix or backend != Backend.DensityMatrix:
        raise RuntimeError("The noisy channel can only run in density matrix mode.")
    
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
    return State(state_data, dtype=dtype, backend=backend)
