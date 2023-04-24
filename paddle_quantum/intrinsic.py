# !/usr/bin/env python3
# Copyright (c) 2020 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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
The intrinsic function of the paddle quantum.
"""

import numpy as np
import paddle
from typing import Union, Iterable, List, Tuple

import paddle_quantum as pq
from .base import get_dtype
from .backend import Backend


def _zero(dtype=None):
    dtype = get_dtype() if dtype is None else dtype
    return paddle.to_tensor(0, dtype=dtype)


def _one(dtype=None):
    dtype = get_dtype() if dtype is None else dtype
    return paddle.to_tensor(1, dtype=dtype)


def _format_qubits_idx(
        qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int, str],
        num_qubits: int, num_acted_qubits: int = 1
) -> Union[List[List[int]], List[int]]:
    r"""Formatting the qubit indices that operations acts on

    Args:
        qubits_idx: input qubit indices, could be a string
        num_qubits: total number of qubits

    Note:
        The shape of output qubit indices are formatted as follows:
        - If num_acted_qubits is 1, the output shape is [# of qubits that one operation acts on];
        - otherwise, the output shape is [# of vertical gates, # of qubits that one operation acts on].

    """
    assert not (isinstance(qubits_idx, str) and num_qubits is None), \
        f"Cannot specify the qubit indices when num_qubits is None: received qubit_idx {qubits_idx} and num_qubits {num_qubits}"
    if num_acted_qubits == 1:
        if qubits_idx == 'full':
            qubits_idx = list(range(0, num_qubits))
        elif qubits_idx == 'even':
            qubits_idx = list(range(0, num_qubits, 2))
        elif qubits_idx == 'odd':
            qubits_idx = list(range(1, num_qubits, 2))
        elif isinstance(qubits_idx, Iterable):
            qubits_idx = list(qubits_idx)
            assert len(qubits_idx) == len(set(qubits_idx)), \
                f"Single-qubit operators do not allow repeated indices: received {qubits_idx}"
        else:
            qubits_idx = [qubits_idx]
    else:
        if qubits_idx == 'cycle':
            qubits_idx = []
            for idx in range(0, num_qubits - num_acted_qubits):
                qubits_idx.append(
                    [i for i in range(idx, idx + num_acted_qubits)])
            for idx in range(num_qubits - num_acted_qubits, num_qubits):
                qubits_idx.append([i for i in range(idx, num_qubits)] +
                                  [i for i in range(idx + num_acted_qubits - num_qubits)])
        elif qubits_idx == 'linear':
            qubits_idx = []
            for idx in range(0, num_qubits - num_acted_qubits + 1):
                qubits_idx.append(
                    [i for i in range(idx, idx + num_acted_qubits)])
        elif len(np.shape(qubits_idx)) == 1 and len(qubits_idx) == num_acted_qubits:
            qubits_idx = [list(qubits_idx)]
        elif len(np.shape(qubits_idx)) == 2 and all((len(indices) == num_acted_qubits for indices in qubits_idx)):
            qubits_idx = [list(indices) for indices in qubits_idx]
        else:
            raise TypeError(
                "The qubits_idx should be iterable such as list, tuple, and so on whose elements are all integers."
                "And the length of acted_qubits should be consistent with the corresponding gate."
                f"\n    Received qubits_idx type {type(qubits_idx)}, qubits # {len(qubits_idx)}, gate dimension {num_acted_qubits}"
            )
    return qubits_idx


def _format_param_shape(depth: int, qubits_idx: Union[List[List[int]], List[int]],
                        num_acted_param: int, param_sharing: bool) -> List[int]:
    r"""Formatting the shape of parameters

    Args:
        depth: depth of the layer
        qubits_idx: list of input qubit indices
        num_acted_param: the number of parameters required for a single operation
        param_sharing: whether all operations are shared by the same parameter set 

    Note:
        The input ``qubits_idx`` must be formatted by ``_format_qubits_idx`` first.
        The shape of parameters are formatted as follows:
        - If param_sharing is True, the shape is [depth, num_acted_param];
        - otherwise, the shape is [depth, len(qubits_idx), num_acted_param].

    """
    if param_sharing:
        return [depth, num_acted_param]
    return [depth, len(qubits_idx), num_acted_param]


def _get_float_dtype(complex_dtype: str) -> str:
    if complex_dtype == 'complex64':
        float_dtype = 'float32'
    elif complex_dtype == 'complex128':
        float_dtype = 'float64'
    else:
        raise ValueError(
            f"The dtype should be 'complex64' or 'complex128': received {complex_dtype}")
    return float_dtype


def _type_fetch(data: Union[np.ndarray, paddle.Tensor, pq.State]) -> str:
    r""" fetch the type of ``data``

    Args:
        data: the input data, and datatype of which should be either ``numpy.ndarray``,
    ''paddle.Tensor'' or ``paddle_quantum.State``

    Returns:
        string of datatype of ``data``, can be either ``"numpy"``, ``"tensor"``,
    ``"state_vector"`` or ``"density_matrix"``

    Raises:
        ValueError: does not support the current backend of input state.
        TypeError: cannot recognize the current type of input data.

    """
    if isinstance(data, np.ndarray):
        return "numpy"

    if isinstance(data, paddle.Tensor):
        return "tensor"

    if isinstance(data, pq.State):
        if data.backend == Backend.StateVector:
            return "state_vector"
        if data.backend == Backend.DensityMatrix:
            return "density_matrix"
        raise ValueError(
            f"does not support the current backend {data.backend} of input state.")

    raise TypeError(
        f"cannot recognize the current type {type(data)} of input data.")


def __density_to_vector(rho: Union[np.ndarray, paddle.Tensor]) -> Union[np.ndarray, paddle.Tensor]:
    r""" transform a density matrix to a state vector

    Args:
        rho: a density matrix (pure state)

    Returns:
        a state vector

    Raises:
        ValueError: the output state may not be a pure state

    """
    type_str = _type_fetch(rho)
    rho = paddle.to_tensor(rho)
    eigval, eigvec = paddle.linalg.eigh(rho)

    max_eigval = paddle.max(eigval).item()
    err = np.abs(max_eigval - 1)
    if err > 1e-6:
        raise ValueError(
            f"the output state may not be a pure state, maximum distance: {err}")

    state = eigvec[:, paddle.argmax(eigval)]

    return state.numpy() if type_str == "numpy" else state


def _type_transform(data: Union[np.ndarray, paddle.Tensor, pq.State],
                    output_type: str) -> Union[np.ndarray, paddle.Tensor, pq.State]:
    r""" transform the datatype of ``input`` to ``output_type``

    Args:
        data: data to be transformed
        output_type: datatype of the output data, type is either ``"numpy"``, ``"tensor"``,
    ``"state_vector"`` or ``"density_matrix"``

    Returns:
        the output data with expected type

    Raises:
        ValueError: does not support transformation to type.

    """
    current_type = _type_fetch(data)

    support_type = {"numpy", "tensor", "state_vector", "density_matrix"}
    if output_type not in support_type:
        raise ValueError(
            f"does not support transformation to type {output_type}")

    if current_type == output_type:
        return data

    if current_type == "numpy":
        if output_type == "tensor":
            return paddle.to_tensor(data)

        data = np.squeeze(data)
        # state_vector case
        if output_type == "state_vector":
            if len(data.shape) == 2:
                data = __density_to_vector(data)
            return pq.State(data, backend=Backend.StateVector)
        # density_matrix case
        if len(data.shape) == 1:
            data = data.reshape([len(data), 1])
            data = data @ np.conj(data.T)
        return pq.State(data, backend=Backend.DensityMatrix)

    if current_type == "tensor":
        if output_type == "numpy":
            return data.numpy()

        data = paddle.squeeze(data)
        # state_vector case
        if output_type == "state_vector":
            if len(data.shape) == 2:
                data = __density_to_vector(data)
            return pq.State(data, backend=Backend.StateVector)

        # density_matrix case
        if len(data.shape) == 1:
            data = data.reshape([len(data), 1])
            data = data @ paddle.conj(data.T)
        return pq.State(data, backend=Backend.DensityMatrix)

    if current_type == "state_vector":
        if output_type == "density_matrix":
            return pq.State(data.ket @ data.bra, backend=Backend.DensityMatrix, num_qubits=data.num_qubits, override=True)
        return data.ket.numpy() if output_type == "numpy" else data.ket

    # density_matrix data
    if output_type == "state_vector":
        return pq.State(__density_to_vector(data.data), backend=Backend.StateVector, num_qubits=data.num_qubits, override=True)
    return data.numpy() if output_type == "numpy" else data.data


def _perm_to_swaps(perm: List[int]) -> List[Tuple[int]]:
    r"""This function takes a permutation as a list of integers and returns its
        decomposition into a list of tuples representing the two-permutation (two conjugated 2-cycles).

    Args:
        perm: the target permutation

    Returns:
        the decomposition of the permutation.
    """
    n = len(perm)
    swapped = [False] * n
    swap_ops = []

    for idx in range(0, n):
        if not swapped[idx]:
            next_idx = idx
            swapped[next_idx] = True
            while not swapped[perm[next_idx]]:
                swapped[perm[next_idx]] = True
                if next_idx < perm[next_idx]:
                    swap_ops.append((next_idx, perm[next_idx]))
                else:
                    swap_ops.append((perm[next_idx], next_idx))
                next_idx = perm[next_idx]

    return swap_ops


def _trans_ops(state: paddle.Tensor, swap_ops: List[Tuple[int]], higher_dims: List[int], num_qubits: int,
               extra_dims: int = 1) -> paddle.Tensor:
    r"""Transpose the state tensor given a list of swap operations.

    Args:
        swap_ops: given list of swap operations
        higher_dims: intrinsic dimension of the state tensor
        num_qubits: the number of qubits in the system
        extra_dims: labeling the dimension of state, 1 for statevector; 2 for density operator

    Returns:
        paddle.Tensor: transposed state tensor given the swap list
    """
    num_higher_dims = len(higher_dims)
    for swap_op in swap_ops:
        shape = higher_dims.copy()
        shape.extend([2**(swap_op[0]), 2, 2**(swap_op[1] - swap_op[0] - 1),
                     2, 2**(extra_dims * num_qubits - swap_op[1] - 1)])
        state = paddle.reshape(state, shape)
        state = paddle.transpose(
            state, list(range(0, num_higher_dims)) +
            [item + num_higher_dims for item in [0, 3, 2, 1, 4]]
        )
    return state


def _perm_of_list(orig_list: List[int], targ_list: List[int]) -> List[int]:
    r"""Find the permutation mapping the original list to the target list
    """
    perm_map = {val: index for index, val in enumerate(orig_list)}
    return [perm_map[val] for val in targ_list]


def __gate_tensor(gate: List[paddle.Tensor]) -> paddle.Tensor:
    gate_tensor = gate[0]
    for i in range(1, len(gate)):
        gate_tensor = paddle.kron(gate_tensor, gate[i])
    return gate_tensor


def _gate_tensor(gate: List[paddle.Tensor]) -> paddle.Tensor:

    # TODO extend to multi-qubit cases
    # tensor product of single-qubit gates
    if len(gate) > 1:
        gate_half1 = __gate_tensor(gate[: len(gate) // 2])
        gate_half2 = __gate_tensor(gate[len(gate) // 2:])
        gate = paddle.kron(gate_half1, gate_half2)
    else:
        gate = gate[0]
    return gate


def _paddle_gather(data: paddle.Tensor, index: paddle.Tensor) -> paddle.Tensor:
    return paddle.gather(paddle.real(data), index=index) + \
           paddle.gather(paddle.imag(data), index=index) * 1.j


def _base_transpose(state: paddle.Tensor, perm: Union[List, Tuple]) -> paddle.Tensor:
    r"""speed-up logic using using np.transpose + paddle.gather.

    Args:
        state: input state data.
        perm: permutation of qubit sequence.

    Returns:
        paddle.Tensor: permuted state.
    """
    num_qubits = len(perm)
    # Using the logic changing the order of each component in a 2**n array
    base_idx = np.arange(2 ** num_qubits).reshape([2] * num_qubits)
    base_idx = np.transpose(base_idx, axes=perm).reshape([-1])

    # whether to use batch in state_vector backend. len(state.shape) equals 1 means not using batch
    if len(state.shape) == 1:
        return _paddle_gather(state, index=paddle.to_tensor(base_idx))

    state = paddle.reshape(state, [-1, 2 ** num_qubits]).T
    state = _paddle_gather(state, index=paddle.to_tensor(base_idx))
    return state.T


def _inverse_gather_for_dm(state: paddle.Tensor, base_idx: paddle.Tensor) -> paddle.Tensor:
    r"""replacement of the usage of paddle.gather(axis=1)
    """
    # TODO When the `axis` bug is fixed, please use the `axis` argument instead of using transpose.

    # whether to use batch in density_matrix backend. len(state.shape) equals 2 means not using batch
    if len(state.shape) == 2:
        state = state.T
        state = _paddle_gather(state, index=base_idx)
        return state.T

    state = paddle.transpose(state, perm=[1, 0, 2])
    state = _paddle_gather(state, index=base_idx)
    return paddle.transpose(state, perm=[2, 1, 0])


def _base_transpose_for_dm(state: paddle.Tensor, perm: Union[List, Tuple]) -> paddle.Tensor:
    r"""speed-up logic using using np.transpose + paddle.gather.

    Args:
        state: input state data.
        perm: permutation of qubit sequence.

    Returns:
        paddle.Tensor: permuted state.
    """
    num_qubits = len(perm)
    # Using the logic changing the order of each component in a 2**n array
    base_idx = np.arange(2 ** num_qubits).reshape([2] * num_qubits)
    base_idx = paddle.to_tensor(np.transpose(base_idx, axes=perm).reshape([-1]))

    # left swap
    # whether to use batch in density_matrix backend. len(state.shape) is greater than 2 means using batch
    if len(state.shape) > 2:
        state = paddle.reshape(state, [-1, 2 ** num_qubits, 2 ** num_qubits])
        state = paddle.transpose(state, perm=[1, 2, 0])

    state = _paddle_gather(state, index=base_idx)

    # right swap
    state = _inverse_gather_for_dm(state, base_idx)

    return state


def _cnot_idx_fetch(num_qubits: int, qubits_idx: List[Tuple[int, int]]) -> List[int]:
    r"""
    Compute the CNOT index obtained by applying the CNOT gate without using matrix multiplication.

    Args:
        num_qubits: The total number of qubits in the system.
        qubits_idx: A list of tuples, where each tuple contains the indices of the two qubits
                    involved in the CNOT gate.

    Returns:
        List: A list of integers representing the decimal values of all binary strings
                obtained by applying the CNOT gate.
    """
    assert len(np.shape(qubits_idx)) == 2, \
        "The CNOT qubits_idx should be list of tuple of integers, e.g., [[0, 1], [1, 2]]."
    binary_list = [bin(i)[2:].zfill(num_qubits) for i in range(2 ** num_qubits)]
    qubits_idx_length = len(qubits_idx)
    for item in range(len(binary_list)):
        for bin_idx in range(qubits_idx_length):
            id1 = qubits_idx[qubits_idx_length - bin_idx - 1][0]
            id2 = qubits_idx[qubits_idx_length - bin_idx - 1][1]
            if binary_list[item][id1] == "1":
                if binary_list[item][id2] == '0':
                    binary_list[item] = binary_list[item][:id2] + '1' + binary_list[item][id2 + 1:]
                else:
                    binary_list[item] = binary_list[item][:id2] + '0' + binary_list[item][id2 + 1:]

    decimal_list = [int(binary, 2) for binary in binary_list]
    return decimal_list
