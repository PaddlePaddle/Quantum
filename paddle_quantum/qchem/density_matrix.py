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
Measure onebody density matrix from quantum state.
"""

from typing import List, Tuple
import paddle
import openfermion
import paddle_quantum

from .complex_utils import *

__all__ = ["get_spinorb_onebody_dm"]


def _get_float_dtype(dtype):
    r"""
    Get the compatible floating data type from the given complex data type.
    """
    if dtype == paddle.complex64:
        return paddle.float32
    elif dtype == paddle.complex128:
        return paddle.float64


def _get_onebody_hermitian_operator(n_qubits: int, i: int, j: int, with_spin: bool, dtype) -> paddle.Tensor:
    r"""
    Return the matrix corresponds to the
    .. math:
        \hat{c}_i^{\dagger}\hat{c}_j+\hat{c}_j^{\dagger}\hat{c}_i.
    
    Args:
        n_qubits: Number of qubits in the quantum circuit.
        i: Operator index.
        j: Operator index.
        with_spin: Whether use spin orbital or orbital to label the qubits in the quantum circuit.

    Returns:
        Hermitian matrix.
    """

    if i == j:
        ops = openfermion.QubitOperator("", 0.5) - openfermion.QubitOperator(f"Z{i}", 0.5)
        ops_array = openfermion.get_sparse_operator(ops, n_qubits).toarray()
    else:
        if with_spin:
            qstr_xzx = f"X{i} " + " ".join([f"Z{k}" for k in range(i + 2, j, 2)]) + f" X{j}"
            qstr_yzy = f"Y{i} " + " ".join([f"Z{k}" for k in range(i + 2, j, 2)]) + f" Y{j}"
        else:
            qstr_xzx = f"X{i} " + " ".join([f"Z{k}" for k in range(i + 1, j)]) + f" X{j}"
            qstr_yzy = f"Y{i} " + " ".join([f"Z{k}" for k in range(i + 1, j)]) + f" Y{j}"
        ops = openfermion.QubitOperator(qstr_xzx, 0.5) + openfermion.QubitOperator(qstr_yzy, 0.5)
        ops_array = openfermion.get_sparse_operator(ops, n_qubits).toarray()
    return paddle.to_tensor(ops_array, dtype=dtype)


class OneBodyDensityMatrix(paddle.autograd.PyLayer):
    r"""
    Measure the onebody density matrix from a quantum state.
    """
    @staticmethod
    def forward(ctx, n_qubits: int, orb_index: List[int], with_spin: bool, state: paddle.Tensor) -> paddle.Tensor:
        ctx.with_spin = with_spin
        ctx.n_qubits = n_qubits
        ctx.orb_index = orb_index
        ctx.save_for_backward(state)

        nao = len(orb_index)
        dm = paddle.zeros((nao, nao), dtype=_get_float_dtype(state.dtype))
        for i in range(nao):
            dm_op = _get_onebody_hermitian_operator(n_qubits, orb_index[i], orb_index[i], with_spin, state.dtype)
            dm[i, i] = _hermitian_expv(dm_op, state)
            for j in range(i + 1, nao):
                dm_op = _get_onebody_hermitian_operator(n_qubits, orb_index[i], orb_index[j], with_spin, state.dtype)
                dm[i, j] = 0.5 * _hermitian_expv(dm_op, state)
                dm[j, i] = dm[i, j]
        return dm

    @staticmethod
    def backward(ctx, grad_dm: paddle.Tensor):
        n_qubits = ctx.n_qubits
        state, = ctx.saved_tensor()
        orb_index = ctx.orb_index

        grad_state = 0.0 + 0.0j
        nao = len(orb_index)
        for i in range(nao):
            dm_op = _get_onebody_hermitian_operator(n_qubits, orb_index[i], orb_index[i], ctx.with_spin, state.dtype)
            grad_state += grad_dm[i, i] * 2 * _general_mv(dm_op, state)
            for j in range(i + 1, nao):
                dm_op = _get_onebody_hermitian_operator(n_qubits, orb_index[i], orb_index[j], ctx.with_spin,
                                                        state.dtype)
                grad_state += (grad_dm[i, j] + grad_dm[j, i]) * _general_mv(dm_op, state)
        return grad_state


get_onebody_dm = OneBodyDensityMatrix.apply


def get_spinorb_onebody_dm(n_qubits: int, state: paddle.Tensor) -> Tuple[paddle.Tensor]:
    r"""
    Get the onebody density matrix from a given state in which qubits are label by spin orbital index.

    Args:
        n_qubits: number of qubits in the quantum circuit.
        state: the given quantum state.
    
    Returns:
        spin up and spin down onebody density matrix.
    """

    assert n_qubits % 2 == 0, "number of spin orbital should be even."
    a_orb_index = range(0, n_qubits, 2)
    b_orb_index = range(1, n_qubits, 2)
    dm_a = get_onebody_dm(n_qubits, a_orb_index, True, state)
    dm_b = get_onebody_dm(n_qubits, b_orb_index, True, state)
    return dm_a, dm_b
