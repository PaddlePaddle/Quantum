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
import numpy as np
import paddle

from typing import Callable, Optional, Tuple, Union
from math import log2

from .angles import qpp_angle_finder, qpp_angle_approximator
from .laurent import Q_generation, pair_generation, laurent_generator, revise_tol

from ..ansatz import Circuit
from ..backend import Backend
from ..base import get_dtype
from ..gate import X
from ..intrinsic import _get_float_dtype
from ..linalg import unitary_random, is_unitary
from ..loss import Measure
from ..operator import Collapse
from ..qinfo import dagger, partial_trace
from ..state import State, _type_transform, zero_state


r"""
QPP circuit and related tools, see Theorem 6 in paper https://arxiv.org/abs/2209.14278 for more details.
"""


__all__ = ["qpp_cir", "simulation_cir", "qps", "qubitize", "purification_block_enc"]


def qpp_cir(
    list_theta: Union[np.ndarray, paddle.Tensor],
    list_phi: Union[np.ndarray, paddle.Tensor],
    U: Union[np.ndarray, paddle.Tensor, float],
) -> Circuit:
    r"""Construct a quantum phase processor of QPP by `list_theta` and `list_phi`.

    Args:
        list_theta: angles for :math:`R_Y` gates.
        list_phi: angles for :math:`R_Z` gates.
        U: unitary or scalar input.

    Returns:
        a multi-qubit version of trigonometric QSP.

    """
    complex_dtype = get_dtype()
    float_dtype = _get_float_dtype(complex_dtype)

    if not isinstance(list_theta, paddle.Tensor):
        list_theta = paddle.to_tensor(list_theta, dtype=float_dtype)
    if not isinstance(list_phi, paddle.Tensor):
        list_phi = paddle.to_tensor(list_phi, dtype=float_dtype)
    if not isinstance(U, paddle.Tensor):
        U = paddle.to_tensor(U, dtype=complex_dtype)
    if len(U.shape) == 1:
        U = U.cast(float_dtype)
    list_theta, list_phi = np.squeeze(list_theta), np.squeeze(list_phi)

    assert len(list_theta) == len(list_phi)
    n = int(log2(U.shape[0]))
    L = len(list_theta) - 1

    cir = Circuit(n + 1)
    all_register = list(range(n + 1))

    for i in range(L):
        cir.rz(0, param=list_phi[i])
        cir.ry(0, param=list_theta[i])

        # input the unitary
        if len(U.shape) == 1:
            cir.rz(0, param=U)
        elif i % 2 == 0:
            cir.control_oracle(U, all_register, latex_name=r"$U$")
        else:
            cir.x(0)
            cir.control_oracle(dagger(U), all_register, latex_name=r"$U^\dagger$")
            cir.x(0)
    cir.rz(0, param=list_phi[-1])
    cir.ry(0, param=list_theta[-1])

    return cir


def simulation_cir(
    fn: Callable[[np.ndarray], np.ndarray],
    U: Union[np.ndarray, paddle.Tensor, float],
    approx: Optional[bool] = True,
    deg: Optional[int] = 50,
    length: Optional[float] = np.pi,
    step_size: Optional[float] = 0.00001 * np.pi,
    tol: Optional[float] = 1e-30,
) -> Circuit:
    r"""Return a QPP circuit approximating `fn`.

    Args:
        fn: function to be approximated.
        U: unitary input.
        deg: degree of approximation, defaults to be :math:`50`.
        approx: whether approximately find the angle of circuits, defaults to be ``True``
        length: half of approximation width, defaults to be :math:`\pi`.
        step_size: sampling frequency of data points, defaults to be :math:`0.00001 \pi`.
        tol: error tolerance, defaults to be :math:`10^{-30}`.

    Returns:
        a QPP circuit approximating `fn` in Theorem 6 of paper https://arxiv.org/abs/2209.14278.

    """
    f = laurent_generator(fn, step_size, deg, length)
    if np.abs(f.max_norm - 1) < 1e-2:
        f = f * (0.999999999 / f.max_norm)
    P, Q = pair_generation(f)
    revise_tol(tol)
    list_theta, list_phi = (
        qpp_angle_approximator(P, Q) if approx else qpp_angle_finder(P, Q)
    )

    return qpp_cir(list_theta, list_phi, U)


def qps(
    U: Union[np.ndarray, paddle.Tensor],
    initial_state: Union[np.ndarray, paddle.Tensor, State],
) -> Tuple[float, State]:
    r"""Algorithm for quantum phase search, see Algorithm 1 and 2 of paper https://arxiv.org/abs/2209.14278.

    Args:
        U: target unitary
        initial_state: the input quantum state

    Returns:
        contains the following two elements:
        *   an eigenphase of U
        *   its corresponding eigenstate that has overlap with the initial state

    """
    assert is_unitary(U)
    U, initial_state = _type_transform(U, "tensor"), _type_transform(
        initial_state, "state_vector"
    )
    initial_state = zero_state(1).kron(initial_state)

    def p_func(x):
        target = 2.0 * np.arctan(np.sin(x) / 0.01) / np.pi
        return np.sqrt((1 + target) / 2)

    # function approximation
    P = laurent_generator(p_func, 0.00001 * np.pi, 160, np.pi)
    Q = Q_generation(P)
    list_theta, list_phi = qpp_angle_approximator(P, Q)

    # Algorithm setting
    T, Q, Delta = 13, 10, 0.2
    Delta_bar = Delta + np.pi / (2 ** (Q + 1))

    op_M = Measure()  # measurement
    op_C = [
        Collapse(0, desired_result="0"),
        Collapse(0, desired_result="1"),
    ]  # collapse operator
    op_X = X(0)  # x gate used to reset the first qubit

    def __measure_and_collapse(target_state: State) -> Tuple[float, State]:
        r"""Measure ``target_state``, and return the measurement outcome and collapsed state"""
        prob_dist = op_M(target_state, 0).tolist()
        result = np.random.choice(2, 1, p=prob_dist)[0]
        return result, op_C[result](target_state)

    def __interval_search(
        U: paddle.Tensor, input_state: State, interval: Tuple[int, int]
    ) -> Tuple[int, int]:
        r"""Find a smaller interval containing eigenphases from ``interval``, Algorithm 1."""
        x_low, x_up = interval[0], interval[1]
        theta = (x_up + x_low) / 2
        state = input_state.clone()

        # if the interval is too wide, the update rule is different
        if x_up - x_low > 2 * np.pi - 2 * Delta:
            state = qpp_cir(list_theta, list_phi, np.exp(-1j * theta).item() * U)(state)
            result, state = __measure_and_collapse(state)
            if result == 0:
                x_low = -Delta
                x_up += Delta
            else:
                x_low -= Delta
                x_up = Delta
                state = op_X(state)
            theta = (x_up + x_low) / 2

        # begin interval search formally
        for _ in range(Q):
            state = qpp_cir(list_theta, list_phi, np.exp(-1j * theta).item() * U)(state)
            result, state = __measure_and_collapse(state)

            # update the interval according to the measurement result of the first qubit
            #   - if the measurement result is 0, choose the RHS interval
            #   - otherwise, choose the LHS and reset the first qubit
            if result == 0:
                x_low = theta - Delta
            else:
                x_up = theta + Delta
                state = op_X(state)

            theta = (x_up + x_low) / 2

        return [x_low, x_up], state

    # Algorithm 2
    phase_estimate = 0
    p = int(np.round(1 / Delta_bar))
    itr_interval = [-np.pi, np.pi]
    itr_state = initial_state
    for j in range(T):
        itr_interval, itr_state = __interval_search(U, itr_state, itr_interval)

        lamb = (itr_interval[0] + itr_interval[1]) / 2
        phase_estimate += lamb * (Delta**j)

        # update unitary and interval, so that we can start a new interval search
        U = np.linalg.matrix_power(U, p) * np.exp(-1j * p * lamb)
        itr_interval = ((np.array(itr_interval) - lamb) * p).tolist()

    return phase_estimate, partial_trace(itr_state, 2**1, U.shape[0], A_or_B=1)


def qubitize(
    block_enc: Union[np.ndarray, paddle.Tensor], num_block_qubits: int
) -> paddle.Tensor:
    r"""Qubitize the block encoding to keep subspace invariant using one more ancilla qubit,
    see paper http://arxiv.org/abs/1610.06546 for more details.

    Args:
        block_enc: the target block encoding.
        num_block_qubits: number of ancilla qubits used in block encoding.

    Returns:
        the qubitized version for ``block_enc``.

    """
    if isinstance(block_enc, np.ndarray):
        block_enc = paddle.to_tensor(block_enc)
    n = int(log2(block_enc.shape[0]))
    cir = Circuit(n + 1)
    cir.h(0)

    cir.x(0)
    cir.control_oracle(block_enc, list(range(n + 1)))
    cir.x(0)
    cir.control_oracle(dagger(block_enc), list(range(n + 1)))
    cir.x(0)

    cir.h(0)

    num_qubits = n - num_block_qubits
    zero_states = zero_state(num_block_qubits, backend=Backend.DensityMatrix)
    zero_projector = paddle.kron(
        2 * zero_states.data - paddle.eye(2**num_block_qubits),
        paddle.eye(2**num_qubits),
    )
    return (cir.unitary_matrix() @ zero_projector).cast(block_enc.dtype)


def purification_block_enc(num_qubits: int, num_block_qubits: int) -> paddle.Tensor:
    r"""Randomly generate a :math:`(n + m)`-qubit qubitized block encoding of a :math:`n`-qubit density matrix.

    Args:
        num_qubits: number of qubits :math:`n`.
        num_block_qubits: number of ancilla qubits :math:`m > n` used in block encoding.

    Returns:
         a :math:`2^{n + m} \times 2^{n + m}` unitary matrix that its upper-left block is a density matrix.

    """
    assert (
        num_qubits < num_block_qubits
    ), f"the number of block qubits need to be larger than the number of qubits {num_qubits}: received {num_block_qubits}"

    V = unitary_random(num_block_qubits)
    aux1_reg = list(range(num_block_qubits - num_qubits))
    aux2_reg = list(range(num_block_qubits - num_qubits, num_block_qubits))
    sys_reg = list(range(num_block_qubits, num_qubits + num_block_qubits))

    cir = Circuit(num_qubits + num_block_qubits)
    cir.oracle(V, aux1_reg + aux2_reg)

    for i in range(num_qubits):
        cir.swap([aux2_reg[i], sys_reg[i]])

    cir.oracle(dagger(V), aux1_reg + aux2_reg)
    zero_states = paddle.eye(2**num_block_qubits, 1)
    reflector = 2 * zero_states @ dagger(zero_states) - paddle.eye(
        2**num_block_qubits
    )
    cir.oracle(reflector, list(range(num_block_qubits)))
    return cir.unitary_matrix()
