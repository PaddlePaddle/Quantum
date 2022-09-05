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
The source file of the class for the measurement.
"""

import paddle
import paddle_quantum
from ..backend import quleaf
from ..backend import Backend
from ..intrinsic import _get_float_dtype
from typing import Optional, Union, Iterable


class ExpecVal(paddle_quantum.Operator):
    r"""The class of the loss function to compute the expectation value for the observable.

    This interface can make you using the expectation value for the observable as the loss function.

    Args:
        hamiltonian: The input observable.
        shots: The number of measurement shots. Defaults to ``0``. Now it just need to be input when the backend is QuLeaf.
    """
    def __init__(self, hamiltonian: paddle_quantum.Hamiltonian, shots: Optional[int] = 0):
        super().__init__()
        self.hamiltonian = hamiltonian
        self.shots = shots
        self.num_terms = self.hamiltonian.n_terms
        self.coeffs = paddle.to_tensor(self.hamiltonian.coefficients)
        self.sites = self.hamiltonian.sites
        self.matrices = self.hamiltonian.pauli_words_matrix

    def __state_qubits_swap(self, pauli_site, state_data, num_qubits):
        # generate swap_list
        origin_seq = list(range(0, num_qubits))
        target_seq = pauli_site + [x for x in origin_seq if x not in pauli_site]
        swapped = [False] * num_qubits
        swap_list = []
        for idx in range(0, num_qubits):
            if not swapped[idx]:
                next_idx = idx
                swapped[next_idx] = True
                while not swapped[target_seq[next_idx]]:
                    swapped[target_seq[next_idx]] = True
                    if next_idx < target_seq[next_idx]:
                        swap_list.append((next_idx, target_seq[next_idx]))
                    else:
                        swap_list.append((target_seq[next_idx], next_idx))
                    next_idx = target_seq[next_idx]

        # function for swapping ath and bth qubit in state
        def swap_a_and_b(target_state, size, pos_a, pos_b):
            target_state = target_state.reshape([2 ** pos_a, 2, 2 ** (pos_b - pos_a - 1), 2, 2 ** (size - pos_b - 1)])
            return paddle.transpose(target_state, [0, 3, 2, 1, 4])

        # begin swap
        if self.backend == paddle_quantum.Backend.DensityMatrix:
            for a, b in swap_list:
                state_data = swap_a_and_b(state_data, 2 * num_qubits, a, b)
                state_data = swap_a_and_b(state_data, 2 * num_qubits, a + num_qubits, b + num_qubits)
        else:
            for a, b in swap_list:
                state_data = swap_a_and_b(state_data, num_qubits, a, b)
        return state_data

    def forward(self, state: paddle_quantum.State) -> paddle.Tensor:
        r"""Compute the expectation value of the observable with respect to the input state.

        The value computed by this function can be used as a loss function to optimize.

        Args:
            state: The input state which will be used to compute the expectation value.

        Raises:
            NotImplementedError: The backend is wrong or not supported.

        Returns:
            The expectation value. If the backend is QuLeaf, it is computed by sampling.
        """
        if self.backend == Backend.QuLeaf:
            if len(state.param_list) > 0:
                param = paddle.concat(state.param_list)
            else:
                param = paddle.to_tensor(state.param_list)
            expec_val = quleaf.ExpecValOp.apply(
                param,
                state, self.hamiltonian, self.shots
            )
            return expec_val

        num_qubits = state.num_qubits
        expec_val = paddle.zeros([1])
        state_data = state.data
        for i in range(0, self.num_terms):
            pauli_site = self.sites[i]
            if pauli_site == ['']:
                expec_val += self.coeffs[i]
                continue
            num_applied_qubits = len(pauli_site)
            matrix = self.matrices[i]
            # extract current state and do swap operation
            if num_qubits != 1:
                _state_data = self.__state_qubits_swap(pauli_site, state_data, num_qubits)
            else:
                _state_data = state_data
            # use einstein sum notation to shrink the size of operation of matrix multiplication
            if self.backend == Backend.StateVector:
                _state_data = _state_data.reshape([2 ** num_applied_qubits, 2 ** (num_qubits - num_applied_qubits)])
                output_state = paddle.einsum('ia, ab->ib', matrix, _state_data).reshape([2 ** num_qubits])
                _state_data = paddle.conj(_state_data.reshape([2 ** num_qubits]))
                expec_val += paddle.real(paddle.matmul(_state_data, output_state)) * self.coeffs[i]
            elif self.backend == Backend.DensityMatrix:
                _state_data = _state_data.reshape([2 ** num_applied_qubits, 2 ** (2 * num_qubits - num_applied_qubits)])
                output_state = paddle.einsum('ia, ab->ib', matrix, _state_data)
                output_state = output_state.reshape([2 ** num_qubits, 2 ** num_qubits])
                expec_val += paddle.real(paddle.trace(output_state)) * self.coeffs[i]
            else:
                raise NotImplementedError
        return expec_val


class Measure(paddle_quantum.Operator):
    r"""Compute the probability of the specified measurement result.

    Args:
        measure_basis: Specify the basis of the measurement. Defaults to ``'z'``.

    Raises:
        NotImplementedError: Currently we just support the z basis.
    """
    def __init__(self, measure_basis: Optional[Union[Iterable[paddle.Tensor], str]] = 'z'):
        super().__init__()
        if measure_basis == 'z' or measure_basis == 'computational_basis':
            self.measure_basis = 'z'
        else:
            raise NotImplementedError

    def forward(
            self, state: paddle_quantum.State, qubits_idx: Optional[Union[Iterable[int], int, str]] = 'full',
            desired_result: Optional[Union[Iterable[str], str]] = None
    ) -> paddle.Tensor:
        r"""Compute the probability of measurement to the input state.

        Args:
            state: The quantum state to be measured.
            qubits_idx: The index of the qubits to be measured. Defaults to ``'full'`` which means measure all the qubits.
            desired_result: Specify the results of the measurement to return. Defaults to ``None`` which means return the probability of all the results.

        Raises:
            NotImplementedError: The backend is wrong or not supported.
            NotImplementedError: The qubit index is wrong or not supported.
            NotImplementedError: Currently we just support the z basis.
            
        Returns:
            The probability of the measurement.
        """
        float_dtype = _get_float_dtype(paddle_quantum.get_dtype())
        num_qubits = state.num_qubits
        if self.measure_basis == 'z':
            if state.backend == paddle_quantum.Backend.StateVector:
                prob_amplitude = paddle.multiply(paddle.conj(state.data), state.data).real()
            elif state.backend == paddle_quantum.Backend.DensityMatrix:
                prob_amplitude = paddle.zeros([2 ** num_qubits], dtype=float_dtype)
                for idx in range(0, 2 ** num_qubits):
                    prob_amplitude[idx] += state.data[idx, idx].real()
            else:
                raise NotImplementedError("The backend is wrong or not supported.")

            if isinstance(qubits_idx, int):
                qubits_idx = [qubits_idx]
            if isinstance(qubits_idx, Iterable) and all((isinstance(idx, int) for idx in qubits_idx)):
                qubits_idx = list(qubits_idx)
                measured_num = len(qubits_idx)
                prob_array = paddle.zeros([2 ** measured_num], dtype=float_dtype)
                for idx in range(0, 2 ** num_qubits):
                    binary = bin(idx)[2:]
                    binary = '0' * (num_qubits - len(binary)) + binary
                    target_qubits = ''
                    for qubit_idx in qubits_idx:
                        target_qubits += binary[qubit_idx]
                    prob_array[int(target_qubits, base=2)] += prob_amplitude[idx]
            elif qubits_idx == 'full':
                prob_array = prob_amplitude
            else:
                raise NotImplementedError("The qubit index is wrong or not supported.")

            prob_array = prob_array / paddle.sum(prob_array) # normalize calculation error
            if desired_result is None:
                return prob_array
            if isinstance(desired_result, str):
                desired_result = [desired_result]
                prob_array = paddle.concat([prob_array[int(res, base=2)] for res in desired_result])
            return prob_array
        else:
            raise NotImplementedError("Currently we just support the z basis.")
