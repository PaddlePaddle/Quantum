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
from ..backend import Backend, state_vector
from ..intrinsic import _get_float_dtype, _perm_of_list, _base_transpose, _base_transpose_for_dm
from typing import Optional, Union, Iterable, List


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

    def forward(self, state: paddle_quantum.State, decompose: Optional[bool] = False) -> Union[
        paddle.Tensor, List[paddle.Tensor]]:
        r"""Compute the expectation value of the observable with respect to the input state.

        The value computed by this function can be used as a loss function to optimize.

        Args:
            state: The input state which will be used to compute the expectation value.
            decompose: Defaults to ``False``.  If decompose is ``True``, it will return the expectation value of each term.

        Raises:
            NotImplementedError: The backend is wrong or not supported.

        Returns:
            The expectation value. If the backend is QuLeaf, it is computed by sampling.
        """
        if self.backend == Backend.QuLeaf:
            param_list = [history['param'] for history in filter(lambda x: 'param' in x, state.oper_history)]
            expec_val = quleaf.ExpecValOp.apply(
                state,
                self.hamiltonian,
                paddle.to_tensor([self.shots], dtype=paddle.get_default_dtype()),
                *param_list
            )
            return expec_val

        num_qubits = state.num_qubits
        float_dtype = _get_float_dtype(paddle_quantum.get_dtype())
        origin_seq = list(range(num_qubits))
        state_data = state.data
        if self.backend == Backend.StateVector:
            expec_val = paddle.zeros([state_data.reshape([-1, 2 ** num_qubits]).shape[0]], dtype=float_dtype)
        elif self.backend == Backend.DensityMatrix:
            expec_val = paddle.zeros([state_data.reshape([-1, 2 ** num_qubits, 2 ** num_qubits]).shape[0]],
                                     dtype=float_dtype)
        else:
            raise NotImplementedError
        expec_val_each_term = []
        for i in range(self.num_terms):
            pauli_site = self.sites[i]
            if pauli_site == ['']:
                expec_val += self.coeffs[i]
                expec_val_each_term.append(self.coeffs[i])
                continue
            num_applied_qubits = len(pauli_site)
            matrix = self.matrices[i]

            if self.backend == Backend.StateVector:
                output_state, seq_for_acted = state_vector.unitary_transformation_without_swapback(
                    state_data, [matrix], pauli_site, num_qubits, origin_seq)
                perm_map = _perm_of_list(seq_for_acted, origin_seq)
                output_state = _base_transpose(output_state, perm_map).reshape([-1, 2 ** num_qubits, 1])
                state_data_bra = paddle.conj(state_data.reshape([-1, 1, 2 ** num_qubits]))
                batch_values = paddle.squeeze(paddle.real(paddle.matmul(state_data_bra, output_state)), axis=[-2, -1]) * \
                               self.coeffs[i]
                expec_val_each_term.append(batch_values)
                expec_val += batch_values

            elif self.backend == Backend.DensityMatrix:
                seq_for_acted = pauli_site + [x for x in origin_seq if x not in pauli_site]
                perm_map = _perm_of_list(origin_seq, seq_for_acted)
                output_state = _base_transpose_for_dm(state_data, perm=perm_map).reshape(
                    [-1, 2 ** num_applied_qubits, 2 ** (2 * num_qubits - num_applied_qubits)])
                output_state = paddle.matmul(matrix, output_state).reshape(
                    [-1, 2 ** num_qubits, 2 ** num_qubits])
                batch_values = paddle.real(paddle.trace(output_state, axis1=-2, axis2=-1)) * self.coeffs[i]
                expec_val_each_term.append(batch_values)
                expec_val += batch_values

            else:
                raise NotImplementedError

        if decompose:
            return expec_val_each_term
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

            prob_array = prob_array / paddle.sum(prob_array)  # normalize calculation error
            if desired_result is None:
                return prob_array
            if isinstance(desired_result, str):
                desired_result = [desired_result]
                prob_array = paddle.concat([prob_array[int(res, base=2)] for res in desired_result])
            return prob_array
        else:
            raise NotImplementedError("Currently we just support the z basis.")
