# !/usr/bin/env python3
# Copyright (c) 2021 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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
Shadow sample module.
"""

import math
import numpy as np
import paddle
import re
import paddle_quantum
from paddle_quantum import Hamiltonian
from typing import List, Optional

__all__ = [
    "shadow_sample"
]


def shadow_sample(
        state: 'paddle_quantum.State', num_qubits: int, sample_shots: int, mode: paddle_quantum.Backend,
        hamiltonian: Optional[paddle_quantum.Hamiltonian] = None, method: str = 'CS'
) -> list:
    r"""Measure a given quantum state with random Pauli operators and return the measurement results.

    Args:
        state: Input quantum state, which is either a state vector or a density matrix.
        num_qubits: The number of qubits.
        sample_shots: The number of random samples.
        mode: Representation form of the input quantum state.
        hamiltonian: A ``Hamiltonian`` object representing the observable to be measured. Defaults to ``None``.
        method: Method for sampling random Pauli operators, which should be one of ``'CS'``, ``'LBCS'``, and ``'APS'``. Defaults to ``'CS'``.
    
    Raises:
        ValueError: Hamiltonian has a bad form
        NotImplementedError: The backend of ``state`` should be ``StateVector`` or ``DensityMatrix``


    Returns:
        Randomly chosen Pauli operators and their corresponding measurement result in a list of shape ``(sample_shots, 2)``.
    """
    if hamiltonian is not None:
        if isinstance(hamiltonian, Hamiltonian):
            hamiltonian = hamiltonian.pauli_str

    def prepare_hamiltonian(hamiltonian, num_qubits):
        new_hamiltonian = []
        for idx, (coeff, pauli_term) in enumerate(hamiltonian):
            pauli_term = re.split(r',\s*', pauli_term.lower())
            pauli_list = ['i'] * num_qubits
            for item in pauli_term:
                if len(item) > 1:
                    pauli_list[int(item[1:])] = item[0]
                elif item[0].lower() != 'i':
                    raise ValueError('Expecting I for ', item[0])
            new_term = [coeff, ''.join(pauli_list)]
            new_hamiltonian.append(new_term)
        return new_hamiltonian

    if hamiltonian is not None:
        hamiltonian = prepare_hamiltonian(hamiltonian, num_qubits)

    pauli2index = {'x': 0, 'y': 1, 'z': 2}

    def random_pauli_sample(num_qubits: int, beta: Optional[List[float]]=None):
        # assume beta obeys a uniform distribution if it is not given
        if beta is None:
            beta = []
            for _ in range(0, num_qubits):
                beta.append([1 / 3] * 3)
        pauli_sample = str()
        for qubit_idx in range(num_qubits):
            sample = np.random.choice(['x', 'y', 'z'], 1, p=beta[qubit_idx])
            pauli_sample += sample[0]
        return pauli_sample

    def measure_by_pauli_str(pauli_str, input_state, num_qubits, method):
        if method == "clifford":
            # Add the clifford function
            pass
        else:
            # Other method are transformed as follows
            # Convert to tensor form
            for qubit in range(num_qubits):
                if pauli_str[qubit] == 'x':
                    input_state = paddle_quantum.gate.functional.h(input_state, qubit, input_state.dtype, input_state.backend)
                elif pauli_str[qubit] == 'y':
                    input_state = paddle_quantum.gate.functional.s(input_state, qubit, input_state.dtype, input_state.backend)
                    input_state = paddle_quantum.gate.functional.z(input_state, qubit, input_state.dtype, input_state.backend)
                    input_state = paddle_quantum.gate.functional.h(input_state, qubit, input_state.dtype, input_state.backend)
            if input_state.backend == paddle_quantum.Backend.StateVector:
                data = input_state.data.numpy()
                prob_array = np.real(np.multiply(data, np.conj(data)))
            elif input_state.backend == paddle_quantum.Backend.DensityMatrix:
                data = input_state.data.numpy()
                prob_array = np.real(np.diag(data))
            else:
                raise NotImplementedError
            sample = np.random.choice(range(0, 2 ** num_qubits), size=1, p=prob_array)
            bit_string = bin(sample[0])[2:].zfill(num_qubits)
            return bit_string

    # Define the function used to update the beta of the LBCS algorithm
    def calculate_diagonal_product(pauli_str, beta):
        product = 1
        for qubit_idx in range(len(pauli_str)):
            if pauli_str[qubit_idx] != 'i':
                index = pauli2index[pauli_str[qubit_idx]]
                b = beta[qubit_idx][index]
                if b == 0:
                    return float('inf')
                else:
                    product *= b

        return 1 / product

    def lagrange_restriction_numerator(qubit_idx, hamiltonian, beta):
        tally = [0, 0, 0]
        for coeff, pauli_term in hamiltonian:
            if pauli_term[qubit_idx] == 'x':
                tally[0] += (coeff ** 2) * calculate_diagonal_product(pauli_term, beta)
            elif pauli_term[qubit_idx] == 'y':
                tally[1] += (coeff ** 2) * calculate_diagonal_product(pauli_term, beta)
            elif pauli_term[qubit_idx] == 'z':
                tally[2] += (coeff ** 2) * calculate_diagonal_product(pauli_term, beta)
        return tally

    def lagrange_restriction_denominator(qubit_idx, random_observable, beta):
        tally = 0.0
        for coeff, pauli_term in random_observable:
            if pauli_term[qubit_idx] != "i":
                tally += (coeff ** 2) * calculate_diagonal_product(pauli_term, beta)
        if tally == 0.0:
            tally = 1
        return tally

    def lagrange_restriction(qubit_idx, hamiltonian, beta, denominator=None):
        numerator = lagrange_restriction_numerator(qubit_idx, hamiltonian, beta)
        if denominator is None:
            denominator = lagrange_restriction_denominator(qubit_idx, hamiltonian, beta)
        return [item / denominator for item in numerator]

    def beta_distance(beta1, beta2):
        two_norm_squared = 0.0
        for qubit in range(len(beta1)):
            two_norm_squared_qubit = np.sum((np.array(beta1[qubit]) - np.array(beta2[qubit])) ** 2)
            two_norm_squared += two_norm_squared_qubit
        return np.sqrt(two_norm_squared)

    def update_beta_in_lbcs(hamiltonian, num_qubit, beta_old=None, weight=0.1):
        if beta_old is None:
            beta_old = list()
            for _ in range(0, num_qubit):
                beta_old.append([1 / 3] * 3)

        beta_new = list()
        for qubit in range(num_qubit):
            denominator = lagrange_restriction_denominator(qubit, hamiltonian, beta_old)
            lagrange_rest = lagrange_restriction(qubit, hamiltonian, beta_old, denominator)
            beta_new.append(lagrange_rest)
            if sum(beta_new[qubit]) != 0:
                beta_new[qubit] = [item / sum(beta_new[qubit]) for item in beta_new[qubit]]
            else:
                beta_new[qubit] = beta_old[qubit]
            for idx in range(len(beta_new[qubit])):
                beta_new[qubit][idx] = (1 - weight) * beta_old[qubit][idx] + weight * beta_new[qubit][idx]
        return beta_new, beta_distance(beta_new, beta_old)

    # Define the function used to update the beta of the APS algorithm
    def in_omega(pauli_str, qubit_idx, qubit_shift, base_shift):
        if pauli_str[qubit_shift[qubit_idx]] == 'i':
            return False
        for former_qubit in range(qubit_idx):
            idx = qubit_shift[former_qubit]
            if not pauli_str[idx] in ('i', base_shift[former_qubit]):
                return False
        return True

    def update_in_aps(qubit_idx, qubits_shift, bases_shift, hamiltonian):
        constants = [0.0, 0.0, 0.0]
        for coeff, pauli_term in hamiltonian:
            if in_omega(pauli_term, qubit_idx, qubits_shift, bases_shift):
                pauli = pauli_term[qubits_shift[qubit_idx]]
                index = pauli2index[pauli]
                constants[index] += coeff ** 2
        beta_sqrt = np.sqrt(constants)
        # The beta may be zero, use a judgment statement to avoid
        if np.sum(beta_sqrt) == 0.0:
            beta = [1 / 3, 1 / 3, 1 / 3]
        else:
            beta = beta_sqrt / np.sum(beta_sqrt)
        return beta

    def single_random_pauli_sample_in_aps(qubit_idx, qubits_shift, pauli_str_shift, hamiltonian):
        assert len(pauli_str_shift) == qubit_idx
        beta = update_in_aps(qubit_idx, qubits_shift, pauli_str_shift, hamiltonian)
        single_pauli = np.random.choice(['x', 'y', 'z'], p=beta)
        return single_pauli

    def random_pauli_sample_in_aps(hamiltonian):
        num_qubits = len(hamiltonian[0][1])
        # The qubits_shift is used to ignore the order of the qubits
        qubits_shift = list(np.random.choice(range(num_qubits), size=num_qubits, replace=False))
        pauli_str_shift = []
        for qubit_idx in range(num_qubits):
            single_pauli = single_random_pauli_sample_in_aps(qubit_idx, qubits_shift, pauli_str_shift, hamiltonian)
            pauli_str_shift.append(single_pauli)
        pauli_sample = str()
        for i in range(num_qubits):
            j = qubits_shift.index(i)
            # The qubits_shift.index(i) sorts the qubits in order
            pauli_sample = pauli_sample + pauli_str_shift[j]
        return pauli_sample

    sample_result = []
    if method == "CS":
        for _ in range(sample_shots):
            random_pauli_str = random_pauli_sample(num_qubits, beta=None)
            measurement_result = measure_by_pauli_str(random_pauli_str, state, num_qubits, method)
            sample_result.append((random_pauli_str, measurement_result))
        return sample_result
    elif method == "LBCS":
        beta = []
        for _ in range(0, num_qubits):
            beta.append([1 / 3] * 3)
        beta_opt_iter_num = 10000
        distance_limit = 1.0e-6
        for _ in range(beta_opt_iter_num):
            beta_opt, distance = update_beta_in_lbcs(hamiltonian, num_qubits, beta)
            beta = beta_opt
            if distance < distance_limit:
                break
        sample_result = []
        for _ in range(sample_shots):
            random_pauli_str = random_pauli_sample(num_qubits, beta)
            measurement_result = measure_by_pauli_str(random_pauli_str, state, num_qubits, method)
            sample_result.append((random_pauli_str, measurement_result))
        return sample_result, beta
    elif method == "APS":
        for _ in range(sample_shots):
            random_pauli_str = random_pauli_sample_in_aps(hamiltonian)
            measurement_result = measure_by_pauli_str(random_pauli_str, state, num_qubits, method)
            sample_result.append((random_pauli_str, measurement_result))
        return sample_result
